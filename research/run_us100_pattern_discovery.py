"""
research/run_us100_pattern_discovery.py
────────────────────────────────────────
US100-only pattern-edge discovery.

The existing strategy_candidates.json was produced from BTC + XAU
cross-asset validation; few of those patterns generalise to US100.
This runner re-discovers candidates fitted purely to US100:

    1. Load US100 bars + build the compact feature matrix.
    2. Generate candidate patterns level-wise (1, 2, 3 conditions).
    3. ForwardTester returns top-K survivors with stability-checked
       train/test splits.
    4. Convert to StrategyCandidate, backtest each on the same df.
    5. Apply USER filters:
           total_trades   >= 50
           profit_factor  >= 1.25
           expectancy     > 0
    6. Save survivors to:
           research/output/strategy_candidates_us100.json
       in the format the meta-strategy runner expects, i.e.

           {"single_asset": {"US100.cash": [...candidate_to_dict...]}}

CLI example
───────────
    python research/run_us100_pattern_discovery.py \\
        --bars-limit 120000 --snapshots 5000 \\
        --top-discovery 1000 \\
        --min-trades 50 --min-profit-factor 1.25
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aion.execution.execution_model import ExecutionModel

from research.pattern_discovery import (
    FeatureBuilder,
    ForwardTester,
    PatternGenerator,
)
from research.pattern_discovery.feature_builder import BIN_COLUMNS
from research.pattern_strategies import (
    backtest_candidate,
    candidate_to_dict,
    convert_compact_result,
)

# Reuse multi-asset runner helpers.
from research.run_pattern_discovery import (
    ASSET_REGISTRY,
    DATA_DIR,
    EXEC_CONFIG,
    FORWARD_BARS,
    SIGMA_WINDOW,
    _filter_bars_early,
    _load_bars,
    _parse_date,
)
from scripts.run_us100_replay import _build_snapshots


DEFAULT_OUTPUT = (
    _ROOT / "research" / "output" / "strategy_candidates_us100.json"
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="US100 pattern-edge discovery")
    p.add_argument("--snapshots", type=int, default=5000)
    p.add_argument("--bars-limit", type=int, default=120_000)
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date", type=str, default=None)
    # Discovery params
    p.add_argument("--max-conditions", type=int, default=3)
    p.add_argument("--min-samples", type=int, default=100)
    p.add_argument("--minimal-edge-threshold", type=float, default=0.00005,
                   help="Loose: lots of candidates surface, filter happens later.")
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--top-discovery", type=int, default=1000,
                   help="How many top patterns to backtest after discovery.")
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--stability-tolerance", type=float, default=0.50)
    # User filters (applied AFTER per-candidate backtest)
    p.add_argument("--min-trades", type=int, default=50)
    p.add_argument("--min-profit-factor", type=float, default=1.25)
    p.add_argument("--min-expectancy", type=float, default=0.0)
    # Execution defaults
    p.add_argument("--stop-mult", type=float, default=1.5)
    p.add_argument("--tp-mult", type=float, default=2.5)
    p.add_argument("--max-hold", type=int, default=20)
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--top-print", type=int, default=20)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()

    print()
    print("AION US100 Pattern Discovery")
    print("=" * 72)
    print(
        f"  bars-limit={args.bars_limit}  snapshots<={args.snapshots}  "
        f"max-cond={args.max_conditions}  min-samples={args.min_samples}  "
        f"top-discovery={args.top_discovery}"
    )
    print(
        f"  filters: trades>={args.min_trades}  "
        f"pf>={args.min_profit_factor}  expectancy>{args.min_expectancy}"
    )
    print()

    csv_name, spec_factory = ASSET_REGISTRY["us100"]
    instrument = spec_factory()
    csv_path = DATA_DIR / csv_name
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        return

    t_total = time.perf_counter()
    execution_model = ExecutionModel.from_config(EXEC_CONFIG)

    # ── 1. Bars + snapshots ────────────────────────────────────────────────
    t = time.perf_counter()
    bars = _load_bars(csv_path, instrument)
    raw_total = len(bars)
    bars = _filter_bars_early(
        bars,
        start=_parse_date(args.start_date),
        end=_parse_date(args.end_date),
        bars_limit=args.bars_limit,
    )
    raw_after = len(bars)
    snapshots = _build_snapshots(bars, instrument)
    del bars
    if len(snapshots) > args.snapshots:
        snapshots = snapshots[-args.snapshots:]
    print(
        f"  bars: raw={raw_total:,}  after_filter={raw_after:,}  "
        f"snapshots={len(snapshots):,}  ({time.perf_counter() - t:.1f}s)"
    )
    if not snapshots:
        print("No snapshots; aborting.")
        return

    # ── 2. Compact matrix ──────────────────────────────────────────────────
    t = time.perf_counter()
    fb = FeatureBuilder(compression_lookback=10, sigma_window=SIGMA_WINDOW)
    df = fb.build_compact_matrix(
        snapshots,
        forward_bars=FORWARD_BARS,
        execution_model=execution_model,
    )
    del snapshots
    print(
        f"  compact matrix: {len(df):,} rows  "
        f"({time.perf_counter() - t:.1f}s)"
    )

    # ── 3. Level-wise pattern generation ───────────────────────────────────
    t = time.perf_counter()
    gen = PatternGenerator(
        min_bucket_occurrences=args.min_samples,
        max_conditions=args.max_conditions,
        minimal_edge_threshold=args.minimal_edge_threshold,
    )
    keys: list = []
    for batch in gen.generate_patterns_levelwise(
        df,
        bin_columns=BIN_COLUMNS,
        batch_size=args.batch_size,
        min_samples=args.min_samples,
        max_conditions=args.max_conditions,
        minimal_edge_threshold=args.minimal_edge_threshold,
    ):
        keys.extend(batch)
    print(
        f"  candidate patterns generated: {len(keys):,}  "
        f"({time.perf_counter() - t:.1f}s)"
    )

    if not keys:
        print("No candidate patterns generated; aborting.")
        return

    # ── 4. Forward-test → top-K ────────────────────────────────────────────
    t = time.perf_counter()
    tester = ForwardTester(
        execution_model=execution_model,
        forward_bars=FORWARD_BARS,
        train_fraction=args.train_fraction,
        min_samples_train=max(50, args.min_samples // 2),
        min_samples_test=max(25, args.min_samples // 4),
        stability_tolerance=args.stability_tolerance,
    )
    raw_results = tester.evaluate_patterns(
        df,
        iter(keys),
        batch_size=args.batch_size,
        top_k=args.top_discovery,
        min_samples=args.min_samples,
        train_fraction=args.train_fraction,
        min_samples_train=max(50, args.min_samples // 2),
        min_samples_test=max(25, args.min_samples // 4),
        stability_tolerance=args.stability_tolerance,
        progress_every=0,
    )
    print(
        f"  forward-test survivors (top-K): {len(raw_results):,}  "
        f"({time.perf_counter() - t:.1f}s)"
    )

    # ── 5. Convert + backtest + filter ─────────────────────────────────────
    t = time.perf_counter()
    survivors: list[tuple] = []
    rejected_counts = {"trades": 0, "expectancy": 0, "pf": 0, "convert": 0}
    for r in raw_results:
        cand = convert_compact_result(
            r,
            stop_mult=args.stop_mult,
            tp_mult=args.tp_mult,
            max_hold=args.max_hold,
        )
        if cand is None:
            rejected_counts["convert"] += 1
            continue
        rep = backtest_candidate(df, cand)
        if rep.total_trades < args.min_trades:
            rejected_counts["trades"] += 1
            continue
        if rep.expectancy <= args.min_expectancy:
            rejected_counts["expectancy"] += 1
            continue
        if rep.profit_factor is None or rep.profit_factor < args.min_profit_factor:
            rejected_counts["pf"] += 1
            continue
        survivors.append((cand, rep))
    print(
        f"  backtest filter: {len(survivors):,} survivors  "
        f"(rejected: {rejected_counts})  "
        f"({time.perf_counter() - t:.1f}s)"
    )

    if not survivors:
        print()
        print("No candidates survived the filter.")
        return

    # ── 6. Rank by expectancy * sqrt(trades) ──────────────────────────────
    survivors.sort(
        key=lambda cr: cr[1].expectancy * math.sqrt(max(cr[1].total_trades, 1)),
        reverse=True,
    )

    # ── 7. Print top-N ────────────────────────────────────────────────────
    print()
    print(f"Top {min(args.top_print, len(survivors))} US100 patterns")
    print("-" * 100)
    print(
        f"  {'rank':>4}  {'dir':<5}  {'trades':>6}  {'wr':>6}  {'pf':>5}  "
        f"{'exp':>10}  {'dd':>9}  pattern"
    )
    print("-" * 100)
    for i, (cand, rep) in enumerate(survivors[: args.top_print], start=1):
        pf = f"{rep.profit_factor:5.2f}" if rep.profit_factor else "  n/a"
        print(
            f"  {i:>4}  {cand.direction:<5}  "
            f"{rep.total_trades:>6}  "
            f"{rep.winrate * 100:5.1f}%  "
            f"{pf}  "
            f"{rep.expectancy:+.5f}  "
            f"{rep.max_drawdown:+.4f}  "
            f"{cand.description}"
        )

    # ── 8. JSON output (run_meta_strategy-compatible) ─────────────────────
    asset_symbol = instrument.symbol
    cand_dicts = []
    for cand, rep in survivors:
        d = candidate_to_dict(cand)
        d["backtest"] = {
            "total_trades": rep.total_trades,
            "winrate": float(rep.winrate),
            "expectancy": float(rep.expectancy),
            "avg_return": float(rep.avg_return),
            "profit_factor": (
                float(rep.profit_factor) if rep.profit_factor is not None else None
            ),
            "max_drawdown": float(rep.max_drawdown),
            "total_return": float(rep.total_return),
        }
        cand_dicts.append(d)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asset": asset_symbol,
        "discovery_params": {
            "bars_limit": args.bars_limit,
            "snapshots": args.snapshots,
            "max_conditions": args.max_conditions,
            "min_samples": args.min_samples,
            "minimal_edge_threshold": args.minimal_edge_threshold,
            "top_discovery": args.top_discovery,
            "train_fraction": args.train_fraction,
            "stability_tolerance": args.stability_tolerance,
        },
        "filter_thresholds": {
            "min_trades": args.min_trades,
            "min_profit_factor": args.min_profit_factor,
            "min_expectancy": args.min_expectancy,
        },
        "execution_defaults": {
            "stop_rule": {
                "type": "atr_multiplier",
                "period": 14,
                "multiplier": args.stop_mult,
            },
            "exit_rule": {
                "take_profit": {
                    "type": "atr_multiplier",
                    "period": 14,
                    "multiplier": args.tp_mult,
                },
                "max_hold_bars": args.max_hold,
            },
        },
        # Same shape that run_meta_strategy.py reads.
        "cross_asset": [],
        "single_asset": {asset_symbol: cand_dicts},
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"  candidates_discovered      : {len(survivors)}")
    print(f"  patterns generated         : {len(keys):,}")
    print(f"  forward-test top-K kept    : {len(raw_results):,}")
    print(f"  rejected (trades/exp/pf)   : {rejected_counts}")
    print(f"  wrote -> {out_path}")
    print(f"  total runtime: {time.perf_counter() - t_total:.1f}s")
    print()


if __name__ == "__main__":
    main()
