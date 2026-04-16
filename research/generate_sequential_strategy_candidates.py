"""
research/generate_sequential_strategy_candidates.py
────────────────────────────────────────────────────
Convert a `research/output/sequential_edges.json` file into executable
SequentialStrategyCandidates, optionally backtest each one, and write:

    research/output/sequential_strategy_candidates.json

Pipeline
────────
  1. Load sequential_edges.json (produced by
     research/run_sequential_discovery.py).
  2. Take the top-N sequences (by |score|).
  3. Map each to a SequentialStrategyCandidate.
  4. (Optional) build the compact matrix for the asset and run
     backtest_sequential_candidate on every candidate for enrichment.
  5. Sort candidates by |score| and write to JSON.

Filtering (optional)
────────────────────
    --min-trades              (default 50)
    --min-profit-factor       (default 1.3)
    --max-dd-abs              (default 0.05)
    --min-expectancy          (default 0.0, strictly >)

Disable the standalone backtest entirely with `--no-backtest` when you
just want to emit the candidates with discovery-time metrics only.

CLI example
───────────
    python research/generate_sequential_strategy_candidates.py \\
        --input  research/output/sequential_edges.json \\
        --output research/output/sequential_strategy_candidates.json \\
        --top-candidates 100 \\
        --asset xauusd \\
        --bars-limit 120000 --snapshots 5000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aion.execution.execution_model import ExecutionModel

from research.pattern_discovery import FeatureBuilder
from research.sequential_strategies import (
    backtest_sequential_candidate,
    candidate_to_dict,
    convert_sequence_dict,
)

# Reuse multi-asset runner helpers (instrument specs, loader, filter).
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


DEFAULT_INPUT = _ROOT / "research" / "output" / "sequential_edges.json"
DEFAULT_OUTPUT = _ROOT / "research" / "output" / "sequential_strategy_candidates.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert sequential edges into SequentialStrategyCandidates"
    )
    p.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument(
        "--asset",
        type=str,
        default=None,
        help="Asset key (us100|xauusd|btcusd).  If omitted, uses the "
             "asset stored in the input JSON.",
    )
    p.add_argument("--top-candidates", type=int, default=100)
    p.add_argument("--no-backtest", action="store_true",
                   help="Skip standalone backtest enrichment.")
    p.add_argument("--snapshots", type=int, default=3000)
    p.add_argument("--bars-limit", type=int, default=None)
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date", type=str, default=None)
    # Filter thresholds
    p.add_argument("--min-trades", type=int, default=50)
    p.add_argument("--min-profit-factor", type=float, default=1.3)
    p.add_argument("--max-dd-abs", type=float, default=0.05)
    p.add_argument("--min-expectancy", type=float, default=0.0)
    # Execution defaults (match pattern-strategy defaults for uniformity)
    p.add_argument("--stop-mult", type=float, default=1.5)
    p.add_argument("--tp-mult", type=float, default=2.5)
    p.add_argument("--max-hold", type=int, default=20)
    return p.parse_args()


def _asset_key_for_symbol(symbol: str) -> str | None:
    """Reverse-lookup the CLI asset key from an InstrumentSpec.symbol."""
    for key, (_csv, spec_factory) in ASSET_REGISTRY.items():
        if spec_factory().symbol == symbol:
            return key
    return None


def _build_compact_df(
    asset_key: str,
    args: argparse.Namespace,
):
    csv_name, spec_factory = ASSET_REGISTRY[asset_key]
    instrument = spec_factory()
    csv_path = DATA_DIR / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

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
        return None, instrument

    execution_model = ExecutionModel.from_config(EXEC_CONFIG)
    fb = FeatureBuilder(compression_lookback=10, sigma_window=SIGMA_WINDOW)
    df = fb.build_compact_matrix(
        snapshots,
        forward_bars=FORWARD_BARS,
        execution_model=execution_model,
    )
    return df, instrument


def main() -> None:
    args = _parse_args()

    print()
    print("AION Sequential Strategy Candidate Generator")
    print("=" * 72)

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}")
        print("       Run research/run_sequential_discovery.py first.")
        return
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    sequences_in = payload.get("sequences", [])
    source_asset = payload.get("asset")
    print(
        f"  input   : {in_path}  "
        f"(asset={source_asset}, sequences={len(sequences_in)})"
    )

    if not sequences_in:
        print("No sequences in input; nothing to do.")
        return

    # Resolve the asset key for the compact-matrix build.
    asset_key = args.asset
    if asset_key is None and source_asset is not None:
        asset_key = _asset_key_for_symbol(source_asset)
    if asset_key is None:
        print("ERROR: cannot resolve asset key; pass --asset explicitly.")
        return
    if asset_key not in ASSET_REGISTRY:
        print(f"ERROR: unknown asset '{asset_key}'.")
        return
    print(f"  asset   : {asset_key}  → {ASSET_REGISTRY[asset_key][0]}")

    # ── Convert top-N by |score| ────────────────────────────────────────────
    sequences_in.sort(
        key=lambda s: abs(float(s.get("score", 0.0))), reverse=True
    )
    converted = []
    for s in sequences_in[: args.top_candidates]:
        cand = convert_sequence_dict(
            s,
            stop_mult=args.stop_mult,
            tp_mult=args.tp_mult,
            max_hold=args.max_hold,
        )
        if cand is not None:
            converted.append(cand)
    print(
        f"  converted: {len(converted)} of "
        f"{min(args.top_candidates, len(sequences_in))} "
        f"(rejected: zero-mean)"
    )

    if not converted:
        print("No candidates to emit.")
        return

    # ── Optional backtest enrichment + filtering ───────────────────────────
    backtest_reports: dict[str, dict] = {}
    survivors = converted
    if not args.no_backtest:
        try:
            df, instrument = _build_compact_df(asset_key, args)
        except FileNotFoundError as exc:
            print(f"  [skip backtest] {exc}")
            df, instrument = None, None

        if df is None or df.empty:
            print("  [skip backtest] no compact matrix")
        else:
            t = time.perf_counter()
            filtered: list = []
            for cand in converted:
                rep = backtest_sequential_candidate(df, cand)
                backtest_reports[cand.name] = {
                    "total_trades": rep.total_trades,
                    "winrate": float(rep.winrate),
                    "expectancy": float(rep.expectancy),
                    "avg_return": float(rep.avg_return),
                    "profit_factor": (
                        float(rep.profit_factor)
                        if rep.profit_factor is not None
                        else None
                    ),
                    "max_drawdown": float(rep.max_drawdown),
                    "total_return": float(rep.total_return),
                }
                # Apply filters
                if rep.total_trades < args.min_trades:
                    continue
                if rep.expectancy <= args.min_expectancy:
                    continue
                pf = rep.profit_factor
                if pf is None or pf < args.min_profit_factor:
                    continue
                if abs(rep.max_drawdown) >= args.max_dd_abs:
                    continue
                filtered.append(cand)
            survivors = filtered
            print(
                f"  backtested: {len(converted)}  "
                f"survivors: {len(survivors)}  "
                f"({time.perf_counter() - t:.1f}s)"
            )

    # ── Serialise ───────────────────────────────────────────────────────────
    candidates_out = []
    for cand in survivors:
        d = candidate_to_dict(cand)
        if cand.name in backtest_reports:
            d["backtest"] = backtest_reports[cand.name]
        candidates_out.append(d)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_input": str(in_path),
        "asset": source_asset or asset_key,
        "execution_defaults": {
            "stop_rule": {
                "type": "atr_multiplier", "period": 14,
                "multiplier": args.stop_mult,
            },
            "exit_rule": {
                "take_profit": {
                    "type": "atr_multiplier", "period": 14,
                    "multiplier": args.tp_mult,
                },
                "max_hold_bars": args.max_hold,
            },
        },
        "filter_thresholds": (
            None if args.no_backtest else {
                "min_trades": args.min_trades,
                "min_profit_factor": args.min_profit_factor,
                "min_expectancy": args.min_expectancy,
                "max_drawdown_abs": args.max_dd_abs,
            }
        ),
        "n_candidates_before_filter": len(converted),
        "n_candidates_after_filter": len(candidates_out),
        "candidates": candidates_out,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print()
    print(f"  candidates_before_filter : {len(converted)}")
    print(f"  candidates_after_filter  : {len(candidates_out)}")
    print(f"  wrote → {out_path}")
    print()


if __name__ == "__main__":
    main()
