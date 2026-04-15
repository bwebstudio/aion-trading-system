"""
research/run_meta_strategy.py
──────────────────────────────
Run the meta-strategy backtest over one asset.

Pipeline
────────
  1. Load StrategyCandidate list from strategy_candidates.json
     (by default: research/output/strategy_candidates.json).
  2. Build a compact feature matrix for the selected asset.
  3. Classify every row into a regime (rule-based).
  4. StrategySelector picks the top-1 candidate per regime.
  5. Bar-replay backtest activates the selected candidate on its
     matching rows only.
  6. Print a clear per-regime + per-strategy summary.

CLI
───
    python research/run_meta_strategy.py
        [--asset us100|xauusd|btcusd]
        [--candidates research/output/strategy_candidates.json]
        [--source cross_asset | single_asset]
        [--snapshots 3000]
        [--bars-limit N | --start-date YYYY-MM-DD --end-date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aion.execution.execution_model import ExecutionModel

from research.meta_strategy import CandidateFilter, backtest_meta
from research.pattern_discovery import FeatureBuilder
from research.pattern_strategies import candidate_from_dict

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


DEFAULT_CANDIDATES = _ROOT / "research" / "output" / "strategy_candidates.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AION Meta-Strategy backtest")
    p.add_argument("--asset", type=str, default="us100",
                   choices=sorted(ASSET_REGISTRY.keys()))
    p.add_argument("--candidates", type=str, default=str(DEFAULT_CANDIDATES))
    p.add_argument(
        "--source",
        choices=("cross_asset", "single_asset", "auto"),
        default="auto",
        help="Which candidate list to use from the JSON.  'auto' tries "
             "cross_asset first, falls back to the asset's single_asset list.",
    )
    p.add_argument("--snapshots", type=int, default=3000)
    p.add_argument("--bars-limit", type=int, default=None)
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument("--top-k-per-regime", type=int, default=1)
    # Pre-filter thresholds
    p.add_argument("--no-prefilter", action="store_true",
                   help="Skip the pre-filter stage (pass all candidates through).")
    p.add_argument("--filter-min-trades", type=int, default=50)
    p.add_argument("--filter-min-profit-factor", type=float, default=1.3)
    p.add_argument("--filter-max-dd-abs", type=float, default=0.05,
                   help="|max_drawdown| must be strictly less than this.")
    # Selector NO_TRADE quality thresholds
    p.add_argument("--selector-min-pf", type=float, default=1.3,
                   help="If top candidate's profit_factor is below this in a "
                        "regime, the selector returns NO_TRADE.")
    p.add_argument("--selector-min-expectancy", type=float, default=0.0,
                   help="If top candidate's expectancy <= this, NO_TRADE.")
    return p.parse_args()


def _load_candidates(path: Path, asset_symbol: str, source: str) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    cross = data.get("cross_asset", [])
    single = data.get("single_asset", {}).get(asset_symbol, [])

    if source == "cross_asset":
        selected = cross
    elif source == "single_asset":
        selected = single
    else:  # auto
        selected = cross if cross else single

    return [candidate_from_dict(d) for d in selected], len(cross), len(single)


def main() -> None:
    args = _parse_args()
    asset = args.asset.lower()

    print()
    print("AION Meta-Strategy Backtest")
    print("=" * 72)
    print(f"  asset    = {asset}")
    print(f"  source   = {args.source}")
    print(f"  snapshots<= {args.snapshots}")
    print()

    csv_name, spec_factory = ASSET_REGISTRY[asset]
    instrument = spec_factory()
    csv_path = DATA_DIR / csv_name
    if not csv_path.exists():
        print(f"ERROR: data file not found: {csv_path}")
        return

    # ── Candidates ──────────────────────────────────────────────────────────
    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"ERROR: candidates JSON not found: {candidates_path}")
        print("       Run research/generate_strategy_candidates.py first.")
        return
    candidates, n_cross, n_single = _load_candidates(
        candidates_path, instrument.symbol, args.source
    )
    print(
        f"  candidates: cross={n_cross} single({instrument.symbol})={n_single}"
        f"  → using {len(candidates)}"
    )
    if not candidates:
        print("No candidates to evaluate.")
        return

    # ── Bars + snapshots + compact matrix ──────────────────────────────────
    t0 = time.perf_counter()
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
        f"snapshots_built={len(snapshots):,}  "
        f"({time.perf_counter() - t0:.1f}s)"
    )
    if not snapshots:
        print("No snapshots; aborting.")
        return

    t = time.perf_counter()
    execution_model = ExecutionModel.from_config(EXEC_CONFIG)
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

    # ── Meta backtest (with pre-filter) ────────────────────────────────────
    filt = CandidateFilter(
        min_trades=args.filter_min_trades,
        min_profit_factor=args.filter_min_profit_factor,
        min_expectancy=0.0,
        max_drawdown_abs=args.filter_max_dd_abs,
    )
    t = time.perf_counter()
    report = backtest_meta(
        df,
        candidates,
        top_k_per_regime=args.top_k_per_regime,
        pre_filter=filt,
        apply_prefilter=not args.no_prefilter,
        selector_min_profit_factor=args.selector_min_pf,
        selector_min_expectancy=args.selector_min_expectancy,
    )
    print(f"  meta backtest: {time.perf_counter() - t:.1f}s")

    # ── Report ─────────────────────────────────────────────────────────────
    print()
    print("Pre-filter")
    print("-" * 72)
    print(f"  thresholds            : min_trades>={filt.min_trades}, "
          f"pf>={filt.min_profit_factor}, expectancy>0, "
          f"|dd|<{filt.max_drawdown_abs}")
    print(f"  candidates_before_filter : {report.candidates_before_filter}")
    print(f"  candidates_after_filter  : {report.candidates_after_filter}")
    if report.prefilter_rejections:
        # Top rejection reasons — concise breakdown.
        from collections import Counter
        reason_counts = Counter(
            r for r in report.prefilter_rejections.values() if r != "ok"
        )
        if reason_counts:
            print("  rejection breakdown   :")
            for reason, c in reason_counts.most_common():
                print(f"    {c:>4}  {reason}")

    print()
    print("Summary")
    print("-" * 72)
    for line in report.summary_lines():
        print(line)

    print()
    print(f"Total runtime: {time.perf_counter() - t0:.1f}s")
    print()


if __name__ == "__main__":
    main()
