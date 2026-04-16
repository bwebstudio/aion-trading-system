"""
research/run_sequential_discovery.py
─────────────────────────────────────
Run sequential (temporal) edge discovery on one asset.

Pipeline
────────
  1. Load bars + build snapshots for the selected asset.
  2. Build the compact feature matrix with forward_return_10 + bin cols.
  3. Run SequenceGenerator (Apriori-style level-wise, max_length=N).
  4. Print top-20 sequences and write JSON output.

Output
──────
    research/output/sequential_edges.json

CLI example
───────────
    python research/run_sequential_discovery.py \\
        --asset xauusd \\
        --bars-limit 120000 \\
        --snapshots 5000 \\
        --max-length 3 \\
        --min-samples 100
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
from research.pattern_discovery.feature_builder import BIN_COLUMNS
from research.sequential_discovery import SequenceGenerator

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


DEFAULT_OUTPUT = _ROOT / "research" / "output" / "sequential_edges.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AION Sequential Edge Discovery"
    )
    p.add_argument("--asset", type=str, default="us100",
                   choices=sorted(ASSET_REGISTRY.keys()))
    p.add_argument("--snapshots", type=int, default=3000)
    p.add_argument("--bars-limit", type=int, default=None)
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument("--max-length", type=int, default=3, choices=(2, 3, 4))
    p.add_argument("--min-samples", type=int, default=100)
    p.add_argument("--minimal-edge-threshold", type=float, default=0.0002)
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--top-n", type=int, default=20,
                   help="How many top sequences to print to stdout.")
    p.add_argument("--max-output", type=int, default=1000,
                   help="Cap sequences written to JSON (sorted by |score|).")
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    return p.parse_args()


def _result_to_dict(r) -> dict:
    return {
        "sequence": [list(step) for step in r.sequence],
        "description": r.description,
        "length": r.length,
        "n_samples": r.n_samples,
        "mean_return": float(r.mean_return),
        "expectancy": float(r.expectancy),
        "profit_factor": (
            float(r.profit_factor) if r.profit_factor is not None else None
        ),
        "winrate": float(r.winrate),
        "score": float(r.score),
        "train_n": r.train_n,
        "test_n": r.test_n,
        "train_mean": (
            float(r.train_mean) if r.train_mean is not None else None
        ),
        "test_mean": (
            float(r.test_mean) if r.test_mean is not None else None
        ),
        "stability": (
            float(r.stability) if r.stability is not None else None
        ),
    }


def main() -> None:
    args = _parse_args()
    asset = args.asset.lower()

    print()
    print("AION Sequential Discovery")
    print("=" * 72)
    print(
        f"  asset={asset}   max-length={args.max_length}   "
        f"min-samples={args.min_samples}   "
        f"edge-threshold={args.minimal_edge_threshold}   "
        f"snapshots<={args.snapshots}"
    )
    print()

    t_total = time.perf_counter()
    csv_name, spec_factory = ASSET_REGISTRY[asset]
    instrument = spec_factory()
    csv_path = DATA_DIR / csv_name
    if not csv_path.exists():
        print(f"ERROR: data file not found: {csv_path}")
        return

    # ── 1. Bars + snapshots ─────────────────────────────────────────────────
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
        f"snapshots={len(snapshots):,}  "
        f"({time.perf_counter() - t:.1f}s)"
    )
    if not snapshots:
        print("No snapshots; aborting.")
        return

    # ── 2. Compact matrix ──────────────────────────────────────────────────
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

    # ── 3. Sequence discovery ──────────────────────────────────────────────
    t = time.perf_counter()
    generator = SequenceGenerator(
        min_samples=args.min_samples,
        minimal_edge_threshold=args.minimal_edge_threshold,
        max_length=args.max_length,
        train_fraction=args.train_fraction,
    )
    results = generator.discover(df, BIN_COLUMNS)
    print(
        f"  discovered: {len(results):,} sequences  "
        f"({time.perf_counter() - t:.1f}s)"
    )

    # ── 4. Report ──────────────────────────────────────────────────────────
    if not results:
        print()
        print("No sequences passed the filters.")
        return

    print()
    print(f"Top {min(args.top_n, len(results))} sequences by |score|")
    print("-" * 72)
    for r in results[: args.top_n]:
        print(f"  {r.describe()}")

    # ── 5. JSON output ─────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = results[: args.max_output]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asset": instrument.symbol,
        "params": {
            "max_length": args.max_length,
            "min_samples": args.min_samples,
            "minimal_edge_threshold": args.minimal_edge_threshold,
            "train_fraction": args.train_fraction,
            "snapshots": args.snapshots,
            "bars_limit": args.bars_limit,
            "start_date": args.start_date,
            "end_date": args.end_date,
        },
        "n_sequences_total": len(results),
        "n_sequences_kept": len(kept),
        "sequences": [_result_to_dict(r) for r in kept],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"wrote {len(kept)} sequence(s) → {output_path}")
    print(f"total runtime: {time.perf_counter() - t_total:.1f}s")
    print()


if __name__ == "__main__":
    main()
