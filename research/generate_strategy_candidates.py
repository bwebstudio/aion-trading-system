"""
research/generate_strategy_candidates.py
─────────────────────────────────────────
Convert top discovered patterns into StrategyCandidate JSON.

Pipeline
────────
  1. Run multi-asset pattern discovery (reuses the v4 runner helpers)
  2. Collect the top-k cross-asset survivors
  3. Map each survivor through convert_multi_asset()
  4. Write research/output/strategy_candidates.json

Additionally emits per-asset top-k as a supplementary list (also in the
same file, under the key "single_asset") so downstream consumers can
inspect edges that appear on a single symbol.

CLI
───
  python research/generate_strategy_candidates.py
      --assets us100,xauusd,btcusd
      [--top-candidates 50]
      [--output research/output/strategy_candidates.json]
      (plus all CLI flags supported by run_pattern_discovery.py)
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

from research.pattern_discovery import validate_across_assets
from research.pattern_strategies import (
    candidate_to_dict,
    convert_compact_result,
    convert_multi_asset,
)

# Reuse the multi-asset discovery helpers.
from research.run_pattern_discovery import (
    ASSET_REGISTRY,
    EXEC_CONFIG,
    _discover_asset,
)


DEFAULT_OUTPUT = _ROOT / "research" / "output" / "strategy_candidates.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert discovered patterns into StrategyCandidate JSON"
    )
    p.add_argument("--assets", type=str, default="us100,xauusd,btcusd")
    p.add_argument("--max-patterns", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--top-k", type=int, default=500)
    p.add_argument("--min-samples", type=int, default=100)
    p.add_argument("--snapshots", type=int, default=3000)
    p.add_argument("--min-assets", type=int, default=2)
    p.add_argument("--bars-limit", type=int, default=None)
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument("--max-conditions", type=int, default=3)
    p.add_argument("--minimal-edge-threshold", type=float, default=0.0002)
    # Candidate-specific knobs
    p.add_argument(
        "--top-candidates",
        type=int,
        default=50,
        help="How many cross-asset patterns to convert.",
    )
    p.add_argument(
        "--single-asset-top",
        type=int,
        default=20,
        help="How many per-asset top patterns to also emit (informational).",
    )
    p.add_argument(
        "--stop-mult",
        type=float,
        default=1.5,
        help="ATR stop multiplier.",
    )
    p.add_argument(
        "--tp-mult",
        type=float,
        default=2.5,
        help="ATR take-profit multiplier.",
    )
    p.add_argument(
        "--max-hold",
        type=int,
        default=20,
        help="Maximum bars to hold a position.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    assets = [a.strip().lower() for a in args.assets.split(",") if a.strip()]
    if not assets:
        print("ERROR: no assets provided.")
        return

    print()
    print("AION Strategy-Candidate Generator")
    print("=" * 72)
    print(f"  assets={','.join(assets)}   top_candidates={args.top_candidates}")
    print()

    t0 = time.perf_counter()
    execution_model = ExecutionModel.from_config(EXEC_CONFIG)

    asset_results: dict[str, list] = {}
    canonical_name: dict[str, str] = {}

    for asset in assets:
        print(f"[{asset}] running discovery...")
        try:
            results, _ = _discover_asset(asset, args, execution_model)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"  [{asset}] skipped: {exc}")
            continue
        _, spec_factory = ASSET_REGISTRY[asset]
        canonical = spec_factory().symbol
        canonical_name[asset] = canonical
        asset_results[canonical] = results

    if not asset_results:
        print("No asset produced results; aborting.")
        return

    cross = validate_across_assets(
        asset_results,
        min_assets=args.min_assets,
        require_sign_agreement=False,
    )
    print()
    print(f"cross-asset survivors: {len(cross)}")

    # ── Convert cross-asset ─────────────────────────────────────────────────
    cross_candidates = []
    for r in cross[: args.top_candidates]:
        cand = convert_multi_asset(
            r,
            stop_mult=args.stop_mult,
            tp_mult=args.tp_mult,
            max_hold=args.max_hold,
        )
        if cand is not None:
            cross_candidates.append(cand)

    # ── Also convert top-N per asset (informational) ────────────────────────
    single_asset_candidates: dict[str, list] = {}
    for asset_canonical, results in asset_results.items():
        converted = []
        for r in results[: args.single_asset_top]:
            cand = convert_compact_result(
                r,
                stop_mult=args.stop_mult,
                tp_mult=args.tp_mult,
                max_hold=args.max_hold,
            )
            if cand is not None:
                converted.append(cand)
        single_asset_candidates[asset_canonical] = converted

    # ── Serialise ───────────────────────────────────────────────────────────
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "discovery_params": {
            "assets": assets,
            "min_samples": args.min_samples,
            "min_assets": args.min_assets,
            "max_conditions": args.max_conditions,
            "minimal_edge_threshold": args.minimal_edge_threshold,
            "snapshots": args.snapshots,
            "bars_limit": args.bars_limit,
            "start_date": args.start_date,
            "end_date": args.end_date,
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
        "cross_asset": [candidate_to_dict(c) for c in cross_candidates],
        "single_asset": {
            asset: [candidate_to_dict(c) for c in cands]
            for asset, cands in single_asset_candidates.items()
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print()
    print(f"wrote {len(cross_candidates)} cross-asset candidate(s)")
    for asset, cands in single_asset_candidates.items():
        print(f"       {len(cands):>3} single-asset candidates for {asset}")
    print(f"  → {out_path}")
    print(f"total runtime: {time.perf_counter() - t0:.1f}s")
    print()


if __name__ == "__main__":
    main()
