"""
research.edge_decay.run_edge_decay
────────────────────────────────────
Edge-decay analysis runner.

Loads pattern + sequential strategy candidates, wraps them through the
meta-strategy UnifiedCandidate adapter, backtests each one on a
compact feature matrix, and produces a DecayReport per candidate.

Output
──────
Printed table + JSON dump at:
    research/output/edge_decay_<asset>.json

CLI example
───────────
    python -m research.edge_decay.run_edge_decay \\
        --asset xauusd \\
        --pattern-candidates    research/output/strategy_candidates.json \\
        --sequential-candidates research/output/sequential_strategy_candidates_xauusd.json \\
        --bars-limit 120000 --snapshots 5000 \\
        --window-sizes 50,100,200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aion.execution.execution_model import ExecutionModel

from research.edge_decay.decay_report import (
    BROKEN_DD_THRESHOLD,
    DECAYING_REL_CHANGE,
    IMPROVING_REL_CHANGE,
    STABLE_PF_MIN,
    build_report,
)
from research.meta_strategy import (
    UnifiedCandidate,
    backtest_for,
    wrap_pattern,
    wrap_sequential,
)
from research.pattern_discovery import FeatureBuilder
from research.pattern_strategies import candidate_from_dict as pattern_from_dict
from research.sequential_strategies import (
    candidate_from_dict as sequential_from_dict,
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


DEFAULT_PATTERN_CANDIDATES = (
    _ROOT / "research" / "output" / "strategy_candidates.json"
)
DEFAULT_SEQUENTIAL_BY_ASSET = {
    "us100":  _ROOT / "research" / "output" / "sequential_strategy_candidates.json",
    "xauusd": _ROOT / "research" / "output" / "sequential_strategy_candidates_xauusd.json",
    "btcusd": _ROOT / "research" / "output" / "sequential_strategy_candidates_btcusd.json",
}
OUTPUT_DIR = _ROOT / "research" / "output"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AION edge-decay analysis")
    p.add_argument("--asset", type=str, default="us100",
                   choices=sorted(ASSET_REGISTRY.keys()))
    p.add_argument(
        "--pattern-candidates",
        type=str,
        default=str(DEFAULT_PATTERN_CANDIDATES),
    )
    p.add_argument("--sequential-candidates", type=str, default=None)
    p.add_argument(
        "--source",
        choices=("cross_asset", "single_asset", "auto"),
        default="auto",
    )
    p.add_argument("--no-patterns", action="store_true")
    p.add_argument("--no-sequences", action="store_true")
    p.add_argument("--snapshots", type=int, default=3000)
    p.add_argument("--bars-limit", type=int, default=None)
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument(
        "--window-sizes",
        type=str,
        default="50,100,200",
        help="Comma-separated trade-window sizes for rolling metrics.",
    )
    p.add_argument("--broken-dd-threshold", type=float,
                   default=BROKEN_DD_THRESHOLD)
    p.add_argument("--improving-threshold", type=float,
                   default=IMPROVING_REL_CHANGE)
    p.add_argument("--decaying-threshold", type=float,
                   default=DECAYING_REL_CHANGE)
    p.add_argument("--stable-pf-min", type=float, default=STABLE_PF_MIN)
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path.  Defaults to "
             "research/output/edge_decay_<asset>.json",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Cap printed + serialised rows to the top-N by |decay_score|.",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────


def _load_pattern_candidates(
    path: Path, asset_symbol: str, source: str
) -> list:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    cross = data.get("cross_asset", [])
    single = data.get("single_asset", {}).get(asset_symbol, [])
    if source == "cross_asset":
        selected = cross
    elif source == "single_asset":
        selected = single
    else:
        selected = cross if cross else single
    return [pattern_from_dict(d) for d in selected]


def _load_sequential_candidates(path: Path) -> list:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [sequential_from_dict(d) for d in data.get("candidates", [])]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()
    asset = args.asset.lower()
    window_sizes = tuple(
        int(x) for x in args.window_sizes.split(",") if x.strip()
    )
    if not window_sizes:
        print("ERROR: --window-sizes must contain at least one positive int.")
        return

    print()
    print("AION Edge-Decay Analysis")
    print("=" * 72)
    print(
        f"  asset={asset}   windows={window_sizes}   "
        f"snapshots<={args.snapshots}"
    )
    print()

    csv_name, spec_factory = ASSET_REGISTRY[asset]
    instrument = spec_factory()
    csv_path = DATA_DIR / csv_name
    if not csv_path.exists():
        print(f"ERROR: data file not found: {csv_path}")
        return

    # ── Load candidates ────────────────────────────────────────────────────
    unified: list[UnifiedCandidate] = []
    n_pat = n_seq = 0
    if not args.no_patterns:
        pats = _load_pattern_candidates(
            Path(args.pattern_candidates), instrument.symbol, args.source
        )
        unified.extend(wrap_pattern(c) for c in pats)
        n_pat = len(pats)
        print(f"  pattern candidates    : {n_pat}")

    if not args.no_sequences:
        seq_arg = args.sequential_candidates
        seq_path = (
            Path(seq_arg) if seq_arg else DEFAULT_SEQUENTIAL_BY_ASSET.get(asset)
        )
        if seq_path is not None:
            seqs = _load_sequential_candidates(seq_path)
            unified.extend(wrap_sequential(c) for c in seqs)
            n_seq = len(seqs)
            print(
                f"  sequential candidates : {n_seq}"
                + (f"  ({seq_path.name})" if seq_path.exists() else "  (missing)")
            )
        else:
            print("  sequential candidates : (no default for asset)")

    if not unified:
        print("No candidates to analyse.")
        return

    # ── Build compact matrix once ──────────────────────────────────────────
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
        f"snapshots={len(snapshots):,}  ({time.perf_counter() - t0:.1f}s)"
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

    # ── Per-candidate analysis ─────────────────────────────────────────────
    t = time.perf_counter()
    reports = []
    for u in unified:
        try:
            rep = backtest_for(df, u)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {u.name}: {exc}")
            continue
        returns = [t.return_pct for t in rep.trades]
        decay = build_report(
            candidate_name=u.name,
            candidate_type=u.candidate_type,
            asset=instrument.symbol,
            trade_returns=returns,
            window_sizes=window_sizes,
            broken_dd_threshold=args.broken_dd_threshold,
            improving_threshold=args.improving_threshold,
            decaying_threshold=args.decaying_threshold,
            stable_pf_min=args.stable_pf_min,
        )
        reports.append(decay)
    print(
        f"  analysed {len(reports)} candidates  "
        f"({time.perf_counter() - t:.1f}s)"
    )

    # ── Rank + print ───────────────────────────────────────────────────────
    reports.sort(key=lambda r: r.decay_score, reverse=True)
    if args.top_n:
        reports = reports[: args.top_n]

    print()
    print("====================================")
    print("  EDGE DECAY SUMMARY")
    print("====================================")
    print()
    print(f"asset: {instrument.symbol}")
    print(
        f"candidates: pattern={n_pat}  sequence={n_seq}  "
        f"analysed={len(reports)}"
    )
    print()
    counts: dict[str, int] = {}
    for r in reports:
        counts[r.status] = counts.get(r.status, 0) + 1
    print("  " + "  ".join(
        f"{k}={v}" for k, v in sorted(counts.items())
    ))
    print()
    print(
        f"{'type':<8}  {'name':<48}  trades  pf     exp        "
        f"score  status"
    )
    print("-" * 104)
    for r in reports:
        print(f"  {r.summary_line()}")
    print()

    # ── JSON output ────────────────────────────────────────────────────────
    out_path = (
        Path(args.output) if args.output
        else OUTPUT_DIR / f"edge_decay_{asset}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asset": instrument.symbol,
        "window_sizes": list(window_sizes),
        "thresholds": {
            "broken_dd_threshold": args.broken_dd_threshold,
            "improving_threshold": args.improving_threshold,
            "decaying_threshold": args.decaying_threshold,
            "stable_pf_min": args.stable_pf_min,
        },
        "counts": {
            "pattern_candidates": n_pat,
            "sequence_candidates": n_seq,
            "analysed": len(reports),
            "by_status": counts,
        },
        "reports": [r.to_dict() for r in reports],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote → {out_path}")
    print()


if __name__ == "__main__":
    main()
