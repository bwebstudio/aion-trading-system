"""
research/run_pattern_discovery.py
──────────────────────────────────
Pattern Discovery v4 — multi-asset discovery and cross-asset validation.

Pipeline (per asset, sequentially):
    1. Load CSV + build MarketSnapshots
    2. Build compact feature DataFrame (pre-binned, precomputed fwd return)
    3. Stream candidate PatternKeys
    4. Vectorised batched evaluation with top-k heap
    5. Retain per-asset top-k as the asset's discovered set

After all assets have been processed:
    6. Cross-asset validation via multi_asset_validator.validate_across_assets
       Keeps only PatternKeys observed in >= min_assets (default 2).
    7. Rank by score = mean(test_mean_return across assets) * sqrt(total_n).
    8. Print per-asset summary and top cross-asset edges.

Assets are processed one at a time; per-asset snapshot / matrix objects
are dropped before the next asset loads, so peak memory stays bounded.

CLI
───
    python research/run_pattern_discovery.py
        --assets us100,xauusd,btcusd
        [--max-patterns N]  [--batch-size N]
        [--top-k N]         [--min-samples N]
        [--snapshots N]     [--min-assets N]
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aion.core.enums import AssetClass, Timeframe
from aion.core.models import InstrumentSpec, MarketBar
from aion.data.csv_adapter import load_csv_bars
from aion.data.csv_loader import load_bars as _load_bars_default
from aion.data.normalizer import normalize_bars
from aion.execution.execution_model import ExecutionModel


def _load_bars(path: Path, instrument: InstrumentSpec) -> list[MarketBar]:
    """
    CSV loader tolerant to the MT5 `time`-column export format used by the
    multi-asset files (single column named 'time' that already holds a full
    ISO timestamp with tz offset).  Falls back to the default loader for
    files with a standard `timestamp` / `date+time` layout.
    """
    try:
        return _load_bars_default(path, instrument, drop_last=True)
    except Exception:
        raw_bars = load_csv_bars(
            path,
            symbol=instrument.symbol,
            broker_timezone=instrument.broker_timezone,
            timestamp_col="time",
        )
        bars = normalize_bars(raw_bars, instrument, Timeframe.M1)
        bars.sort(key=lambda b: b.timestamp_utc)
        if len(bars) > 1:
            bars = bars[:-1]
        return bars

from research.pattern_discovery import (
    CompactPatternResult,
    FeatureBuilder,
    ForwardTester,
    PatternGenerator,
    validate_across_assets,
)
from research.pattern_discovery.feature_builder import BIN_COLUMNS

from scripts.run_us100_replay import _build_snapshots


DATA_DIR = _ROOT / "data" / "raw"
EXEC_CONFIG = _ROOT / "aion" / "config" / "execution_config.yaml"

FORWARD_BARS = 10
TRAIN_FRACTION = 0.70
MIN_SAMPLES_TRAIN = 100
MIN_SAMPLES_TEST = 50
STABILITY_TOLERANCE = 0.50
SIGMA_WINDOW = 500


# ─────────────────────────────────────────────────────────────────────────────
# Instrument specs
# ─────────────────────────────────────────────────────────────────────────────

def _us100_spec() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="US100.cash",
        broker_symbol="US100.cash",
        asset_class=AssetClass.INDICES,
        price_timezone="America/New_York",
        market_timezone="America/New_York",
        broker_timezone="Etc/UTC",
        tick_size=0.01,
        point_value=1.0,
        contract_size=1.0,
        min_lot=0.01,
        lot_step=0.01,
        currency_profit="USD",
        currency_margin="USD",
        session_calendar="us_equity",
        trading_hours_label="Mon-Fri, nearly 24h (broker dependent)",
    )


def _xauusd_spec() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="XAUUSD",
        broker_symbol="XAUUSD",
        asset_class=AssetClass.COMMODITIES,
        price_timezone="Etc/UTC",
        market_timezone="America/New_York",
        broker_timezone="Etc/UTC",
        tick_size=0.01,
        point_value=1.0,
        contract_size=100.0,
        min_lot=0.01,
        lot_step=0.01,
        currency_profit="USD",
        currency_margin="USD",
        session_calendar="forex_standard",
        trading_hours_label="Sun 22:00 - Fri 22:00 UTC",
    )


def _btcusd_spec() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="BTCUSD",
        broker_symbol="BTCUSD",
        asset_class=AssetClass.CRYPTO,
        price_timezone="Etc/UTC",
        market_timezone="Etc/UTC",
        broker_timezone="Etc/UTC",
        tick_size=0.01,
        point_value=1.0,
        contract_size=1.0,
        min_lot=0.01,
        lot_step=0.01,
        currency_profit="USD",
        currency_margin="USD",
        session_calendar="crypto_24_7",
        trading_hours_label="24/7 (crypto)",
    )


# Asset registry: key (CLI name) -> (csv filename, instrument factory)
ASSET_REGISTRY: dict[str, tuple[str, callable]] = {
    "us100":  ("us100_mt5_all_available_m1.csv",  _us100_spec),
    "xauusd": ("xauusd_mt5_all_available_m1.csv", _xauusd_spec),
    "btcusd": ("btcusd_mt5_all_available_m1.csv", _btcusd_spec),
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AION Pattern Discovery v4 (multi-asset)")
    p.add_argument(
        "--assets",
        type=str,
        default="us100",
        help="Comma-separated asset keys (us100, xauusd, btcusd).",
    )
    p.add_argument("--max-patterns", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--top-k", type=int, default=500)
    p.add_argument("--min-samples", type=int, default=100)
    p.add_argument(
        "--snapshots",
        type=int,
        default=3000,
        help="Cap number of active-session snapshots per asset.",
    )
    p.add_argument(
        "--min-assets",
        type=int,
        default=2,
        help="Minimum assets a pattern must appear in to survive cross-asset validation.",
    )
    p.add_argument(
        "--bars-limit",
        type=int,
        default=None,
        help=(
            "If set, keep only the most recent N raw M1 bars per asset "
            "BEFORE feature computation.  Strong speed win on large CSVs."
        ),
    )
    p.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Inclusive start date (UTC, YYYY-MM-DD). Filters raw bars early.",
    )
    p.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Exclusive end date (UTC, YYYY-MM-DD). Filters raw bars early.",
    )
    p.add_argument(
        "--max-conditions",
        type=int,
        default=3,
        help="Maximum conditions per pattern (Apriori expansion depth).",
    )
    p.add_argument(
        "--minimal-edge-threshold",
        type=float,
        default=0.0002,
        help="Prune patterns whose |mean_return| is below this threshold.",
    )
    return p.parse_args()


def _parse_date(s: str | None) -> "datetime | None":
    if s is None:
        return None
    from datetime import datetime, timezone
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _filter_bars_early(
    bars: list[MarketBar],
    *,
    start: "datetime | None",
    end: "datetime | None",
    bars_limit: int | None,
) -> list[MarketBar]:
    """
    Apply date range then row-count limit on RAW bars, before any feature
    computation.  Bars are already sorted ascending by timestamp_utc.
    """
    out = bars
    if start is not None or end is not None:
        out = [
            b for b in out
            if (start is None or b.timestamp_utc >= start)
            and (end is None or b.timestamp_utc < end)
        ]
    if bars_limit is not None and len(out) > bars_limit:
        out = out[-bars_limit:]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-asset discovery
# ─────────────────────────────────────────────────────────────────────────────

def _discover_asset(
    asset_key: str,
    args: argparse.Namespace,
    execution_model: ExecutionModel,
) -> tuple[list[CompactPatternResult], dict[str, float]]:
    """
    Run the full single-asset discovery pipeline.  Returns the top-k
    CompactPatternResult list plus a timings dict for profiling.
    """
    if asset_key not in ASSET_REGISTRY:
        raise ValueError(
            f"Unknown asset '{asset_key}'. "
            f"Known: {sorted(ASSET_REGISTRY.keys())}"
        )

    csv_name, spec_factory = ASSET_REGISTRY[asset_key]
    csv_path = DATA_DIR / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    instrument = spec_factory()

    timings: dict[str, float] = {}

    # ── Load raw bars ────────────────────────────────────────────────────────
    t = time.perf_counter()
    bars = _load_bars(csv_path, instrument)
    if not bars:
        raise RuntimeError(f"No bars loaded from {csv_path}")
    raw_total = len(bars)

    # ── Early filter BEFORE feature computation ─────────────────────────────
    start_dt = _parse_date(getattr(args, "start_date", None))
    end_dt = _parse_date(getattr(args, "end_date", None))
    bars = _filter_bars_early(
        bars,
        start=start_dt,
        end=end_dt,
        bars_limit=args.bars_limit,
    )
    raw_after = len(bars)
    timings["load_filter"] = time.perf_counter() - t

    if not bars:
        print(f"  [{asset_key}] raw={raw_total:,}  after-filter=0  (skipped)")
        return [], timings

    # ── Snapshot build on filtered bars only ────────────────────────────────
    t = time.perf_counter()
    snapshots = _build_snapshots(bars, instrument)
    del bars
    if len(snapshots) > args.snapshots:
        snapshots = snapshots[-args.snapshots:]
    timings["snapshots"] = time.perf_counter() - t
    snapshots_built = len(snapshots)

    print(
        f"  [{asset_key}] raw_total={raw_total:>8,}  "
        f"after_filter={raw_after:>8,}  "
        f"snapshots_built={snapshots_built:>6,}  "
        f"(load+filter {timings['load_filter']:.1f}s, "
        f"snap {timings['snapshots']:.1f}s)"
    )

    if snapshots_built < args.min_samples + FORWARD_BARS + 10:
        print(f"  [{asset_key}] insufficient snapshots; skipping.")
        return [], timings

    # ── Compact matrix ───────────────────────────────────────────────────────
    t = time.perf_counter()
    fb = FeatureBuilder(compression_lookback=10, sigma_window=SIGMA_WINDOW)
    df = fb.build_compact_matrix(
        snapshots,
        forward_bars=FORWARD_BARS,
        execution_model=execution_model,
    )
    del snapshots  # matrix is self-contained from here
    timings["features"] = time.perf_counter() - t
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(
        f"  [{asset_key}] matrix: {len(df):,} rows  {mem_mb:.1f} MB  "
        f"(snap {timings['snapshots']:.1f}s, feat {timings['features']:.1f}s)"
    )

    # ── Pattern stream (Apriori level-wise) ─────────────────────────────────
    t = time.perf_counter()
    generator = PatternGenerator(
        min_bucket_occurrences=args.min_samples,
        max_conditions=args.max_conditions,
        minimal_edge_threshold=args.minimal_edge_threshold,
    )
    keys: list = []
    for batch in generator.generate_patterns_levelwise(
        df,
        bin_columns=BIN_COLUMNS,
        batch_size=args.batch_size,
        min_samples=args.min_samples,
        minimal_edge_threshold=args.minimal_edge_threshold,
        max_conditions=args.max_conditions,
    ):
        keys.extend(batch)
    timings["generation"] = time.perf_counter() - t

    # ── Vectorised evaluation with top-k ─────────────────────────────────────
    t = time.perf_counter()
    tester = ForwardTester(
        execution_model=execution_model,
        forward_bars=FORWARD_BARS,
        train_fraction=TRAIN_FRACTION,
        min_samples_train=MIN_SAMPLES_TRAIN,
        min_samples_test=MIN_SAMPLES_TEST,
        stability_tolerance=STABILITY_TOLERANCE,
    )
    results = tester.evaluate_patterns(
        df,
        iter(keys),
        batch_size=args.batch_size,
        top_k=args.top_k,
        min_samples=args.min_samples,
        train_fraction=TRAIN_FRACTION,
        min_samples_train=MIN_SAMPLES_TRAIN,
        min_samples_test=MIN_SAMPLES_TEST,
        stability_tolerance=STABILITY_TOLERANCE,
        progress_every=0,
    )
    timings["evaluation"] = time.perf_counter() - t
    print(
        f"  [{asset_key}] generated={len(keys):,}  "
        f"survived={len(results):,}  "
        f"(gen {timings['generation']:.1f}s, eval {timings['evaluation']:.1f}s)"
    )

    # Free the DataFrame before returning — results hold only PatternKeys + floats.
    del df, keys
    gc.collect()
    return results, timings


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    assets = [a.strip().lower() for a in args.assets.split(",") if a.strip()]
    if not assets:
        print("ERROR: no assets provided (use --assets us100,xauusd,btcusd).")
        return

    print()
    print("AION Pattern Discovery v4 — multi-asset")
    print("=" * 72)
    print(
        f"  assets={','.join(assets)}   "
        f"max-patterns={args.max_patterns}   "
        f"batch-size={args.batch_size}   "
        f"top-k={args.top_k}   "
        f"min-samples={args.min_samples}   "
        f"snapshots<={args.snapshots}   "
        f"min-assets={args.min_assets}"
    )
    print()

    t_total = time.perf_counter()
    execution_model = ExecutionModel.from_config(EXEC_CONFIG)

    asset_results: dict[str, list[CompactPatternResult]] = {}
    asset_timings: dict[str, dict[str, float]] = {}
    canonical_name: dict[str, str] = {}

    for asset in assets:
        print(f"[{asset}] starting discovery...")
        try:
            results, timings = _discover_asset(asset, args, execution_model)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"  [{asset}] skipped: {exc}")
            continue

        csv_name, spec_factory = ASSET_REGISTRY[asset]
        canonical_name[asset] = spec_factory().symbol
        asset_results[canonical_name[asset]] = results
        asset_timings[asset] = timings
        gc.collect()

    if not asset_results:
        print()
        print("No assets produced results.")
        return

    # ── Cross-asset validation ───────────────────────────────────────────────
    cross = validate_across_assets(
        asset_results,
        min_assets=args.min_assets,
        require_sign_agreement=False,
    )

    # ── Report ───────────────────────────────────────────────────────────────
    print()
    print("====================================")
    print("  MULTI-ASSET DISCOVERY SUMMARY")
    print("====================================")
    print()
    print("assets analysed:")
    for asset in assets:
        if asset in canonical_name:
            print(f"  {canonical_name[asset]}")
    print()
    print("patterns per asset:")
    for asset_canonical, results in asset_results.items():
        print(f"  {asset_canonical:<12}: {len(results)}")
    print()
    print(
        f"cross-asset patterns (>= {args.min_assets} assets): "
        f"{len(cross)}"
    )
    sign_agreed = sum(1 for r in cross if r.sign_agreement)
    print(f"  with sign agreement across assets            : {sign_agreed}")

    print()
    print(f"Top 20 cross-asset edges (sorted by |mean_score|)")
    print("-" * 72)
    if not cross:
        print("  (none)")
    else:
        for r in cross[:20]:
            print(f"  {r.describe()}")

    # Per-asset timings
    print()
    print("Runtime per asset")
    print("-" * 72)
    for asset, timings in asset_timings.items():
        parts = ", ".join(f"{k}={v:.1f}s" for k, v in timings.items())
        total_a = sum(timings.values())
        print(f"  {asset:<8}: total={total_a:5.1f}s   ({parts})")

    total = time.perf_counter() - t_total
    print()
    print(f"Total runtime: {total:.1f}s")
    print()


if __name__ == "__main__":
    main()
