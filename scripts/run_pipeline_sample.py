"""
scripts/run_pipeline_sample.py
-------------------------------
Run the full historical pipeline over a sample EURUSD M1 CSV file and
print a human-readable summary of the results.

If no CSV file is provided (or the default sample file does not exist),
a synthetic 200-bar dataset is generated and saved to
  data/sample/EURUSD_M1_sample.csv

Usage:
    python scripts/run_pipeline_sample.py
    python scripts/run_pipeline_sample.py path/to/EURUSD_M1.csv
    python scripts/run_pipeline_sample.py --persist
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Ensure the project root is on sys.path when run directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# -----------------------------------------------------------------------------
# Synthetic sample CSV generator
# -----------------------------------------------------------------------------


def generate_sample_csv(path: Path, n_bars: int = 200) -> None:
    """Write a synthetic MT5-format CSV with `n_bars` M1 bars."""
    path.parent.mkdir(parents=True, exist_ok=True)
    base_ts = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
    base_price = 1.1000
    price_step = 0.0001

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>",
             "<TICKVOL>", "<VOL>", "<SPREAD>"]
        )
        for i in range(n_bars):
            ts = base_ts + timedelta(minutes=i)
            close = base_price + i * price_step
            open_ = close - price_step * 0.4
            high = close + price_step * 0.6
            low = open_ - price_step * 0.6
            writer.writerow([
                ts.strftime("%Y.%m.%d"),
                ts.strftime("%H:%M:%S"),
                f"{open_:.5f}",
                f"{high:.5f}",
                f"{low:.5f}",
                f"{close:.5f}",
                "100",
                "0",
                "2",
            ])
    print(f"  Generated {n_bars}-bar sample CSV -> {path}")


# -----------------------------------------------------------------------------
# Instrument definition
# -----------------------------------------------------------------------------


def make_eurusd():
    from aion.core.enums import AssetClass
    from aion.core.models import InstrumentSpec
    return InstrumentSpec(
        symbol="EURUSD",
        broker_symbol="EURUSD",
        asset_class=AssetClass.FOREX,
        price_timezone="Etc/UTC",
        market_timezone="Etc/UTC",
        broker_timezone="Etc/UTC",
        tick_size=0.00001,
        point_value=10.0,
        contract_size=100_000.0,
        min_lot=0.01,
        lot_step=0.01,
        currency_profit="USD",
        currency_margin="USD",
        session_calendar="forex_standard",
        trading_hours_label="Sun 22:00 - Fri 22:00 UTC",
    )


# -----------------------------------------------------------------------------
# Report printer
# -----------------------------------------------------------------------------


def print_result(result) -> None:
    snap = result.snapshot
    qr = result.quality_report
    fv = snap.feature_vector

    print("\n" + "-" * 60)
    print("PIPELINE RESULT")
    print("-" * 60)
    print(f"  Run ID        : {result.pipeline_run_id}")
    print(f"  Elapsed       : {result.elapsed_seconds:.3f}s")
    print()
    print("BARS")
    print(f"  Loaded        : {result.bars_loaded}")
    print(f"  After normalise: {result.bars_after_normalise}")
    print(f"  Dropped (incomplete last bar): {result.bars_dropped_incomplete}")
    print(f"  M1 bars       : {len(result.bars_m1)}")
    print(f"  M5 bars       : {len(result.bars_m5)}")
    print(f"  M15 bars      : {len(result.bars_m15)}")
    print(f"  Features (M1) : {len(result.features_m1)}")
    print()
    print("QUALITY REPORT")
    print(f"  Score         : {qr.quality_score:.4f}  ({'USABLE' if snap.is_usable else 'NOT USABLE'})")
    print(f"  Rows checked  : {qr.rows_checked}")
    print(f"  Missing bars  : {qr.missing_bars}")
    print(f"  Duplicates    : {qr.duplicate_timestamps}")
    print(f"  Out of order  : {qr.out_of_order_rows}")
    print(f"  Stale bars    : {qr.stale_bars}")
    print(f"  Spike bars    : {qr.spike_bars}")
    if qr.warnings:
        for w in qr.warnings:
            print(f"  ! {w}")
    print()
    print("SNAPSHOT")
    print(f"  ID            : {snap.snapshot_id}")
    print(f"  Symbol        : {snap.symbol}")
    print(f"  Timestamp UTC : {snap.timestamp_utc}")
    print(f"  Session       : {snap.session_context.session_name}")
    print(f"  Snapshot ver  : {snap.snapshot_version}")
    print(f"  Latest close  : {snap.latest_bar.close:.5f}")
    print()
    print("FEATURE VECTOR (latest bar)")
    print(f"  ATR-14        : {fv.atr_14}")
    print(f"  Rolling rng 10: {fv.rolling_range_10}")
    print(f"  Rolling rng 20: {fv.rolling_range_20}")
    print(f"  Return 1      : {fv.return_1}")
    print(f"  Return 5      : {fv.return_5}")
    print(f"  Session high  : {fv.session_high}")
    print(f"  Session low   : {fv.session_low}")
    print(f"  VWAP          : {fv.vwap_session}")
    print(f"  Opening R high: {fv.opening_range_high}")
    print(f"  Opening R low : {fv.opening_range_low}")
    print(f"  Spread mean 20: {fv.spread_mean_20}")
    print(f"  Feature ver   : {fv.feature_set_version}")

    if result.persist_paths:
        print()
        print("PERSISTED FILES")
        for p in result.persist_paths:
            print(f"  {p}")

    print("-" * 60 + "\n")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> None:
    from aion.data.pipeline import run_historical_pipeline

    # Parse arguments
    args = sys.argv[1:]
    do_persist = "--persist" in args
    csv_args = [a for a in args if not a.startswith("--")]

    default_csv = _ROOT / "data" / "sample" / "EURUSD_M1_sample.csv"

    if csv_args:
        csv_path = Path(csv_args[0])
        if not csv_path.exists():
            print(f"Error: file not found: {csv_path}", file=sys.stderr)
            sys.exit(1)
    else:
        csv_path = default_csv
        if not csv_path.exists():
            print("No CSV path provided — generating synthetic sample data…")
            generate_sample_csv(csv_path, n_bars=200)

    print(f"\nRunning pipeline on: {csv_path}")

    persist_root = _ROOT / "data" / "pipeline_output" if do_persist else None

    result = run_historical_pipeline(
        csv_path,
        instrument=make_eurusd(),
        local_timezone="Etc/UTC",
        drop_incomplete_last_bar=True,
        persist=do_persist,
        persist_bar_root=persist_root / "bars" if persist_root else None,
        persist_feature_root=persist_root / "features" if persist_root else None,
        persist_snapshot_root=persist_root / "snapshots" if persist_root else None,
    )

    print_result(result)


if __name__ == "__main__":
    main()
