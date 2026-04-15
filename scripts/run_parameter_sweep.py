"""
scripts/run_parameter_sweep.py
────────────────────────────────
Run a grid search over OpeningRangeEngine parameters and print a comparison.

Usage:
  python scripts/run_parameter_sweep.py
  python scripts/run_parameter_sweep.py --snapshots 150

What this script does:
  1. Builds synthetic MarketSnapshots.
  2. Defines several SweepPoints covering different parameter combinations.
  3. Runs run_parameter_sweep() for each configuration.
  4. Prints a comparison table sorted by win_rate.

Sweep configurations:
  baseline        -- default parameters (stop=10, target=20, range 5-40 pips)
  target_25       -- target = 25 pips (exactly at bar.high ceiling)
  target_30       -- target = 30 pips (above bar.high; produces timeouts)
  tight_range     -- max_range = 15 pips (blocks 20-pip OR range)
  spread_strict   -- max_spread_pips = 0.1 (blocks synthetic spread of 0.2p)
  short_only      -- direction_bias = SHORT (blocks LONG-only OR signal)
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aion.analytics.parameter_sweeps import run_parameter_sweep
from aion.analytics.replay_models import SweepPoint
from aion.core.constants import FEATURE_SET_VERSION, SNAPSHOT_VERSION
from aion.core.enums import AssetClass, DataSource, SessionName, Timeframe, TradeDirection
from aion.core.ids import new_snapshot_id
from aion.core.models import (
    DataQualityReport,
    FeatureVector,
    InstrumentSpec,
    MarketBar,
    MarketSnapshot,
    SessionContext,
)

_UTC = timezone.utc
_BASE_TS = datetime(2024, 1, 15, 8, 0, 0, tzinfo=_UTC)


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot builder
# ─────────────────────────────────────────────────────────────────────────────


def _build_snapshots(n: int = 100) -> list[MarketSnapshot]:
    instrument = InstrumentSpec(
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
    or_cutoff = 30
    snaps = []
    for i in range(n):
        ts = _BASE_TS + timedelta(minutes=i)
        or_active = i < or_cutoff
        or_completed = not or_active
        mid = round((1.1045 + 1.1035) / 2, 5)
        bar = MarketBar(
            symbol="EURUSD",
            timestamp_utc=ts,
            timestamp_market=ts,
            timeframe=Timeframe.M1,
            open=mid,
            high=1.1045,
            low=1.1035,
            close=mid,
            tick_volume=100.0,
            real_volume=0.0,
            spread=2.0,
            source=DataSource.SYNTHETIC,
        )
        session_open = ts.replace(hour=8, minute=0, second=0)
        session_close = ts.replace(hour=16, minute=30, second=0)
        session = SessionContext(
            trading_day=ts.date(),
            broker_time=ts,
            market_time=ts,
            local_time=ts,
            is_asia=False,
            is_london=True,
            is_new_york=False,
            is_session_open_window=True,
            opening_range_active=or_active,
            opening_range_completed=or_completed,
            session_name=SessionName.LONDON,
            session_open_utc=session_open,
            session_close_utc=session_close,
        )
        fv = FeatureVector(
            symbol="EURUSD",
            timestamp_utc=ts,
            timeframe=Timeframe.M1,
            atr_14=0.00015,
            rolling_range_10=0.0010,
            rolling_range_20=0.0012,
            volatility_percentile_20=0.50,
            session_high=1.1060,
            session_low=1.0990,
            opening_range_high=1.1020,
            opening_range_low=1.1000,
            vwap_session=1.1010,
            spread_mean_20=2.0,
            spread_zscore_20=0.0,
            return_1=0.0001,
            return_5=0.0003,
            candle_body=0.00005,
            upper_wick=0.00005,
            lower_wick=0.00005,
            distance_to_session_high=-0.0040,
            distance_to_session_low=0.0010,
            feature_set_version=FEATURE_SET_VERSION,
        )
        qr = DataQualityReport(
            symbol="EURUSD",
            timeframe=Timeframe.M1,
            rows_checked=100,
            missing_bars=0,
            duplicate_timestamps=0,
            out_of_order_rows=0,
            stale_bars=0,
            spike_bars=0,
            null_rows=0,
            quality_score=1.0,
            warnings=[],
        )
        snaps.append(
            MarketSnapshot(
                snapshot_id=new_snapshot_id(),
                symbol="EURUSD",
                timestamp_utc=ts,
                base_timeframe=Timeframe.M1,
                instrument=instrument,
                session_context=session,
                latest_bar=bar,
                bars_m1=[bar],
                bars_m5=[],
                bars_m15=[],
                feature_vector=fv,
                quality_report=qr,
                snapshot_version=SNAPSHOT_VERSION,
            )
        )
    return snaps


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────


def _section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _pct(v: float | None) -> str:
    return f"{v * 100:.1f}%" if v is not None else "N/A"


def _fp(v: float | None, decimals: int = 1) -> str:
    return f"{v:.{decimals}f}" if v is not None else "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OR parameter sweep.")
    parser.add_argument("--snapshots", type=int, default=100)
    args = parser.parse_args()

    _section("Building synthetic snapshots")
    snaps = _build_snapshots(args.snapshots)
    print(f"  Generated : {len(snaps)} snapshots")
    print(f"  Symbol    : EURUSD  |  OR range: 20 pips (1.1000 - 1.1020)")
    print(f"  Bar OHLC  : high=1.1045 low=1.1035  (entry activates on same bar)")

    sweep_points = [
        SweepPoint(
            label="baseline",
            min_range_pips=5.0,
            max_range_pips=40.0,
            stop_pips=10.0,
            target_pips=20.0,
            max_label_bars=30,
        ),
        SweepPoint(
            label="target_25",
            min_range_pips=5.0,
            max_range_pips=40.0,
            stop_pips=10.0,
            target_pips=25.0,  # exact ceiling; 1.1045 >= 1.1045
            max_label_bars=30,
        ),
        SweepPoint(
            label="target_30_timeout",
            min_range_pips=5.0,
            max_range_pips=40.0,
            stop_pips=10.0,
            target_pips=30.0,  # above bar.high=1.1045; all TIMEOUT
            max_label_bars=30,
        ),
        SweepPoint(
            label="tight_range",
            min_range_pips=5.0,
            max_range_pips=15.0,  # 20-pip OR range > 15 max; blocked
            stop_pips=10.0,
            target_pips=20.0,
            max_label_bars=30,
        ),
        SweepPoint(
            label="spread_strict",
            min_range_pips=5.0,
            max_range_pips=40.0,
            max_spread_pips=0.1,  # spread=0.2pips > 0.1 max; all blocked
            stop_pips=10.0,
            target_pips=20.0,
            max_label_bars=30,
        ),
        SweepPoint(
            label="short_only",
            min_range_pips=5.0,
            max_range_pips=40.0,
            direction_bias=TradeDirection.SHORT,  # OR is LONG; all blocked
            stop_pips=10.0,
            target_pips=20.0,
            max_label_bars=30,
        ),
    ]

    _section(f"Running sweep ({len(sweep_points)} configurations)")
    t0 = time.perf_counter()
    cmp = run_parameter_sweep(snaps, sweep_points)
    t1 = time.perf_counter()
    print(f"  Elapsed : {t1 - t0:.3f}s")

    ranked = cmp.ranked_by_win_rate()

    _section("Results (sorted by win rate, descending)")
    hdr = (
        f"  {'Label':<22}"
        f"  {'Cands':>6}"
        f"  {'Wins':>5}"
        f"  {'Win%':>7}"
        f"  {'Act%':>7}"
        f"  {'AvgMFE':>7}"
        f"  {'AvgMAE':>7}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in ranked:
        m = r.metrics
        print(
            f"  {r.sweep_point.label:<22}"
            f"  {m.candidate_count:>6}"
            f"  {m.win_count:>5}"
            f"  {_pct(m.win_rate_on_activated):>7}"
            f"  {_pct(m.activation_rate):>7}"
            f"  {_fp(m.avg_mfe):>7}"
            f"  {_fp(m.avg_mae):>7}"
        )

    _section("Observations")
    print("  baseline / target_25 : high win rate; bar.high=1.1045 reaches both targets")
    print("  target_30_timeout    : target above bar ceiling; all outcomes = TIMEOUT")
    print("  tight_range          : OR range=20pips > max_range=15; no candidates generated")
    print("  spread_strict        : spread=0.2pips > max=0.1; SpreadFilter blocks all")
    print("  short_only           : direction_bias=SHORT shifts entry below range_low=1.1000;")
    print("                         bars stay at 1.1035-1.1045 so entry never activates (0%)")
    print()


if __name__ == "__main__":
    main()
