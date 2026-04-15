"""
scripts/run_replay_analysis.py
───────────────────────────────
Run a historical replay and print a full analytics report.

Usage:
  python scripts/run_replay_analysis.py
  python scripts/run_replay_analysis.py --snapshots 150 --stop-pips 10 --target-pips 20

What this script does:
  1. Builds synthetic MarketSnapshots.
  2. Runs OpeningRangeEngine replay with RuleBasedRegimeDetector.
  3. Computes metrics and builds a structured analytics report.
  4. Prints overall metrics + all breakdowns to stdout.
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

from aion.analytics.replay_metrics import compute_metrics
from aion.analytics.replay_reports import build_report
from aion.core.constants import FEATURE_SET_VERSION, MIN_QUALITY_SCORE, SNAPSHOT_VERSION
from aion.core.enums import AssetClass, DataSource, SessionName, Timeframe
from aion.core.ids import new_snapshot_id
from aion.core.models import (
    DataQualityReport,
    FeatureVector,
    InstrumentSpec,
    MarketBar,
    MarketSnapshot,
    SessionContext,
)
from aion.regime.rules import RuleBasedRegimeDetector
from aion.replay.models import LabelConfig
from aion.replay.runner import run_replay
from aion.strategies.models import OpeningRangeDefinition
from aion.strategies.opening_range import OpeningRangeEngine

_UTC = timezone.utc
_BASE_TS = datetime(2024, 1, 15, 8, 0, 0, tzinfo=_UTC)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic snapshot builder
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
    parser = argparse.ArgumentParser(description="Run replay + analytics report.")
    parser.add_argument("--snapshots", type=int, default=100)
    parser.add_argument("--stop-pips", type=float, default=10.0)
    parser.add_argument("--target-pips", type=float, default=20.0)
    parser.add_argument("--max-label-bars", type=int, default=30)
    args = parser.parse_args()

    _section("Building synthetic snapshots")
    t0 = time.perf_counter()
    snaps = _build_snapshots(args.snapshots)
    print(f"  Generated : {len(snaps)} snapshots")
    print(f"  Symbol    : EURUSD  |  OR range: 20 pips (1.1000 - 1.1020)")
    print(f"  Session   : LONDON  |  OR window: first 30 bars")

    engine = OpeningRangeEngine(
        OpeningRangeDefinition(
            strategy_id="or_london_v1",
            session_name="LONDON",
            min_range_pips=5.0,
            max_range_pips=40.0,
        ),
        min_quality_score=MIN_QUALITY_SCORE,
    )
    label_cfg = LabelConfig(
        stop_pips=args.stop_pips,
        target_pips=args.target_pips,
        max_bars=args.max_label_bars,
    )
    detector = RuleBasedRegimeDetector()

    _section("Running replay")
    result = run_replay(snaps, engine, regime_detector=detector, label_config=label_cfg)
    t1 = time.perf_counter()
    print(f"  Run ID    : {result.summary.run_id}")
    print(f"  Elapsed   : {t1 - t0:.3f}s")

    _section("Building analytics report")
    report = build_report(result)
    m = report.overall_metrics

    _section("Overall Metrics")
    print(f"  Total records        : {m.total_records}")
    print(f"  Candidates           : {m.candidate_count}")
    print(f"  No-trade             : {m.no_trade_count}")
    print(f"  Insufficient data    : {m.insufficient_data_count}")
    print(f"  Total labeled        : {m.total_labeled}")
    print(f"  Entry activated      : {m.entry_activated_count}")
    print(f"  Activation rate      : {_pct(m.activation_rate)}")
    print(f"  Win rate (activated) : {_pct(m.win_rate_on_activated)}")
    print(f"    Wins               : {m.win_count}")
    print(f"    Losses             : {m.loss_count}")
    print(f"    Timeouts           : {m.timeout_count}")
    print(f"    Not activated      : {m.entry_not_activated_count}")
    print(f"  Avg MFE              : {_fp(m.avg_mfe)} pips")
    print(f"  Avg MAE              : {_fp(m.avg_mae)} pips")
    print(f"  Avg bars to entry    : {_fp(m.avg_bars_to_entry)}")
    print(f"  Avg bars to resolve  : {_fp(m.avg_bars_to_resolution)}")

    if report.by_session:
        _section("Breakdown by Session")
        for row in report.by_session:
            print(
                f"  {row.group_key:<15}"
                f"  cands={row.candidate_count:3d}"
                f"  wins={row.win_count:3d}"
                f"  win%={_pct(row.win_rate):>6}"
                f"  act%={_pct(row.activation_rate):>6}"
            )

    if report.by_regime:
        _section("Breakdown by Regime")
        for row in report.by_regime:
            print(
                f"  {row.group_key:<15}"
                f"  records={row.record_count:3d}"
                f"  cands={row.candidate_count:3d}"
                f"  wins={row.win_count:3d}"
                f"  win%={_pct(row.win_rate):>6}"
            )

    if report.top_reason_codes:
        _section("Top Reason Codes (No-Trade)")
        for code, cnt in report.top_reason_codes:
            bar_width = min(cnt, 30)
            bar = "#" * bar_width
            print(f"  {code:<30}  {cnt:4d}  {bar}")

    if report.by_direction:
        _section("Breakdown by Direction")
        for row in report.by_direction:
            print(
                f"  {row.group_key:<8}"
                f"  cands={row.candidate_count:3d}"
                f"  wins={row.win_count:3d}"
                f"  win%={_pct(row.win_rate):>6}"
            )

    print()


if __name__ == "__main__":
    main()
