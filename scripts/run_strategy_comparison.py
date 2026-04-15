"""
scripts/run_strategy_comparison.py
────────────────────────────────────
Run OpeningRangeEngine and VWAPFadeEngine on the same snapshot set and
produce a side-by-side StrategyComparisonReport.

Usage:
  python scripts/run_strategy_comparison.py
  python scripts/run_strategy_comparison.py --snapshots 200

What this script does:
  1. Builds shared synthetic snapshots with an OR completed and VWAP far
     from close — so both strategies find candidates on the same bars.
  2. Runs run_strategy_comparison() with a shared LabelConfig.
  3. Prints overall metrics, by_session, and by_regime breakdowns.
  4. Selects the best config for each strategy using their respective sweeps,
     then prints both StrategyBaselineProfiles side by side.

Synthetic data characteristics:
  - OR range: 1.1000–1.1020 (20 pips). OR engine generates LONG entry at 1.1020.
  - Close: 1.1040. VWAP: 1.1010. Distance = 30 pips. VWAP engine generates
    SHORT entry at 1.1040.
  - Both strategies find candidates on the same snapshots → complementary
    signal: OR says LONG (breakout above range), VWAP says SHORT (reversion).
  - Bars: high=1.1045, low=1.1015.
      OR LONG  (entry=1.1020, stop=1.1010, target=1.1040): high≥1.1040 → WIN.
      VWAP SHORT (entry=1.1040, stop=1.1050, target=1.1020): low≤1.1020 → WIN.

Replace _build_snapshots() with your live data loader for real calibration.
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

from aion.analytics.baseline_selection import (
    select_best_opening_range_config,
    select_best_vwap_fade_config,
)
from aion.analytics.parameter_sweeps import run_parameter_sweep, run_vwap_parameter_sweep
from aion.analytics.replay_models import SweepPoint, VWAPSweepPoint
from aion.analytics.strategy_comparison import (
    ComparisonBreakdown,
    StrategyComparisonReport,
    run_strategy_comparison,
)
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
from aion.replay.models import LabelConfig
from aion.strategies.models import OpeningRangeDefinition
from aion.strategies.opening_range import OpeningRangeEngine
from aion.strategies.vwap_fade import VWAPFadeDefinition, VWAPFadeEngine

_UTC = timezone.utc
_BASE_TS = datetime(2024, 1, 15, 8, 0, 0, tzinfo=_UTC)


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot builder (shared dataset)
# ─────────────────────────────────────────────────────────────────────────────


def _build_snapshots(n: int = 150) -> list[MarketSnapshot]:
    """Build shared snapshots valid for both OR and VWAP strategies.

    First n//4 snapshots: OR active (OR engine skips, VWAP may trigger).
    Rest: OR completed (both strategies can trigger).

    Bar layout (post-OR window):
      OR range : low=1.1000, high=1.1020 (20 pip range)
      close    : 1.1040  (above OR, 30 pips above VWAP=1.1010)
      bar high : 1.1045  (reaches OR LONG target = entry+20 = 1.1040)
      bar low  : 1.1015  (reaches VWAP SHORT target = entry-20 = 1.1020)
    """
    instrument = _instrument()
    or_cutoff = n // 4
    snaps: list[MarketSnapshot] = []

    for i in range(n):
        ts = _BASE_TS + timedelta(minutes=i)
        or_active = i < or_cutoff
        or_completed = not or_active

        bar = MarketBar(
            symbol="EURUSD",
            timestamp_utc=ts,
            timestamp_market=ts,
            timeframe=Timeframe.M1,
            open=1.1040,
            high=1.1045,
            low=1.1015,
            close=1.1040,
            tick_volume=100.0,
            real_volume=0.0,
            spread=2.0,
            source=DataSource.SYNTHETIC,
        )
        session_open = ts.replace(hour=8, minute=0, second=0)
        session_close_t = ts.replace(hour=16, minute=30, second=0)
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
            session_close_utc=session_close_t,
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
            session_low=1.0980,
            opening_range_high=1.1020 if or_completed else None,
            opening_range_low=1.1000 if or_completed else None,
            vwap_session=1.1010,
            spread_mean_20=2.0,
            spread_zscore_20=0.0,
            return_1=0.0001,
            return_5=0.0003,
            candle_body=0.00003,
            upper_wick=0.00002,
            lower_wick=0.00025,
            distance_to_session_high=1.1060 - 1.1040,
            distance_to_session_low=1.1040 - 1.0980,
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


def _instrument() -> InstrumentSpec:
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


# ─────────────────────────────────────────────────────────────────────────────
# Engine builders
# ─────────────────────────────────────────────────────────────────────────────


def _or_engine() -> OpeningRangeEngine:
    or_def = OpeningRangeDefinition(
        strategy_id="or_london_v1",
        session_name="LONDON",
        min_range_pips=5.0,
        max_range_pips=40.0,
    )
    return OpeningRangeEngine(or_def, min_quality_score=MIN_QUALITY_SCORE)


def _vwap_engine() -> VWAPFadeEngine:
    vwap_def = VWAPFadeDefinition(
        strategy_id="vwap_fade_london_v1",
        session_name="LONDON",
        min_distance_to_vwap_pips=10.0,
        max_distance_to_vwap_pips=50.0,
    )
    return VWAPFadeEngine(vwap_def, min_quality_score=MIN_QUALITY_SCORE)


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────


def _section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _pct(v: float | None) -> str:
    return f"{v * 100:.1f}%" if v is not None else "  N/A "


def _fp(v: float | None, d: int = 1) -> str:
    return f"{v:.{d}f}" if v is not None else "N/A"


def _print_breakdown(bd: ComparisonBreakdown, label_width: int = 12) -> None:
    a = bd.strategy_a
    b = bd.strategy_b
    print(f"  Group : {bd.group_key}")
    print(
        f"    {'':>{label_width}}  {'OpeningRange':>14}  {'VWAPFade':>14}"
    )
    print(
        f"    {'Candidates':>{label_width}}  {a.candidate_count:>14}  {b.candidate_count:>14}"
    )
    print(
        f"    {'Activated':>{label_width}}  {a.entry_activated_count:>14}  {b.entry_activated_count:>14}"
    )
    print(
        f"    {'WinRate':>{label_width}}  {_pct(a.win_rate_on_activated):>14}  {_pct(b.win_rate_on_activated):>14}"
    )
    print(
        f"    {'ActRate':>{label_width}}  {_pct(a.activation_rate):>14}  {_pct(b.activation_rate):>14}"
    )
    print(
        f"    {'AvgMFE':>{label_width}}  {_fp(a.avg_mfe):>14}  {_fp(b.avg_mfe):>14}"
    )
    print(
        f"    {'AvgMAE':>{label_width}}  {_fp(a.avg_mae):>14}  {_fp(b.avg_mae):>14}"
    )
    print(
        f"    {'ResoBars':>{label_width}}  {_fp(a.avg_bars_to_resolution):>14}  {_fp(b.avg_bars_to_resolution):>14}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OR and VWAP strategies.")
    parser.add_argument("--snapshots", type=int, default=150)
    args = parser.parse_args()

    _section("Building shared synthetic snapshots")
    snaps = _build_snapshots(args.snapshots)
    or_cutoff = args.snapshots // 4
    print(f"  Total     : {len(snaps)} snapshots")
    print(f"  OR active : first {or_cutoff} (OR engine skips, VWAP may trigger)")
    print(f"  Post-OR   : {args.snapshots - or_cutoff} snaps where both can trigger")
    print(f"  Dataset   : OR=20pip, close=1.1040, VWAP=1.1010 (30-pip distance)")
    print(f"  Bar       : high=1.1045 low=1.1015  => both strategies WIN")

    engine_a = _or_engine()
    engine_b = _vwap_engine()
    label_cfg = LabelConfig(stop_pips=10.0, target_pips=20.0, max_bars=20)

    _section("Running strategy comparison")
    t0 = time.perf_counter()
    report = run_strategy_comparison(snaps, engine_a, engine_b, label_config=label_cfg)
    t1 = time.perf_counter()
    print(f"  Elapsed : {t1 - t0:.3f}s")
    print(f"  Engine A: {report.strategy_a_id}")
    print(f"  Engine B: {report.strategy_b_id}")

    # ── Overall ───────────────────────────────────────────────────────────────
    _section("Overall comparison")
    _print_breakdown(report.overall)

    # ── By session ────────────────────────────────────────────────────────────
    if report.by_session:
        _section("By session")
        for bd in report.by_session:
            _print_breakdown(bd)
            print()

    # ── By regime ─────────────────────────────────────────────────────────────
    if report.by_regime:
        _section("By regime")
        for bd in report.by_regime:
            _print_breakdown(bd)
            print()

    # ── Baseline profiles from dedicated sweeps ───────────────────────────────
    _section("Best configuration per strategy (from dedicated parameter sweeps)")

    # OR sweep
    or_sweep_points = [
        SweepPoint(
            label="or_london_20_40",
            min_range_pips=10.0,
            max_range_pips=40.0,
            allowed_sessions=frozenset({"LONDON", "OVERLAP_LONDON_NY"}),
            stop_pips=10.0,
            target_pips=20.0,
            max_label_bars=20,
        ),
        SweepPoint(
            label="or_all_20_60",
            min_range_pips=10.0,
            max_range_pips=60.0,
            allowed_sessions=None,
            stop_pips=10.0,
            target_pips=20.0,
            max_label_bars=20,
        ),
    ]
    or_cmp = run_parameter_sweep(snaps, or_sweep_points)
    or_result = select_best_opening_range_config(or_cmp, min_candidates=1)

    # VWAP sweep
    vwap_sweep_points = [
        VWAPSweepPoint(
            label="vwap_london_10",
            min_distance_to_vwap_pips=10.0,
            session_name="LONDON",
            stop_pips=10.0,
            target_pips=20.0,
            max_label_bars=20,
        ),
        VWAPSweepPoint(
            label="vwap_all_10",
            min_distance_to_vwap_pips=10.0,
            session_name="ALL",
            stop_pips=10.0,
            target_pips=20.0,
            max_label_bars=20,
        ),
    ]
    vwap_cmp = run_vwap_parameter_sweep(snaps, vwap_sweep_points)
    vwap_result = select_best_vwap_fade_config(vwap_cmp, min_candidates=1)

    print()
    if or_result:
        _, or_profile = or_result
        print(f"  OpeningRange best config:")
        print(f"    strategy_id              : {or_profile.strategy_id}")
        print(f"    session                  : {or_profile.session}")
        print(f"    activation_rate          : {_pct(or_profile.activation_rate)}")
        print(f"    win_rate                 : {_pct(or_profile.win_rate)}")
        print(f"    avg_mfe                  : {_fp(or_profile.avg_mfe)} pips")
        print(f"    avg_mae                  : {_fp(or_profile.avg_mae)} pips")
        print(f"    expected_resolution_bars : {_fp(or_profile.expected_resolution_bars)}")

    print()
    if vwap_result:
        _, vwap_profile = vwap_result
        print(f"  VWAPFade best config:")
        print(f"    strategy_id              : {vwap_profile.strategy_id}")
        print(f"    session                  : {vwap_profile.session}")
        print(f"    activation_rate          : {_pct(vwap_profile.activation_rate)}")
        print(f"    win_rate                 : {_pct(vwap_profile.win_rate)}")
        print(f"    avg_mfe                  : {_fp(vwap_profile.avg_mfe)} pips")
        print(f"    avg_mae                  : {_fp(vwap_profile.avg_mae)} pips")
        print(f"    expected_resolution_bars : {_fp(vwap_profile.expected_resolution_bars)}")

    print()


if __name__ == "__main__":
    main()
