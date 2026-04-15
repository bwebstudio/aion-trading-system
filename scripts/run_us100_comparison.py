"""
scripts/run_us100_comparison.py
--------------------------------
Run three strategy engines on real US100 M1 data and compare side-by-side:

  1. OpeningRangeRetestEngine  (new -- faithful to validated bots)
  2. OpeningRangeEngine        (old -- fires every bar after OR)
  3. VWAPFadeEngine            (mean-reversion baseline)

Each engine runs in its own isolated paper trading loop to avoid state
interference between stateful (Retest) and stateless engines.

Usage:
  python scripts/run_us100_comparison.py
"""

from __future__ import annotations

import sys
import time as time_mod
from collections import defaultdict
from datetime import date, datetime, time, timezone
from pathlib import Path

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aion.app import PaperTradingConfig, PaperTradingResult, run_paper_loop
from aion.app.summary import format_summary
from aion.core.constants import FEATURE_SET_VERSION, SNAPSHOT_VERSION
from aion.core.enums import AssetClass, SessionName, Timeframe, TradeDirection
from aion.core.ids import new_snapshot_id
from aion.core.models import (
    DataQualityReport,
    InstrumentSpec,
    MarketBar,
    MarketSnapshot,
)
from aion.data.csv_loader import load_bars
from aion.data.features import compute_feature_series
from aion.data.resampler import resample_bars
from aion.data.sessions import build_session_context
from aion.data.validator import validate_bars
from aion.risk.models import RiskProfile
from aion.strategies.models import OpeningRangeDefinition
from aion.strategies.opening_range import OpeningRangeEngine
from aion.strategies.or_range import OpeningRangeConfig, ORMethod
from aion.strategies.or_retest import OpeningRangeRetestEngine, RetestDefinition
from aion.strategies.vwap_fade import VWAPFadeDefinition, VWAPFadeEngine

_UTC = timezone.utc

# -----------------------------------------------------------------------------
# Instrument
# -----------------------------------------------------------------------------

_TICK_SIZE = 0.01
_PIP_MULTIPLIER = 100.0
_PIP_SIZE = _TICK_SIZE * _PIP_MULTIPLIER  # 1.0 point


def _us100() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="US100.cash",
        broker_symbol="US100.cash",
        asset_class=AssetClass.INDICES,
        price_timezone="America/New_York",
        market_timezone="America/New_York",
        broker_timezone="Etc/UTC",
        tick_size=_TICK_SIZE,
        point_value=1.0,
        contract_size=1.0,
        min_lot=0.01,
        lot_step=0.01,
        currency_profit="USD",
        currency_margin="USD",
        session_calendar="us_equity",
        trading_hours_label="Mon-Fri, nearly 24h",
    )


# -----------------------------------------------------------------------------
# Engines
# -----------------------------------------------------------------------------


def _retest_engine() -> OpeningRangeRetestEngine:
    """New OR Retest -- M1 block 9:30-9:34 ET (5 bars), SL=midpoint, 2R."""
    return OpeningRangeRetestEngine(
        RetestDefinition(
            strategy_id="or_retest_ny",
            session_name="NEW_YORK",
            or_config=OpeningRangeConfig(
                method=ORMethod.CANDLE_BLOCK,
                reference_time=time(9, 30),
                timezone_source="market",
                block_duration_minutes=5,
                block_timeframe=Timeframe.M1,
                min_range_points=5.0,
                max_range_points=200.0,
            ),
            rr_ratio=2.0,
            allow_fake_out_reversal=True,
        ),
        min_quality_score=0.0,
    )


def _old_or_engine() -> OpeningRangeEngine:
    """Old OR -- fires every bar after OR completed, fixed stop."""
    return OpeningRangeEngine(
        OpeningRangeDefinition(
            strategy_id="or_old_ny",
            session_name="NEW_YORK",
            min_range_pips=5.0,
            max_range_pips=200.0,
            pip_multiplier=_PIP_MULTIPLIER,
        ),
        min_quality_score=0.0,
    )


def _vwap_engine() -> VWAPFadeEngine:
    return VWAPFadeEngine(
        VWAPFadeDefinition(
            strategy_id="vwap_fade_overlap",
            session_name="OVERLAP_LONDON_NY",
            min_distance_to_vwap_pips=10.0,
            max_distance_to_vwap_pips=50.0,
            require_rejection=False,
            pip_multiplier=_PIP_MULTIPLIER,
            tick_size=_TICK_SIZE,
        ),
        min_quality_score=0.0,
    )


# -----------------------------------------------------------------------------
# Config -- one per engine type
# -----------------------------------------------------------------------------

_RISK_PROFILE = RiskProfile(
    account_equity=100_000.0,
    max_risk_per_trade_pct=1.0,
    max_daily_risk_pct=3.0,
    max_concurrent_positions=3,
    max_positions_per_strategy=2,
)


def _config_retest() -> PaperTradingConfig:
    """Retest: stop/target come from strategy_detail, but config needs
    sensible defaults for risk sizing.  15 pts ~ typical half-range."""
    return PaperTradingConfig(
        risk_profile=_RISK_PROFILE,
        instrument=_us100(),
        stop_distance_points=15.0,
        target_distance_points=30.0,
        pip_size=_PIP_SIZE,
        max_bars_open=120,
        slippage_points=4.0,
    )


def _config_old_or() -> PaperTradingConfig:
    return PaperTradingConfig(
        risk_profile=_RISK_PROFILE,
        instrument=_us100(),
        stop_distance_points=15.0,
        target_distance_points=30.0,
        pip_size=_PIP_SIZE,
        max_bars_open=120,
        slippage_points=4.0,
    )


def _config_vwap() -> PaperTradingConfig:
    return PaperTradingConfig(
        risk_profile=_RISK_PROFILE,
        instrument=_us100(),
        stop_distance_points=15.0,
        target_distance_points=30.0,
        pip_size=_PIP_SIZE,
        slippage_points=4.0,
        max_bars_open=60,
    )


# -----------------------------------------------------------------------------
# Snapshot builder (reused from run_us100_replay.py, simplified)
# -----------------------------------------------------------------------------

_ACTIVE_SESSIONS = {
    SessionName.LONDON,
    SessionName.NEW_YORK,
    SessionName.OVERLAP_LONDON_NY,
}

_WARMUP = 30


def _build_snapshots(bars_m1: list[MarketBar], instrument: InstrumentSpec):
    n = len(bars_m1)
    print(f"  Computing features for {n:,} M1 bars...")
    t0 = time_mod.perf_counter()
    features = compute_feature_series(
        bars_m1, Timeframe.M1,
        instrument.market_timezone, instrument.broker_timezone, "Etc/UTC",
    )
    print(f"  Features: {time_mod.perf_counter() - t0:.1f}s")

    print("  Resampling M5/M15...")
    bars_m5 = resample_bars(bars_m1, Timeframe.M5)
    bars_m15 = resample_bars(bars_m1, Timeframe.M15)

    print("  Sessions + quality...")
    sessions = [
        build_session_context(b.timestamp_utc, instrument.market_timezone,
                              instrument.broker_timezone, "Etc/UTC")
        for b in bars_m1
    ]

    # Quality per day
    day_bars: dict[date, list[MarketBar]] = defaultdict(list)
    for b, ctx in zip(bars_m1, sessions):
        day_bars[ctx.trading_day].append(b)
    daily_q: dict[date, DataQualityReport] = {}
    for day, db in day_bars.items():
        if len(db) >= 2:
            daily_q[day] = validate_bars(db, Timeframe.M1)

    # M5/M15 timestamp lists for bisect
    import bisect
    m5_ts = [b.timestamp_utc for b in bars_m5]
    m15_ts = [b.timestamp_utc for b in bars_m15]

    print("  Assembling snapshots...")
    snapshots = []
    for i in range(n):
        if i < _WARMUP:
            continue
        ctx = sessions[i]
        if ctx.session_name not in _ACTIVE_SESSIONS:
            continue
        q = daily_q.get(ctx.trading_day)
        if q is None:
            continue

        bar = bars_m1[i]
        ts = bar.timestamp_utc

        start_m1 = max(0, i + 1 - 100)
        snap_m1 = bars_m1[start_m1:i + 1]

        idx5 = bisect.bisect_right(m5_ts, ts)
        snap_m5 = bars_m5[max(0, idx5 - 100):idx5]

        idx15 = bisect.bisect_right(m15_ts, ts)
        snap_m15 = bars_m15[max(0, idx15 - 100):idx15]

        snapshots.append(MarketSnapshot(
            snapshot_id=new_snapshot_id(),
            symbol=instrument.symbol,
            timestamp_utc=ts,
            base_timeframe=Timeframe.M1,
            instrument=instrument,
            session_context=ctx,
            latest_bar=bar,
            bars_m1=snap_m1,
            bars_m5=snap_m5,
            bars_m15=snap_m15,
            feature_vector=features[i],
            quality_report=q,
            snapshot_version=SNAPSHOT_VERSION,
        ))

    return snapshots


# -----------------------------------------------------------------------------
# Result printer
# -----------------------------------------------------------------------------


def _print_result(label: str, result: PaperTradingResult, n_snapshots: int) -> None:
    s = result.summary
    closed = result.state.all_closed()

    total = s.positions_closed
    wr = f"{s.win_count}/{total} ({s.win_count/total*100:.1f}%)" if total > 0 else "n/a"
    avg_r = f"{s.avg_r_multiple:+.2f}R" if s.avg_r_multiple is not None else "n/a"

    print(f"\n  {label}")
    print(f"  {'-' * 50}")
    print(f"  Snapshots          : {n_snapshots:,}")
    print(f"  Signals            : {s.total_signals:,}")
    print(f"  Executed           : {s.total_executed:,}")
    print(f"  Closed             : {s.positions_closed:,}")
    print(f"  Still open         : {s.positions_still_open}")
    print(f"  Win rate           : {wr}")
    print(f"  Avg R              : {avg_r}")
    print(f"  Total P&L          : ${s.total_pnl:+,.2f}")
    print(f"  Elapsed (loop)     : {s.elapsed_seconds:.1f}s")

    # Sample trades
    if closed:
        print(f"  Sample trades (first 5 of {len(closed)}):")
        for cp in closed[:5]:
            d = cp.open_position.order.direction.value
            fill = cp.open_position.fill.fill_price
            close_p = cp.close_price
            r = cp.r_multiple
            reason = cp.close_reason.value
            ts = cp.open_position.fill.fill_timestamp.strftime("%Y-%m-%d %H:%M")
            print(f"    {ts}  {d:<5} fill={fill:>10.2f} close={close_p:>10.2f} "
                  f"R={r:+.2f} {reason}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compare OR Retest / OR Old / VWAP on US100 real data.",
    )
    parser.add_argument("--start-date", type=str, default=None,
                        help="First date to include (YYYY-MM-DD). Default: all.")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Last date to include (YYYY-MM-DD). Default: all.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    csv_path = _root / "data" / "raw" / "us100_3months_m1.csv"
    instrument = _us100()

    # -- Parse date filters ---------------------------------------------------
    start_date = date.fromisoformat(args.start_date) if args.start_date else None
    end_date = date.fromisoformat(args.end_date) if args.end_date else None

    if start_date and end_date and start_date > end_date:
        print(f"ERROR: --start-date ({start_date}) is after --end-date ({end_date}).")
        sys.exit(1)

    print()
    print("AION Strategy Comparison -- US100.cash M1 Real Data")
    print("=" * 55)

    # -- Load -----------------------------------------------------------------
    print("\nStep 1: Loading CSV...")
    t0 = time_mod.perf_counter()
    all_bars = load_bars(csv_path, instrument, drop_last=True)
    print(f"  {len(all_bars):,} bars total, {all_bars[0].timestamp_utc.date()} - "
          f"{all_bars[-1].timestamp_utc.date()} ({time_mod.perf_counter()-t0:.1f}s)")

    # -- Filter by date -------------------------------------------------------
    if start_date or end_date:
        bars = [
            b for b in all_bars
            if (start_date is None or b.timestamp_utc.date() >= start_date)
            and (end_date is None or b.timestamp_utc.date() <= end_date)
        ]
        label_start = start_date or all_bars[0].timestamp_utc.date()
        label_end = end_date or all_bars[-1].timestamp_utc.date()
        print(f"  Filtered to {label_start} - {label_end}: {len(bars):,} bars")
    else:
        bars = all_bars
        print("  Using full dataset (no date filter)")

    if not bars:
        print("ERROR: No bars in the selected date range.")
        sys.exit(1)

    # -- Build snapshots ------------------------------------------------------
    print("\nStep 2: Building snapshots...")
    t1 = time_mod.perf_counter()
    snapshots = _build_snapshots(bars, instrument)
    t_build = time_mod.perf_counter() - t1
    print(f"  {len(snapshots):,} snapshots ({t_build:.1f}s)")

    if not snapshots:
        print("ERROR: No snapshots generated. The date range may not overlap active sessions.")
        sys.exit(1)
    if len(snapshots) < 500:
        print(f"  WARNING: Only {len(snapshots):,} snapshots -- results may be noisy.")

    # -- Verify OR bar existence ----------------------------------------------
    print("\nStep 3: Verifying M1/M5 bars in snapshots...")
    sample = snapshots[500] if len(snapshots) > 500 else snapshots[-1]
    print(f"  Sample snapshot at {sample.timestamp_utc}:")
    print(f"    bars_m1: {len(sample.bars_m1)}")
    print(f"    bars_m5: {len(sample.bars_m5)}")
    print(f"    bars_m15: {len(sample.bars_m15)}")
    # Check for M1 bars at 9:30 market time
    m1_930 = [b for b in sample.bars_m1 if b.timestamp_market.time() == time(9, 30)]
    print(f"    M1 bars at 9:30 ET: {len(m1_930)}")
    if m1_930:
        b = m1_930[0]
        print(f"      H={b.high:.2f} L={b.low:.2f} range={b.high-b.low:.1f}")

    # -- Run engines ----------------------------------------------------------
    n = len(snapshots)

    print(f"\nStep 4: Running 3 engines on {n:,} snapshots each...")

    # Engine 1: OR Retest (new)
    print("\n  [1/3] OpeningRangeRetestEngine...")
    r_retest = run_paper_loop(snapshots, [_retest_engine()], _config_retest())

    # Engine 2: Old OR
    print("  [2/3] OpeningRangeEngine (old)...")
    r_old = run_paper_loop(snapshots, [_old_or_engine()], _config_old_or())

    # Engine 3: VWAP Fade
    print("  [3/3] VWAPFadeEngine...")
    r_vwap = run_paper_loop(snapshots, [_vwap_engine()], _config_vwap())

    # -- Results --------------------------------------------------------------
    print("\n" + "=" * 55)
    print("RESULTS")
    print("=" * 55)

    _print_result("OR Retest (new) -- 1 trade/session, SL=midpoint, 2R", r_retest, n)
    _print_result("OR Old -- fires every bar, fixed 15pt stop/30pt target", r_old, n)
    _print_result("VWAP Fade -- overlap session, 15pt stop/30pt target", r_vwap, n)

    # -- Comparison table -----------------------------------------------------
    print("\n" + "-" * 55)
    print("COMPARISON TABLE")
    print("-" * 55)
    print(f"{'Metric':<22} {'OR Retest':>12} {'OR Old':>12} {'VWAP':>12}")
    print(f"{'-'*22} {'-'*12} {'-'*12} {'-'*12}")

    for label, r in [("OR Retest", r_retest), ("OR Old", r_old), ("VWAP", r_vwap)]:
        pass  # just for reference

    s1, s2, s3 = r_retest.summary, r_old.summary, r_vwap.summary
    rows = [
        ("Signals", s1.total_signals, s2.total_signals, s3.total_signals),
        ("Executed", s1.total_executed, s2.total_executed, s3.total_executed),
        ("Closed", s1.positions_closed, s2.positions_closed, s3.positions_closed),
        ("Wins", s1.win_count, s2.win_count, s3.win_count),
        ("Losses", s1.loss_count, s2.loss_count, s3.loss_count),
    ]
    for label, v1, v2, v3 in rows:
        print(f"{label:<22} {v1:>12,} {v2:>12,} {v3:>12,}")

    # Win rate
    def _wr(s):
        return f"{s.win_count/s.positions_closed*100:.1f}%" if s.positions_closed > 0 else "n/a"
    print(f"{'Win rate':<22} {_wr(s1):>12} {_wr(s2):>12} {_wr(s3):>12}")

    # Avg R
    def _ar(s):
        return f"{s.avg_r_multiple:+.2f}R" if s.avg_r_multiple is not None else "n/a"
    print(f"{'Avg R':<22} {_ar(s1):>12} {_ar(s2):>12} {_ar(s3):>12}")

    # P&L
    print(f"{'Total P&L':<22} {'$'+f'{s1.total_pnl:+,.0f}':>12} "
          f"{'$'+f'{s2.total_pnl:+,.0f}':>12} {'$'+f'{s3.total_pnl:+,.0f}':>12}")

    # Signals per trade
    def _spt(s):
        return f"{s.total_signals/s.total_executed:.0f}" if s.total_executed > 0 else "n/a"
    print(f"{'Signals/trade':<22} {_spt(s1):>12} {_spt(s2):>12} {_spt(s3):>12}")

    print()
    print("=" * 55)
    total = time_mod.perf_counter() - t0
    print(f"Total time: {total:.0f}s")
    print()


if __name__ == "__main__":
    main()
