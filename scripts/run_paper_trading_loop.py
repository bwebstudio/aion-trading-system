"""
scripts/run_paper_trading_loop.py
──────────────────────────────────
Demonstration of the full paper trading loop (aion.app).

Shows how all AION components work together end-to-end:
  Market snapshots → Strategy engine → Risk allocator
  → Order creation → Fill → Position management → Close → Summary

Scenario
─────────
  EURUSD, LONDON session, Opening Range breakout strategy.
  Account: $10,000 | Risk: 1% per trade | Max 2 concurrent positions.
  Stop: 10 pips | Target: 20 pips (2R) | Timeout: 10 bars

  Two trades are generated across 10 synthetic snapshots:

    [0]  OR completed, price above OR high  → LONG signal queued
    [1]  Fill at 1.10220  — price holds (no close)
    [2]  Price drifts up, no trigger
    [3]  Target hit → TAKE_PROFIT (+2R)
    [4]  Price drops below OR low  → SHORT signal queued
    [5]  Fill at 1.09990  — price holds (no close)
    [6]  Price continues down, no trigger
    [7]  Stop hit → STOP_LOSS (-1R)
    [8-9] Quiet bars, no signal

Usage:
  python scripts/run_paper_trading_loop.py

Output is plain text, readable by a non-technical user.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aion.app import PaperTradingConfig, format_summary, run_paper_loop
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
from aion.risk.models import RiskProfile
from aion.strategies.models import OpeningRangeDefinition
from aion.strategies.opening_range import OpeningRangeEngine

_UTC = timezone.utc
_TS_BASE = datetime(2024, 1, 15, 8, 30, 0, tzinfo=_UTC)

# EURUSD OR levels
_OR_HIGH = 1.1020
_OR_LOW  = 1.1000
_PIP     = 0.0001  # tick_size(0.00001) * pip_multiplier(10)
_STOP    = 10.0    # pips  → stop_price LONG = OR_HIGH - 10 * 0.0001 = 1.1010
_TARGET  = 20.0    # pips  → target_price LONG = OR_HIGH + 20 * 0.0001 = 1.1040


# ─────────────────────────────────────────────────────────────────────────────
# Instrument
# ─────────────────────────────────────────────────────────────────────────────


def _eurusd() -> InstrumentSpec:
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
# Snapshot factory
# ─────────────────────────────────────────────────────────────────────────────


def _bar(
    ts: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> MarketBar:
    return MarketBar(
        symbol="EURUSD",
        timestamp_utc=ts,
        timestamp_market=ts,
        timeframe=Timeframe.M1,
        open=open_,
        high=high,
        low=low,
        close=close,
        tick_volume=200.0,
        real_volume=0.0,
        spread=2.0,
        source=DataSource.SYNTHETIC,
    )


def _session(ts: datetime, session_name: SessionName, or_completed: bool) -> SessionContext:
    is_open = session_name != SessionName.OFF_HOURS
    is_london = session_name in (SessionName.LONDON, SessionName.OVERLAP_LONDON_NY)
    is_ny = session_name in (SessionName.NEW_YORK, SessionName.OVERLAP_LONDON_NY)
    return SessionContext(
        trading_day=ts.date(),
        broker_time=ts,
        market_time=ts,
        local_time=ts,
        is_asia=session_name == SessionName.ASIA,
        is_london=is_london,
        is_new_york=is_ny,
        is_session_open_window=is_open,
        opening_range_active=False,
        opening_range_completed=or_completed,
        session_name=session_name,
        session_open_utc=ts.replace(hour=8, minute=0, second=0) if is_open else None,
        session_close_utc=ts.replace(hour=16, minute=30, second=0) if is_open else None,
    )


def _fv(ts: datetime, or_high: float = _OR_HIGH, or_low: float = _OR_LOW) -> FeatureVector:
    return FeatureVector(
        symbol="EURUSD",
        timestamp_utc=ts,
        timeframe=Timeframe.M1,
        atr_14=0.00015,
        rolling_range_10=0.0010,
        rolling_range_20=0.0020,
        volatility_percentile_20=0.50,
        session_high=1.1060,
        session_low=0.9990,
        opening_range_high=or_high,
        opening_range_low=or_low,
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


def _quality() -> DataQualityReport:
    return DataQualityReport(
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        rows_checked=200,
        missing_bars=0,
        duplicate_timestamps=0,
        out_of_order_rows=0,
        stale_bars=0,
        spike_bars=0,
        null_rows=0,
        quality_score=1.0,
        warnings=[],
    )


def _snapshot(
    i: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    or_completed: bool = True,
    session_name: SessionName = SessionName.LONDON,
    or_high: float = _OR_HIGH,
    or_low: float = _OR_LOW,
) -> MarketSnapshot:
    ts = _TS_BASE + timedelta(minutes=i)
    b = _bar(ts, open_=open_, high=high, low=low, close=close)
    return MarketSnapshot(
        snapshot_id=new_snapshot_id(),
        symbol="EURUSD",
        timestamp_utc=ts,
        base_timeframe=Timeframe.M1,
        instrument=_eurusd(),
        session_context=_session(ts, session_name, or_completed),
        latest_bar=b,
        bars_m1=[b],
        bars_m5=[],
        bars_m15=[],
        feature_vector=_fv(ts, or_high=or_high, or_low=or_low),
        quality_report=_quality(),
        snapshot_version=SNAPSHOT_VERSION,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Demo scenario: build 10 synthetic snapshots
# ─────────────────────────────────────────────────────────────────────────────


def _build_snapshots() -> list[MarketSnapshot]:
    """
    Ten-bar LONDON session sequence.

    Bar | Price action                         | Expected loop event
    ────┼─────────────────────────────────────┼──────────────────────────────
     0  | OR completed, close above OR high    | LONG signal → order queued
     1  | open=1.10220 — price holds           | Fill at 1.10220, no close
     2  | 1.10230 – 1.10350 / 1.10190         | Open, no trigger
     3  | 1.10310 – 1.10420 / 1.10290         | Target 1.1040 hit → TAKE_PROFIT
     4  | price at OR low, close below OR low  | SHORT signal → order queued
     5  | open=1.09990 — price holds           | Fill at 1.09990, no close
     6  | 1.09960 – 1.09990 / 1.09880         | Open, no trigger
     7  | 1.09970 – 1.10320 / 1.09920         | Stop 1.1010 hit → STOP_LOSS
     8  | quiet                                | No signal
     9  | quiet                                | No signal
    """
    return [
        # ── Trade 1: LONG setup ────────────────────────────────────────────
        # [0] OR completed, bar is above OR high → OR engine issues LONG signal
        _snapshot(0, open_=1.1020, high=1.1028, low=1.1018, close=1.1025),

        # [1] Fill bar: open=1.10220 (fills LONG order), stays between stop/target
        _snapshot(1, open_=1.1022, high=1.1030, low=1.1015, close=1.1025),

        # [2] Price drifts up but doesn't reach target (1.1040)
        _snapshot(2, open_=1.1023, high=1.1035, low=1.1019, close=1.1030),

        # [3] Bar high reaches 1.1042 → target (1.1040) hit → TAKE_PROFIT
        _snapshot(3, open_=1.1031, high=1.1042, low=1.1028, close=1.1040),

        # ── Trade 2: SHORT setup (new OR low) ─────────────────────────────
        # [4] OR completed, bar closes below OR low → OR engine issues SHORT signal
        _snapshot(
            4,
            open_=1.1000, high=1.1005, low=1.0995, close=1.0998,
            or_high=1.1020, or_low=1.1000,  # same OR levels — short side
        ),

        # [5] Fill bar: open=1.09990 (fills SHORT order), stays between stop/target
        _snapshot(5, open_=1.0999, high=1.1005, low=1.0992, close=1.0998),

        # [6] Price continues lower, doesn't reach stop (1.1010) or target (1.0980)
        _snapshot(6, open_=1.0996, high=1.0999, low=1.0988, close=1.0992),

        # [7] Bar high reaches 1.1032 → stop (1.1010) hit → STOP_LOSS
        _snapshot(7, open_=1.0997, high=1.1032, low=1.0992, close=1.1028),

        # ── Quiet bars ────────────────────────────────────────────────────
        # [8-9] OR still completed but no fresh breakout beyond OR high/low
        _snapshot(8, open_=1.1020, high=1.1025, low=1.1015, close=1.1020),
        _snapshot(9, open_=1.1018, high=1.1023, low=1.1013, close=1.1018),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Engines
# ─────────────────────────────────────────────────────────────────────────────


def _engines() -> list:
    """Two OR engines: one LONG-biased, one SHORT-biased, same session/range."""
    long_engine = OpeningRangeEngine(
        OpeningRangeDefinition(
            strategy_id="or_london_long",
            session_name="LONDON",
            min_range_pips=5.0,
            max_range_pips=40.0,
            direction_bias=TradeDirection.LONG,
        )
    )
    short_engine = OpeningRangeEngine(
        OpeningRangeDefinition(
            strategy_id="or_london_short",
            session_name="LONDON",
            min_range_pips=5.0,
            max_range_pips=40.0,
            direction_bias=TradeDirection.SHORT,
        )
    )
    return [long_engine, short_engine]


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


def _config() -> PaperTradingConfig:
    return PaperTradingConfig(
        risk_profile=RiskProfile(
            account_equity=10_000.0,
            max_risk_per_trade_pct=1.0,
            max_daily_risk_pct=5.0,
            max_concurrent_positions=2,
            max_positions_per_strategy=2,
        ),
        instrument=_eurusd(),
        stop_distance_points=_STOP,
        target_distance_points=_TARGET,
        pip_size=_PIP,
        max_bars_open=10,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────


def _section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print()
    print("Paper Trading Loop v1 - Full Pipeline Demo")
    print("=" * 50)
    print("Account   : $10,000")
    print("Instrument: EURUSD")
    print("Risk      : 1% per trade | Max 2 concurrent positions")
    print("Strategy  : OR LONG + OR SHORT (London session)")
    print(f"Stop      : {_STOP:.0f} pips | Target: {_TARGET:.0f} pips ({_TARGET/_STOP:.0f}R)")
    print(f"OR range  : {_OR_LOW:.4f} – {_OR_HIGH:.4f}")
    print(f"Fill model: next-bar-open | Slippage: 0")

    snapshots = _build_snapshots()
    engines = _engines()
    config = _config()

    result = run_paper_loop(snapshots, engines, config)

    # ── Summary ────────────────────────────────────────────────────────────
    _section("Summary")
    print(format_summary(result))

    # ── Journal events ─────────────────────────────────────────────────────
    _section("Journal (all events in order)")
    for e in result.journal.all_events():
        print(f"  [{e.event_type.value:<18}] {e.reason_text}")

    # ── Closed positions detail ─────────────────────────────────────────────
    closed = result.state.all_closed()
    if closed:
        _section("Closed Positions Detail")
        for cp in closed:
            sid = cp.open_position.order.strategy_id
            direction = cp.open_position.order.direction.value
            fill = cp.open_position.fill.fill_price
            close = cp.close_price
            pnl_sign = "+" if cp.pnl_amount >= 0 else ""
            r_sign = "+" if cp.r_multiple >= 0 else ""
            print(
                f"  {sid:<20} {direction:<5} "
                f"fill={fill:.5f}  close={close:.5f}  "
                f"P&L={pnl_sign}${cp.pnl_amount:.2f}  "
                f"R={r_sign}{cp.r_multiple:.2f}  "
                f"({cp.close_reason.value})"
            )

    # ── Still-open positions ────────────────────────────────────────────────
    open_positions = result.state.all_open()
    if open_positions:
        _section(f"Still Open at End of Data ({len(open_positions)} position(s))")
        for p in open_positions:
            sid = p.order.strategy_id
            direction = p.order.direction.value
            fill = p.fill.fill_price
            stop = p.order.stop_price
            target = p.order.target_price
            target_str = f"{target:.5f}" if target is not None else "none"
            print(
                f"  {sid:<20} {direction:<5} "
                f"fill={fill:.5f}  stop={stop:.5f}  target={target_str}"
            )

    print()


if __name__ == "__main__":
    main()
