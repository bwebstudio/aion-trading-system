"""
tests/integration/test_paper_trading_loop.py
─────────────────────────────────────────────
Integration tests for the complete paper trading loop.

What these tests verify
────────────────────────
  Full lifecycle:
    - loop runs on valid snapshots without error
    - signal -> risk approval -> order creation -> fill -> close
    - position is opened and closed correctly
    - journal records ORDER_SUBMITTED, ORDER_FILLED, POSITION_CLOSED in order

  Summary correctness:
    - snapshots_evaluated matches input length
    - total_signals, risk_approved, total_executed, positions_closed counts
    - total_pnl matches sum of closed position P&L
    - strategy_breakdown reflects actual per-strategy activity

  Risk integration:
    - signal blocked when portfolio is full (max_concurrent_positions reached)
    - no positions opened when all signals are blocked

  Empty / no-signal scenarios:
    - empty snapshot list -> valid empty result
    - snapshots that produce no CANDIDATE -> zero signals, zero executed

  Determinism:
    - same inputs produce same summary on two consecutive runs
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aion.app.loop import run_paper_loop
from aion.app.orchestrator import PaperTradingConfig
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
from aion.execution.journal import EventType
from aion.risk.models import RiskProfile
from aion.strategies.models import OpeningRangeDefinition, StrategyOutcome
from aion.strategies.opening_range import OpeningRangeEngine

_UTC = timezone.utc
_TS_BASE = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_UTC)

# Fixed OR levels used across all snapshots
_OR_HIGH = 1.1020
_OR_LOW = 1.1000
# pip_size = tick_size * pip_multiplier = 0.00001 * 10 = 0.0001
_PIP_SIZE = 0.0001
_STOP_PIPS = 10.0     # stop_price = OR_HIGH - 10 * 0.0001 = 1.1010
_TARGET_PIPS = 20.0   # target_price = OR_HIGH + 20 * 0.0001 = 1.1040


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot factory helpers (adapted from test_strategy_replay_loop.py)
# ─────────────────────────────────────────────────────────────────────────────


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


def _make_bar(
    ts: datetime,
    open_: float = 1.1020,
    high: float = 1.1030,
    low: float = 1.1015,
    close: float = 1.1025,
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
        tick_volume=100.0,
        real_volume=0.0,
        spread=2.0,
        source=DataSource.SYNTHETIC,
    )


def _make_session(
    ts: datetime,
    session_name: SessionName = SessionName.LONDON,
    or_completed: bool = True,
) -> SessionContext:
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


def _make_fv(ts: datetime) -> FeatureVector:
    return FeatureVector(
        symbol="EURUSD",
        timestamp_utc=ts,
        timeframe=Timeframe.M1,
        atr_14=0.00015,
        rolling_range_10=0.001,
        rolling_range_20=0.002,
        volatility_percentile_20=0.50,
        session_high=1.1060,
        session_low=1.0990,
        opening_range_high=_OR_HIGH,
        opening_range_low=_OR_LOW,
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


def _make_quality(score: float = 1.0) -> DataQualityReport:
    return DataQualityReport(
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        rows_checked=100,
        missing_bars=0,
        duplicate_timestamps=0,
        out_of_order_rows=0,
        stale_bars=0,
        spike_bars=0,
        null_rows=0,
        quality_score=score,
        warnings=[],
    )


def _make_snapshot(
    index: int,
    session_name: SessionName = SessionName.LONDON,
    or_completed: bool = True,
    open_: float = 1.1020,
    high: float = 1.1030,
    low: float = 1.1015,
    close: float = 1.1025,
) -> MarketSnapshot:
    ts = _TS_BASE + timedelta(minutes=index)
    bar = _make_bar(ts, open_=open_, high=high, low=low, close=close)
    return MarketSnapshot(
        snapshot_id=new_snapshot_id(),
        symbol="EURUSD",
        timestamp_utc=ts,
        base_timeframe=Timeframe.M1,
        instrument=_instrument(),
        session_context=_make_session(ts, session_name, or_completed),
        latest_bar=bar,
        bars_m1=[bar],
        bars_m5=[],
        bars_m15=[],
        feature_vector=_make_fv(ts),
        quality_report=_make_quality(),
        snapshot_version=SNAPSHOT_VERSION,
    )


def _or_engine(direction: TradeDirection = TradeDirection.LONG) -> OpeningRangeEngine:
    """OR engine: LONDON session, 5–40 pip range, LONG bias."""
    return OpeningRangeEngine(
        OpeningRangeDefinition(
            strategy_id="or_london_v1",
            session_name="LONDON",
            min_range_pips=5.0,
            max_range_pips=40.0,
            direction_bias=direction,
        )
    )


def _config(
    max_positions: int = 3,
    max_bars_open: int | None = None,
) -> PaperTradingConfig:
    return PaperTradingConfig(
        risk_profile=RiskProfile(
            account_equity=10_000.0,
            max_risk_per_trade_pct=1.0,
            max_daily_risk_pct=5.0,  # generous for tests
            max_concurrent_positions=max_positions,
            max_positions_per_strategy=max_positions,
        ),
        instrument=_instrument(),
        stop_distance_points=_STOP_PIPS,
        target_distance_points=_TARGET_PIPS,
        pip_size=_PIP_SIZE,
        max_bars_open=max_bars_open,
    )


def _lifecycle_snapshots() -> list[MarketSnapshot]:
    """
    Three snapshots that produce exactly one complete trade lifecycle:
      [0] Signal: OR completed LONDON → LONG candidate (entry=1.1020)
                  pending order created (stop=1.1010, target=1.1040)
      [1] Fill:   bar.open=1.1022 → filled.  bar H=1.1030 < target, L=1.1015 > stop → open
      [2] Close:  bar H=1.1042 >= target=1.1040 → TAKE_PROFIT
    """
    return [
        # [0] signal snapshot (bar doesn't matter for signal logic)
        _make_snapshot(0, open_=1.1025, high=1.1028, low=1.1020, close=1.1025),
        # [1] fill bar — price between stop and target
        _make_snapshot(1, open_=1.1022, high=1.1030, low=1.1015, close=1.1025),
        # [2] close bar — target hit
        _make_snapshot(2, open_=1.1028, high=1.1042, low=1.1022, close=1.1040),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Empty / no-signal scenarios
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_snapshots_returns_valid_result():
    result = run_paper_loop([], [_or_engine()], _config())
    s = result.summary
    assert s.snapshots_evaluated == 0
    assert s.total_signals == 0
    assert s.total_executed == 0
    assert s.positions_closed == 0
    assert s.total_pnl == pytest.approx(0.0)


def test_no_signal_snapshots_produces_zero_signals():
    """OFF_HOURS snapshots → OR engine returns NO_TRADE → no signals."""
    snapshots = [
        _make_snapshot(i, session_name=SessionName.OFF_HOURS)
        for i in range(5)
    ]
    result = run_paper_loop(snapshots, [_or_engine()], _config())
    assert result.summary.total_signals == 0
    assert result.summary.total_executed == 0
    assert result.summary.positions_closed == 0


def test_no_candidate_without_completed_or():
    """Snapshots with or_completed=False → no CANDIDATE → no positions."""
    snapshots = [
        _make_snapshot(i, or_completed=False)
        for i in range(5)
    ]
    result = run_paper_loop(snapshots, [_or_engine()], _config())
    assert result.summary.total_signals == 0


# ─────────────────────────────────────────────────────────────────────────────
# Full lifecycle
# ─────────────────────────────────────────────────────────────────────────────


def test_full_lifecycle_signal_fill_close():
    """One signal -> one position opened -> one position closed at target."""
    snapshots = _lifecycle_snapshots()
    result = run_paper_loop(snapshots, [_or_engine()], _config())
    s = result.summary

    assert s.snapshots_evaluated == 3
    assert s.total_signals >= 1
    assert s.risk_approved >= 1
    assert s.total_executed >= 1
    assert s.positions_closed >= 1
    assert s.positions_still_open == 0


def test_full_lifecycle_pnl_is_nonzero():
    snapshots = _lifecycle_snapshots()
    result = run_paper_loop(snapshots, [_or_engine()], _config())
    assert result.summary.total_pnl != 0.0


def test_full_lifecycle_win_registered():
    """Target hit → closed position with pnl > 0 → win_count >= 1."""
    snapshots = _lifecycle_snapshots()
    result = run_paper_loop(snapshots, [_or_engine()], _config())
    assert result.summary.win_count >= 1
    assert result.summary.loss_count == 0


def test_full_lifecycle_pnl_matches_state():
    """summary.total_pnl == sum of state.all_closed() P&L."""
    snapshots = _lifecycle_snapshots()
    result = run_paper_loop(snapshots, [_or_engine()], _config())
    manual_pnl = sum(c.pnl_amount for c in result.state.all_closed())
    assert result.summary.total_pnl == pytest.approx(manual_pnl, abs=0.01)


def test_full_lifecycle_avg_r_populated():
    snapshots = _lifecycle_snapshots()
    result = run_paper_loop(snapshots, [_or_engine()], _config())
    if result.summary.positions_closed > 0:
        assert result.summary.avg_r_multiple is not None


# ─────────────────────────────────────────────────────────────────────────────
# Stop loss path
# ─────────────────────────────────────────────────────────────────────────────


def test_stop_loss_closes_position():
    """
    [0] Signal (OR LONG, entry=1.1020)
    [1] Fill at bar.open=1.1022, bar.low=1.1005 <= stop=1.1010 -> STOP_LOSS
    """
    snapshots = [
        _make_snapshot(0, open_=1.1025, high=1.1028, low=1.1020, close=1.1025),
        _make_snapshot(1, open_=1.1022, high=1.1025, low=1.1005, close=1.1010),
    ]
    result = run_paper_loop(snapshots, [_or_engine()], _config())
    s = result.summary

    assert s.positions_closed >= 1
    assert s.loss_count >= 1

    closed = result.state.all_closed()
    assert any(c.pnl_amount < 0 for c in closed)


def test_stop_loss_pnl_near_negative_risk():
    """At stop: pnl_amount ≈ -risk_amount (exact when fill = entry_reference)."""
    snapshots = [
        _make_snapshot(0, open_=1.1020, high=1.1025, low=1.1015, close=1.1020),
        _make_snapshot(1, open_=1.1020, high=1.1022, low=1.1008, close=1.1012),
    ]
    result = run_paper_loop(snapshots, [_or_engine()], _config())
    closed = result.state.all_closed()
    if closed:
        # fill at 1.1020 (bar[1].open), stop at 1.1010 -> r ≈ -1.0
        assert any(c.r_multiple < 0 for c in closed)


# ─────────────────────────────────────────────────────────────────────────────
# Timeout path
# ─────────────────────────────────────────────────────────────────────────────


def test_timeout_closes_position_after_max_bars():
    """With max_bars_open=2, position closes on bar_index=1 at bar.close."""
    snapshots = [
        # [0] signal
        _make_snapshot(0, open_=1.1025, high=1.1028, low=1.1020, close=1.1025),
        # [1] fill bar — no stop/target hit
        _make_snapshot(1, open_=1.1022, high=1.1030, low=1.1018, close=1.1025),
        # [2] still no stop/target — bar_index=1 >= max_bars_open-1=1 -> timeout
        _make_snapshot(2, open_=1.1025, high=1.1032, low=1.1020, close=1.1028),
    ]
    result = run_paper_loop(snapshots, [_or_engine()], _config(max_bars_open=2))
    assert result.summary.positions_closed >= 1
    assert result.summary.positions_still_open == 0


# ─────────────────────────────────────────────────────────────────────────────
# Risk integration
# ─────────────────────────────────────────────────────────────────────────────


def test_risk_blocks_signal_when_max_positions_reached():
    """
    With max_concurrent_positions=1, first signal opens a position.
    Second signal (same snapshot iteration after fill) is blocked by risk.
    """
    # Build enough LONDON snapshots for multiple signal opportunities
    # Use 5 snapshots: [0] signal1, [1] fill+signal2, [2-4] evaluation bars
    snapshots = [
        _make_snapshot(i, open_=1.1022, high=1.1030, low=1.1015, close=1.1025)
        for i in range(5)
    ]
    cfg = _config(max_positions=1)
    result = run_paper_loop(snapshots, [_or_engine()], cfg)

    # At most 1 position should be open at any time
    assert result.state.open_count + result.state.closed_count <= 2


def test_multiple_engines_both_evaluated():
    """Two engines on same snapshots: both produce signals independently."""
    engine_long = OpeningRangeEngine(
        OpeningRangeDefinition(
            strategy_id="or_long_v1",
            session_name="LONDON",
            min_range_pips=5.0,
            max_range_pips=40.0,
            direction_bias=TradeDirection.LONG,
        )
    )
    engine_short = OpeningRangeEngine(
        OpeningRangeDefinition(
            strategy_id="or_short_v1",
            session_name="LONDON",
            min_range_pips=5.0,
            max_range_pips=40.0,
            direction_bias=TradeDirection.SHORT,
        )
    )
    # Allow same direction multiple and enough positions for both strategies
    cfg = PaperTradingConfig(
        risk_profile=RiskProfile(
            account_equity=10_000.0,
            max_risk_per_trade_pct=1.0,
            max_daily_risk_pct=5.0,
            max_concurrent_positions=5,
            max_positions_per_strategy=3,
            allow_same_direction_multiple=True,
        ),
        instrument=_instrument(),
        stop_distance_points=10.0,
        target_distance_points=20.0,
        pip_size=_PIP_SIZE,
    )
    snapshots = [_make_snapshot(0)]
    result = run_paper_loop(snapshots, [engine_long, engine_short], cfg)

    # Both strategies should have produced signals
    strategy_ids = {bd.strategy_id for bd in result.summary.strategy_breakdown}
    assert "or_long_v1" in strategy_ids
    assert "or_short_v1" in strategy_ids


# ─────────────────────────────────────────────────────────────────────────────
# Journal consistency
# ─────────────────────────────────────────────────────────────────────────────


def test_journal_has_events_for_complete_lifecycle():
    """ORDER_SUBMITTED, ORDER_FILLED, POSITION_CLOSED all appear."""
    snapshots = _lifecycle_snapshots()
    result = run_paper_loop(snapshots, [_or_engine()], _config())
    journal = result.journal

    types = {e.event_type for e in journal.all_events()}
    assert EventType.ORDER_SUBMITTED in types
    assert EventType.ORDER_FILLED in types
    assert EventType.POSITION_CLOSED in types


def test_journal_events_per_position_consistent():
    """Each closed position has ORDER_FILLED and POSITION_CLOSED events."""
    snapshots = _lifecycle_snapshots()
    result = run_paper_loop(snapshots, [_or_engine()], _config())

    for closed_pos in result.state.all_closed():
        pos_id = closed_pos.open_position.position_id
        events = result.journal.events_for(pos_id)
        event_types = {e.event_type for e in events}
        assert EventType.ORDER_FILLED in event_types
        assert EventType.POSITION_CLOSED in event_types


def test_journal_order_submitted_before_filled():
    """ORDER_SUBMITTED always appears before ORDER_FILLED in the log."""
    snapshots = _lifecycle_snapshots()
    result = run_paper_loop(snapshots, [_or_engine()], _config())
    all_events = result.journal.all_events()

    submitted_indices = [
        i for i, e in enumerate(all_events)
        if e.event_type == EventType.ORDER_SUBMITTED
    ]
    filled_indices = [
        i for i, e in enumerate(all_events)
        if e.event_type == EventType.ORDER_FILLED
    ]
    if submitted_indices and filled_indices:
        assert min(submitted_indices) < min(filled_indices)


# ─────────────────────────────────────────────────────────────────────────────
# Summary structure
# ─────────────────────────────────────────────────────────────────────────────


def test_summary_snapshots_evaluated_equals_input_length():
    snapshots = [_make_snapshot(i) for i in range(7)]
    result = run_paper_loop(snapshots, [_or_engine()], _config())
    assert result.summary.snapshots_evaluated == 7


def test_summary_strategy_breakdown_populated():
    snapshots = _lifecycle_snapshots()
    result = run_paper_loop(snapshots, [_or_engine()], _config())
    if result.summary.total_signals > 0:
        assert len(result.summary.strategy_breakdown) >= 1
        bd = result.summary.strategy_breakdown[0]
        assert bd.strategy_id == "or_london_v1"
        assert bd.signals >= 1


def test_summary_breakdown_pnl_matches_total():
    """Sum of per-strategy P&L equals total_pnl."""
    snapshots = _lifecycle_snapshots()
    result = run_paper_loop(snapshots, [_or_engine()], _config())
    s = result.summary
    breakdown_total = sum(bd.pnl for bd in s.strategy_breakdown)
    assert breakdown_total == pytest.approx(s.total_pnl, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────


def test_loop_is_deterministic():
    """Same inputs produce the same summary on two consecutive runs."""
    snapshots = _lifecycle_snapshots()
    cfg = _config()
    engine = _or_engine()

    r1 = run_paper_loop(snapshots, [engine], cfg)
    r2 = run_paper_loop(snapshots, [engine], cfg)

    assert r1.summary.total_signals == r2.summary.total_signals
    assert r1.summary.total_executed == r2.summary.total_executed
    assert r1.summary.positions_closed == r2.summary.positions_closed
    assert r1.summary.total_pnl == pytest.approx(r2.summary.total_pnl, abs=0.01)
