"""
tests/unit/test_paper_execution.py
────────────────────────────────────
Unit tests for aion.execution.paper.PaperExecutionEngine.

Tests verify:
  create_order:
    - fields copied correctly from RiskDecision + CandidateSetup
    - stop_price and target_price stored as provided
    - raises on rejected RiskDecision

  fill_order:
    - fill price equals bar.open
    - slippage is 0
    - OpenPosition is created with correct fields

  evaluate_bar:
    - LONG: stop triggered when bar.low <= stop_price
    - LONG: target triggered when bar.high >= target_price
    - SHORT: stop triggered when bar.high >= stop_price
    - SHORT: target triggered when bar.low <= target_price
    - stop wins over target when both hit on same bar
    - timeout closes at bar.close when bar_index >= max_bars_open - 1
    - returns None when position should remain open
    - P&L at stop = -risk_amount (R = -1.0)
    - P&L at 2R target = +2 * risk_amount (R = +2.0)
    - P&L at timeout (price unchanged) = 0
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aion.core.enums import AssetClass, DataSource, SessionName, Timeframe, TradeDirection
from aion.core.models import InstrumentSpec, MarketBar
from aion.execution.models import CloseReason, ExecutionOrder, FillResult, OpenPosition
from aion.execution.paper import PaperExecutionEngine
from aion.risk.models import RiskDecision
from aion.strategies.models import CandidateSetup

_UTC = timezone.utc
_TS = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_UTC)
_TS2 = datetime(2024, 1, 15, 10, 31, 0, tzinfo=_UTC)
_TS3 = datetime(2024, 1, 15, 10, 32, 0, tzinfo=_UTC)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _approved_decision(
    position_size: float = 1.0,
    risk_amount: float = 100.0,
    stop_distance_points: float = 10.0,
    target_distance_points: float | None = 20.0,
) -> RiskDecision:
    return RiskDecision(
        approved=True,
        reason_code="APPROVED",
        reason_text="All checks passed.",
        candidate_setup_id="setup_test",
        strategy_id="or_london_v1",
        position_size=position_size,
        risk_amount=risk_amount,
        stop_distance_points=stop_distance_points,
        target_distance_points=target_distance_points,
    )


def _rejected_decision() -> RiskDecision:
    return RiskDecision(
        approved=False,
        reason_code="MAX_POSITIONS_REACHED",
        reason_text="Too many positions.",
        candidate_setup_id="setup_test",
        strategy_id="or_london_v1",
        stop_distance_points=10.0,
    )


def _candidate(
    direction: TradeDirection = TradeDirection.LONG,
    entry: float = 1.1020,
) -> CandidateSetup:
    return CandidateSetup(
        strategy_id="or_london_v1",
        strategy_version="1.0.0",
        symbol="EURUSD",
        timestamp_utc=_TS,
        direction=direction,
        entry_reference=entry,
        range_high=1.1020,
        range_low=1.1000,
        range_size_pips=20.0,
        session_name=SessionName.LONDON.value,
        quality_score=1.0,
        atr_14=0.00015,
    )


def _bar(
    open_: float,
    high: float,
    low: float,
    close: float,
    ts: datetime | None = None,
) -> MarketBar:
    return MarketBar(
        symbol="EURUSD",
        timestamp_utc=ts or _TS2,
        timestamp_market=ts or _TS2,
        timeframe=Timeframe.M1,
        open=open_,
        high=high,
        low=low,
        close=close,
        tick_volume=100.0,
        real_volume=0.0,
        spread=1.0,
        source=DataSource.CSV,
    )


def _make_long_position(
    fill_price: float = 1.1020,
    stop: float = 1.1010,
    target: float | None = 1.1040,
    risk_amount: float = 100.0,
) -> OpenPosition:
    """Build a LONG OpenPosition directly without going through the engine."""
    order = ExecutionOrder(
        setup_id="setup_test",
        strategy_id="or_london_v1",
        symbol="EURUSD",
        direction=TradeDirection.LONG,
        entry_price=fill_price,
        stop_price=stop,
        target_price=target,
        position_size=1.0,
        risk_amount=risk_amount,
        stop_distance_points=10.0,
        target_distance_points=20.0,
        created_at=_TS,
    )
    fill = FillResult(order_id=order.order_id, fill_price=fill_price, fill_timestamp=_TS2)
    return OpenPosition(order=order, fill=fill, opened_at=_TS2, bars_open=0)


def _make_short_position(
    fill_price: float = 1.1020,
    stop: float = 1.1030,
    target: float | None = 1.1000,
    risk_amount: float = 100.0,
) -> OpenPosition:
    order = ExecutionOrder(
        setup_id="setup_test",
        strategy_id="or_london_v1",
        symbol="EURUSD",
        direction=TradeDirection.SHORT,
        entry_price=fill_price,
        stop_price=stop,
        target_price=target,
        position_size=1.0,
        risk_amount=risk_amount,
        stop_distance_points=10.0,
        target_distance_points=20.0,
        created_at=_TS,
    )
    fill = FillResult(order_id=order.order_id, fill_price=fill_price, fill_timestamp=_TS2)
    return OpenPosition(order=order, fill=fill, opened_at=_TS2, bars_open=0)


engine = PaperExecutionEngine()


# ─────────────────────────────────────────────────────────────────────────────
# create_order
# ─────────────────────────────────────────────────────────────────────────────


def test_create_order_fields_from_decision_and_candidate():
    decision = _approved_decision()
    candidate = _candidate()
    order = engine.create_order(decision, candidate, stop_price=1.1010, target_price=1.1040)

    assert order.setup_id == candidate.setup_id
    assert order.strategy_id == "or_london_v1"
    assert order.symbol == "EURUSD"
    assert order.direction == TradeDirection.LONG
    assert order.entry_price == pytest.approx(1.1020)
    assert order.stop_price == pytest.approx(1.1010)
    assert order.target_price == pytest.approx(1.1040)
    assert order.position_size == pytest.approx(1.0)
    assert order.risk_amount == pytest.approx(100.0)
    assert order.stop_distance_points == pytest.approx(10.0)
    assert order.target_distance_points == pytest.approx(20.0)


def test_create_order_no_target():
    decision = _approved_decision(target_distance_points=None)
    order = engine.create_order(decision, _candidate(), stop_price=1.1010)
    assert order.target_price is None


def test_create_order_raises_on_rejected_decision():
    with pytest.raises(ValueError, match="rejected"):
        engine.create_order(_rejected_decision(), _candidate(), stop_price=1.1010)


def test_create_order_created_at_matches_candidate_timestamp():
    candidate = _candidate()
    order = engine.create_order(_approved_decision(), candidate, stop_price=1.1010)
    assert order.created_at == candidate.timestamp_utc


# ─────────────────────────────────────────────────────────────────────────────
# fill_order
# ─────────────────────────────────────────────────────────────────────────────


def test_fill_order_fill_price_equals_bar_open():
    decision = _approved_decision()
    order = engine.create_order(decision, _candidate(), stop_price=1.1010)
    fill_bar = _bar(open_=1.1022, high=1.1030, low=1.1015, close=1.1025)
    fill, position = engine.fill_order(order, fill_bar)
    assert fill.fill_price == pytest.approx(1.1022)


def test_fill_order_slippage_is_zero():
    order = engine.create_order(_approved_decision(), _candidate(), stop_price=1.1010)
    fill, _ = engine.fill_order(order, _bar(1.1020, 1.1030, 1.1015, 1.1025))
    assert fill.slippage_points == pytest.approx(0.0)


def test_fill_order_creates_open_position():
    order = engine.create_order(_approved_decision(), _candidate(), stop_price=1.1010)
    fill_bar = _bar(1.1020, 1.1030, 1.1015, 1.1025)
    fill, position = engine.fill_order(order, fill_bar)
    assert position.order.order_id == order.order_id
    assert position.fill.fill_price == pytest.approx(fill_bar.open)
    assert position.bars_open == 0


def test_fill_order_position_opened_at_fill_bar_timestamp():
    order = engine.create_order(_approved_decision(), _candidate(), stop_price=1.1010)
    fill_bar = _bar(1.1020, 1.1030, 1.1015, 1.1025, ts=_TS2)
    _, position = engine.fill_order(order, fill_bar)
    assert position.opened_at == _TS2


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_bar — LONG
# ─────────────────────────────────────────────────────────────────────────────


def test_long_stop_triggered_when_bar_low_touches_stop():
    """bar.low <= stop_price -> STOP_LOSS."""
    pos = _make_long_position(fill_price=1.1020, stop=1.1010)
    bar = _bar(open_=1.1018, high=1.1022, low=1.1010, close=1.1015)
    closed = engine.evaluate_bar(pos, bar, bar_index=0)
    assert closed is not None
    assert closed.close_reason == CloseReason.STOP_LOSS
    assert closed.close_price == pytest.approx(1.1010)


def test_long_stop_triggered_when_bar_low_below_stop():
    pos = _make_long_position(fill_price=1.1020, stop=1.1010)
    bar = _bar(open_=1.1018, high=1.1022, low=1.1005, close=1.1015)
    closed = engine.evaluate_bar(pos, bar, bar_index=0)
    assert closed is not None
    assert closed.close_reason == CloseReason.STOP_LOSS


def test_long_target_triggered_when_bar_high_touches_target():
    """bar.high >= target_price -> TAKE_PROFIT."""
    pos = _make_long_position(fill_price=1.1020, stop=1.1010, target=1.1040)
    bar = _bar(open_=1.1025, high=1.1040, low=1.1020, close=1.1038)
    closed = engine.evaluate_bar(pos, bar, bar_index=0)
    assert closed is not None
    assert closed.close_reason == CloseReason.TAKE_PROFIT
    assert closed.close_price == pytest.approx(1.1040)


def test_long_target_triggered_when_bar_high_above_target():
    pos = _make_long_position(fill_price=1.1020, stop=1.1010, target=1.1040)
    bar = _bar(open_=1.1025, high=1.1045, low=1.1020, close=1.1042)
    closed = engine.evaluate_bar(pos, bar, bar_index=0)
    assert closed is not None
    assert closed.close_reason == CloseReason.TAKE_PROFIT


def test_long_no_close_when_price_between_stop_and_target():
    pos = _make_long_position(fill_price=1.1020, stop=1.1010, target=1.1040)
    bar = _bar(open_=1.1022, high=1.1035, low=1.1015, close=1.1030)
    result = engine.evaluate_bar(pos, bar, bar_index=0)
    assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_bar — SHORT
# ─────────────────────────────────────────────────────────────────────────────


def test_short_stop_triggered_when_bar_high_touches_stop():
    """SHORT: bar.high >= stop_price -> STOP_LOSS."""
    pos = _make_short_position(fill_price=1.1020, stop=1.1030, target=1.1000)
    bar = _bar(open_=1.1022, high=1.1030, low=1.1015, close=1.1018)
    closed = engine.evaluate_bar(pos, bar, bar_index=0)
    assert closed is not None
    assert closed.close_reason == CloseReason.STOP_LOSS
    assert closed.close_price == pytest.approx(1.1030)


def test_short_target_triggered_when_bar_low_touches_target():
    """SHORT: bar.low <= target_price -> TAKE_PROFIT."""
    pos = _make_short_position(fill_price=1.1020, stop=1.1030, target=1.1000)
    bar = _bar(open_=1.1015, high=1.1022, low=1.1000, close=1.1005)
    closed = engine.evaluate_bar(pos, bar, bar_index=0)
    assert closed is not None
    assert closed.close_reason == CloseReason.TAKE_PROFIT
    assert closed.close_price == pytest.approx(1.1000)


def test_short_no_close_when_price_between_stop_and_target():
    pos = _make_short_position(fill_price=1.1020, stop=1.1030, target=1.1000)
    bar = _bar(open_=1.1018, high=1.1025, low=1.1008, close=1.1012)
    result = engine.evaluate_bar(pos, bar, bar_index=0)
    assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Stop wins over target on same bar
# ─────────────────────────────────────────────────────────────────────────────


def test_stop_wins_over_target_long_when_both_hit_same_bar():
    """
    LONG: bar hits both stop (low <= stop) and target (high >= target).
    Stop is checked first -> STOP_LOSS.
    """
    pos = _make_long_position(fill_price=1.1020, stop=1.1010, target=1.1040)
    # Volatile bar: goes down to stop AND up to target
    bar = _bar(open_=1.1020, high=1.1045, low=1.1005, close=1.1030)
    closed = engine.evaluate_bar(pos, bar, bar_index=0)
    assert closed is not None
    assert closed.close_reason == CloseReason.STOP_LOSS


def test_stop_wins_over_target_short_when_both_hit_same_bar():
    pos = _make_short_position(fill_price=1.1020, stop=1.1030, target=1.1000)
    bar = _bar(open_=1.1020, high=1.1035, low=0.9990, close=1.1010)
    closed = engine.evaluate_bar(pos, bar, bar_index=0)
    assert closed is not None
    assert closed.close_reason == CloseReason.STOP_LOSS


# ─────────────────────────────────────────────────────────────────────────────
# Timeout
# ─────────────────────────────────────────────────────────────────────────────


def test_timeout_closes_at_bar_close():
    pos = _make_long_position(fill_price=1.1020, stop=1.1010, target=1.1040)
    # bar_index=4 with max_bars_open=5 -> timeout
    bar = _bar(open_=1.1025, high=1.1035, low=1.1015, close=1.1028)
    closed = engine.evaluate_bar(pos, bar, bar_index=4, max_bars_open=5)
    assert closed is not None
    assert closed.close_reason == CloseReason.TIMEOUT
    assert closed.close_price == pytest.approx(1.1028)


def test_no_timeout_before_limit():
    pos = _make_long_position(fill_price=1.1020, stop=1.1010, target=1.1040)
    bar = _bar(open_=1.1025, high=1.1035, low=1.1015, close=1.1028)
    result = engine.evaluate_bar(pos, bar, bar_index=3, max_bars_open=5)
    assert result is None


def test_no_timeout_when_max_bars_not_set():
    pos = _make_long_position(fill_price=1.1020, stop=1.1010, target=1.1040)
    bar = _bar(open_=1.1025, high=1.1035, low=1.1015, close=1.1028)
    # No max_bars_open -> never times out
    result = engine.evaluate_bar(pos, bar, bar_index=100)
    assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# P&L correctness
# ─────────────────────────────────────────────────────────────────────────────


def test_pnl_at_stop_equals_negative_risk_amount():
    """At stop: pnl_amount = -risk_amount, r_multiple = -1.0."""
    pos = _make_long_position(fill_price=1.1020, stop=1.1010, risk_amount=100.0)
    # Stop is exactly at stop_price
    bar = _bar(open_=1.1018, high=1.1022, low=1.1010, close=1.1015)
    closed = engine.evaluate_bar(pos, bar, bar_index=0)
    assert closed is not None
    assert closed.pnl_amount == pytest.approx(-100.0)
    assert closed.r_multiple == pytest.approx(-1.0)


def test_pnl_at_2r_target():
    """2:1 target (20 pips vs 10 pip stop): pnl = +2 * risk_amount, r = +2.0."""
    pos = _make_long_position(
        fill_price=1.1020, stop=1.1010, target=1.1040, risk_amount=100.0
    )
    bar = _bar(open_=1.1025, high=1.1040, low=1.1022, close=1.1038)
    closed = engine.evaluate_bar(pos, bar, bar_index=0)
    assert closed is not None
    assert closed.pnl_amount == pytest.approx(200.0)
    assert closed.r_multiple == pytest.approx(2.0)


def test_pnl_at_timeout_breakeven():
    """Timeout at fill price: pnl = 0, r = 0."""
    pos = _make_long_position(fill_price=1.1020, stop=1.1010, target=1.1040)
    # Close = fill_price -> breakeven
    bar = _bar(open_=1.1022, high=1.1030, low=1.1015, close=1.1020)
    closed = engine.evaluate_bar(pos, bar, bar_index=4, max_bars_open=5)
    assert closed is not None
    assert closed.pnl_amount == pytest.approx(0.0, abs=0.01)
    assert closed.r_multiple == pytest.approx(0.0, abs=0.001)


def test_bars_held_is_bar_index_plus_one():
    pos = _make_long_position(fill_price=1.1020, stop=1.1010)
    bar = _bar(open_=1.1018, high=1.1022, low=1.1009, close=1.1015)
    closed = engine.evaluate_bar(pos, bar, bar_index=2)
    assert closed is not None
    assert closed.bars_held == 3
