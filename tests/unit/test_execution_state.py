"""
tests/unit/test_execution_state.py
────────────────────────────────────
Unit tests for aion.execution.state.ExecutionState.

Tests verify:
  - initial state is empty
  - add_position registers a position
  - close_position moves it from open to closed
  - close_position raises KeyError for unknown position_id
  - get_open returns position by id or None
  - all_open / all_closed return correct lists
  - open_count / closed_count properties
  - total_realized_pnl sums closed P&L
  - to_portfolio_state: empty state
  - to_portfolio_state: counts open positions correctly
  - to_portfolio_state: builds by_strategy and by_direction correctly
  - to_portfolio_state: daily_risk_used_pct = open_count * risk_per_trade_pct
  - to_portfolio_state: daily_realized_pnl from closed positions
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aion.core.enums import TradeDirection
from aion.execution.models import (
    CloseReason,
    ClosedPosition,
    ExecutionOrder,
    FillResult,
    OpenPosition,
)
from aion.execution.state import ExecutionState

_UTC = timezone.utc
_TS = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_UTC)
_TS2 = datetime(2024, 1, 15, 10, 31, 0, tzinfo=_UTC)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _order(
    strategy_id: str = "or_london_v1",
    direction: TradeDirection = TradeDirection.LONG,
) -> ExecutionOrder:
    return ExecutionOrder(
        setup_id="setup_test",
        strategy_id=strategy_id,
        symbol="EURUSD",
        direction=direction,
        entry_price=1.1020,
        stop_price=1.1010,
        target_price=1.1040,
        position_size=1.0,
        risk_amount=100.0,
        stop_distance_points=10.0,
        created_at=_TS,
    )


def _open_pos(
    strategy_id: str = "or_london_v1",
    direction: TradeDirection = TradeDirection.LONG,
) -> OpenPosition:
    o = _order(strategy_id=strategy_id, direction=direction)
    fill = FillResult(order_id=o.order_id, fill_price=1.1020, fill_timestamp=_TS)
    return OpenPosition(order=o, fill=fill, opened_at=_TS, bars_open=0)


def _closed_pos(
    open_position: OpenPosition,
    pnl: float = 200.0,
    r: float = 2.0,
) -> ClosedPosition:
    return ClosedPosition(
        open_position=open_position,
        close_price=1.1040,
        close_timestamp=_TS2,
        close_reason=CloseReason.TAKE_PROFIT,
        pnl_amount=pnl,
        r_multiple=r,
        bars_held=3,
        reason_text="Trade closed at take profit.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Initial state
# ─────────────────────────────────────────────────────────────────────────────


def test_initial_state_is_empty():
    state = ExecutionState()
    assert state.open_count == 0
    assert state.closed_count == 0
    assert state.all_open() == []
    assert state.all_closed() == []
    assert state.total_realized_pnl == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# add_position
# ─────────────────────────────────────────────────────────────────────────────


def test_add_position_increments_open_count():
    state = ExecutionState()
    state.add_position(_open_pos())
    assert state.open_count == 1


def test_add_position_returns_via_get_open():
    state = ExecutionState()
    pos = _open_pos()
    state.add_position(pos)
    assert state.get_open(pos.position_id) is pos


def test_add_multiple_positions():
    state = ExecutionState()
    p1 = _open_pos()
    p2 = _open_pos()
    state.add_position(p1)
    state.add_position(p2)
    assert state.open_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# close_position
# ─────────────────────────────────────────────────────────────────────────────


def test_close_position_moves_to_closed():
    state = ExecutionState()
    pos = _open_pos()
    state.add_position(pos)
    closed = _closed_pos(pos)
    state.close_position(pos.position_id, closed)
    assert state.open_count == 0
    assert state.closed_count == 1


def test_close_position_removes_from_open():
    state = ExecutionState()
    pos = _open_pos()
    state.add_position(pos)
    state.close_position(pos.position_id, _closed_pos(pos))
    assert state.get_open(pos.position_id) is None


def test_close_position_raises_for_unknown_id():
    state = ExecutionState()
    pos = _open_pos()
    with pytest.raises(KeyError):
        state.close_position(pos.position_id, _closed_pos(pos))


# ─────────────────────────────────────────────────────────────────────────────
# Queries
# ─────────────────────────────────────────────────────────────────────────────


def test_get_open_returns_none_for_unknown_id():
    state = ExecutionState()
    assert state.get_open("nonexistent") is None


def test_all_open_returns_snapshot():
    state = ExecutionState()
    p1 = _open_pos()
    p2 = _open_pos()
    state.add_position(p1)
    state.add_position(p2)
    ids = {p.position_id for p in state.all_open()}
    assert p1.position_id in ids
    assert p2.position_id in ids


def test_all_closed_returns_in_order():
    state = ExecutionState()
    p1 = _open_pos()
    p2 = _open_pos()
    state.add_position(p1)
    state.add_position(p2)
    c1 = _closed_pos(p1, pnl=100.0)
    c2 = _closed_pos(p2, pnl=-100.0)
    state.close_position(p1.position_id, c1)
    state.close_position(p2.position_id, c2)
    assert state.all_closed()[0].pnl_amount == pytest.approx(100.0)
    assert state.all_closed()[1].pnl_amount == pytest.approx(-100.0)


def test_total_realized_pnl_sums_closed():
    state = ExecutionState()
    p1 = _open_pos()
    p2 = _open_pos()
    state.add_position(p1)
    state.add_position(p2)
    state.close_position(p1.position_id, _closed_pos(p1, pnl=200.0))
    state.close_position(p2.position_id, _closed_pos(p2, pnl=-100.0))
    assert state.total_realized_pnl == pytest.approx(100.0)


# ─────────────────────────────────────────────────────────────────────────────
# to_portfolio_state
# ─────────────────────────────────────────────────────────────────────────────


def test_to_portfolio_state_empty():
    state = ExecutionState()
    ps = state.to_portfolio_state(risk_per_trade_pct=1.0)
    assert ps.open_positions_count == 0
    assert ps.open_positions_by_strategy == {}
    assert ps.open_positions_by_direction == {}
    assert ps.daily_risk_used_pct == pytest.approx(0.0)
    assert ps.daily_realized_pnl == pytest.approx(0.0)


def test_to_portfolio_state_open_count():
    state = ExecutionState()
    state.add_position(_open_pos())
    state.add_position(_open_pos())
    ps = state.to_portfolio_state(risk_per_trade_pct=1.0)
    assert ps.open_positions_count == 2


def test_to_portfolio_state_by_strategy():
    state = ExecutionState()
    state.add_position(_open_pos(strategy_id="or_london_v1"))
    state.add_position(_open_pos(strategy_id="or_london_v1"))
    state.add_position(_open_pos(strategy_id="vwap_fade_v1"))
    ps = state.to_portfolio_state()
    assert ps.open_positions_by_strategy["or_london_v1"] == 2
    assert ps.open_positions_by_strategy["vwap_fade_v1"] == 1


def test_to_portfolio_state_by_direction():
    state = ExecutionState()
    state.add_position(_open_pos(direction=TradeDirection.LONG))
    state.add_position(_open_pos(direction=TradeDirection.LONG))
    state.add_position(_open_pos(direction=TradeDirection.SHORT))
    ps = state.to_portfolio_state()
    assert ps.open_positions_by_direction["LONG"] == 2
    assert ps.open_positions_by_direction["SHORT"] == 1


def test_to_portfolio_state_daily_risk_used():
    """daily_risk_used_pct = open_count * risk_per_trade_pct."""
    state = ExecutionState()
    state.add_position(_open_pos())
    state.add_position(_open_pos())
    ps = state.to_portfolio_state(risk_per_trade_pct=1.0)
    assert ps.daily_risk_used_pct == pytest.approx(2.0)


def test_to_portfolio_state_daily_realized_pnl():
    state = ExecutionState()
    pos = _open_pos()
    state.add_position(pos)
    state.close_position(pos.position_id, _closed_pos(pos, pnl=150.0))
    ps = state.to_portfolio_state()
    assert ps.daily_realized_pnl == pytest.approx(150.0)
