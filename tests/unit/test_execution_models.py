"""
tests/unit/test_execution_models.py
─────────────────────────────────────
Unit tests for aion.execution.models.

Tests verify:
  CloseReason:
    - expected enum values

  ExecutionOrder:
    - fields are stored correctly
    - is frozen (immutable)
    - order_id auto-generated

  FillResult:
    - fields correct
    - slippage_points defaults to 0
    - is frozen

  OpenPosition:
    - fields correct
    - is frozen
    - bars_open defaults to 0

  ClosedPosition:
    - fields correct (pnl_amount, r_multiple, bars_held, reason_text)
    - is frozen
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

_UTC = timezone.utc
_TS = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_UTC)
_TS2 = datetime(2024, 1, 15, 10, 31, 0, tzinfo=_UTC)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _order(
    direction: TradeDirection = TradeDirection.LONG,
    entry: float = 1.1020,
    stop: float = 1.1010,
    target: float | None = 1.1040,
) -> ExecutionOrder:
    return ExecutionOrder(
        setup_id="setup_test",
        strategy_id="or_london_v1",
        symbol="EURUSD",
        direction=direction,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        position_size=1.0,
        risk_amount=100.0,
        stop_distance_points=10.0,
        target_distance_points=20.0,
        created_at=_TS,
    )


def _fill(order: ExecutionOrder) -> FillResult:
    return FillResult(
        order_id=order.order_id,
        fill_price=1.1020,
        fill_timestamp=_TS,
    )


def _open_position(order: ExecutionOrder | None = None) -> OpenPosition:
    o = order or _order()
    return OpenPosition(
        order=o,
        fill=_fill(o),
        opened_at=_TS,
        bars_open=0,
    )


def _closed_position(open_pos: OpenPosition | None = None) -> ClosedPosition:
    pos = open_pos or _open_position()
    return ClosedPosition(
        open_position=pos,
        close_price=1.1040,
        close_timestamp=_TS2,
        close_reason=CloseReason.TAKE_PROFIT,
        pnl_amount=200.0,
        r_multiple=2.0,
        bars_held=3,
        reason_text="Trade closed at take profit (1.1040). Profit: $200.00.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CloseReason
# ─────────────────────────────────────────────────────────────────────────────


def test_close_reason_values():
    assert CloseReason.STOP_LOSS == "STOP_LOSS"
    assert CloseReason.TAKE_PROFIT == "TAKE_PROFIT"
    assert CloseReason.TIMEOUT == "TIMEOUT"
    assert CloseReason.MANUAL == "MANUAL"


# ─────────────────────────────────────────────────────────────────────────────
# ExecutionOrder
# ─────────────────────────────────────────────────────────────────────────────


def test_execution_order_fields():
    o = _order()
    assert o.setup_id == "setup_test"
    assert o.strategy_id == "or_london_v1"
    assert o.symbol == "EURUSD"
    assert o.direction == TradeDirection.LONG
    assert o.entry_price == pytest.approx(1.1020)
    assert o.stop_price == pytest.approx(1.1010)
    assert o.target_price == pytest.approx(1.1040)
    assert o.position_size == pytest.approx(1.0)
    assert o.risk_amount == pytest.approx(100.0)
    assert o.stop_distance_points == pytest.approx(10.0)
    assert o.target_distance_points == pytest.approx(20.0)


def test_execution_order_is_frozen():
    o = _order()
    with pytest.raises(Exception):
        o.position_size = 2.0  # type: ignore[misc]


def test_execution_order_id_auto_generated():
    o = _order()
    assert o.order_id.startswith("ord_")
    assert len(o.order_id) > 4


def test_execution_order_two_instances_have_different_ids():
    o1 = _order()
    o2 = _order()
    assert o1.order_id != o2.order_id


def test_execution_order_target_price_optional():
    o = ExecutionOrder(
        setup_id="setup_test",
        strategy_id="or_london_v1",
        symbol="EURUSD",
        direction=TradeDirection.LONG,
        entry_price=1.1020,
        stop_price=1.1010,
        target_price=None,
        position_size=1.0,
        risk_amount=100.0,
        stop_distance_points=10.0,
        target_distance_points=None,
        created_at=_TS,
    )
    assert o.target_price is None
    assert o.target_distance_points is None


# ─────────────────────────────────────────────────────────────────────────────
# FillResult
# ─────────────────────────────────────────────────────────────────────────────


def test_fill_result_fields():
    o = _order()
    f = _fill(o)
    assert f.order_id == o.order_id
    assert f.fill_price == pytest.approx(1.1020)
    assert f.fill_timestamp == _TS


def test_fill_result_default_slippage_is_zero():
    f = _fill(_order())
    assert f.slippage_points == pytest.approx(0.0)


def test_fill_result_is_frozen():
    f = _fill(_order())
    with pytest.raises(Exception):
        f.fill_price = 1.1025  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# OpenPosition
# ─────────────────────────────────────────────────────────────────────────────


def test_open_position_fields():
    pos = _open_position()
    assert pos.opened_at == _TS
    assert pos.bars_open == 0


def test_open_position_default_bars_open():
    pos = _open_position()
    assert pos.bars_open == 0


def test_open_position_is_frozen():
    pos = _open_position()
    with pytest.raises(Exception):
        pos.bars_open = 5  # type: ignore[misc]


def test_open_position_id_auto_generated():
    pos = _open_position()
    assert pos.position_id.startswith("pos_")


def test_open_position_two_instances_have_different_ids():
    p1 = _open_position()
    p2 = _open_position()
    assert p1.position_id != p2.position_id


# ─────────────────────────────────────────────────────────────────────────────
# ClosedPosition
# ─────────────────────────────────────────────────────────────────────────────


def test_closed_position_fields():
    c = _closed_position()
    assert c.close_price == pytest.approx(1.1040)
    assert c.close_reason == CloseReason.TAKE_PROFIT
    assert c.pnl_amount == pytest.approx(200.0)
    assert c.r_multiple == pytest.approx(2.0)
    assert c.bars_held == 3


def test_closed_position_is_frozen():
    c = _closed_position()
    with pytest.raises(Exception):
        c.pnl_amount = 0.0  # type: ignore[misc]


def test_closed_position_reason_text_present():
    c = _closed_position()
    assert len(c.reason_text) > 0


def test_closed_position_negative_pnl_for_stop():
    pos = _open_position()
    c = ClosedPosition(
        open_position=pos,
        close_price=1.1010,
        close_timestamp=_TS2,
        close_reason=CloseReason.STOP_LOSS,
        pnl_amount=-100.0,
        r_multiple=-1.0,
        bars_held=2,
        reason_text="Trade closed at stop loss.",
    )
    assert c.pnl_amount == pytest.approx(-100.0)
    assert c.r_multiple == pytest.approx(-1.0)
