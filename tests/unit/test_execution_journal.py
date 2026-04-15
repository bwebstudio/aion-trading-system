"""
tests/unit/test_execution_journal.py
──────────────────────────────────────
Unit tests for aion.execution.journal.

Tests verify:
  ExecutionEvent:
    - is frozen (immutable)
    - event_id auto-generated
    - reason_text is present

  ExecutionJournal:
    - log_order_submitted appends ORDER_SUBMITTED event
    - log_order_filled appends ORDER_FILLED event
    - log_position_closed appends POSITION_CLOSED event
    - events_for returns only events matching position_id
    - events_by_type filters correctly
    - all_events returns all logged events
    - log() appends raw event
    - event_count property
    - POSITION_CLOSED event carries pnl and r_multiple in details
    - reason_text is non-empty for all log helpers
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aion.core.enums import TradeDirection
from aion.execution.journal import EventType, ExecutionEvent, ExecutionJournal
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
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _order(direction: TradeDirection = TradeDirection.LONG) -> ExecutionOrder:
    return ExecutionOrder(
        setup_id="setup_test",
        strategy_id="or_london_v1",
        symbol="EURUSD",
        direction=direction,
        entry_price=1.1020,
        stop_price=1.1010,
        target_price=1.1040,
        position_size=1.0,
        risk_amount=100.0,
        stop_distance_points=10.0,
        target_distance_points=20.0,
        created_at=_TS,
    )


def _open_pos(order: ExecutionOrder | None = None) -> OpenPosition:
    o = order or _order()
    fill = FillResult(order_id=o.order_id, fill_price=1.1020, fill_timestamp=_TS2)
    return OpenPosition(order=o, fill=fill, opened_at=_TS2, bars_open=0)


def _closed_pos(open_position: OpenPosition) -> ClosedPosition:
    return ClosedPosition(
        open_position=open_position,
        close_price=1.1040,
        close_timestamp=_TS2,
        close_reason=CloseReason.TAKE_PROFIT,
        pnl_amount=200.0,
        r_multiple=2.0,
        bars_held=3,
        reason_text="Trade closed at take profit (1.1040). Profit: $200.00.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# ExecutionEvent model
# ─────────────────────────────────────────────────────────────────────────────


def test_execution_event_is_frozen():
    event = ExecutionEvent(
        event_type=EventType.ORDER_SUBMITTED,
        order_id="ord_test",
        position_id=None,
        strategy_id="or_london_v1",
        symbol="EURUSD",
        timestamp_utc=_TS,
        reason_text="Test event.",
    )
    with pytest.raises(Exception):
        event.reason_text = "Changed"  # type: ignore[misc]


def test_execution_event_id_auto_generated():
    event = ExecutionEvent(
        event_type=EventType.ORDER_SUBMITTED,
        order_id="ord_test",
        position_id=None,
        strategy_id="or_london_v1",
        symbol="EURUSD",
        timestamp_utc=_TS,
        reason_text="Test event.",
    )
    assert event.event_id.startswith("evt_")


def test_two_events_have_different_ids():
    kwargs = dict(
        event_type=EventType.ORDER_SUBMITTED,
        order_id="ord_test",
        position_id=None,
        strategy_id="or_london_v1",
        symbol="EURUSD",
        timestamp_utc=_TS,
        reason_text="Test.",
    )
    e1 = ExecutionEvent(**kwargs)
    e2 = ExecutionEvent(**kwargs)
    assert e1.event_id != e2.event_id


# ─────────────────────────────────────────────────────────────────────────────
# log_order_submitted
# ─────────────────────────────────────────────────────────────────────────────


def test_log_order_submitted_appends_event():
    journal = ExecutionJournal()
    journal.log_order_submitted(_order())
    assert journal.event_count == 1


def test_log_order_submitted_event_type():
    journal = ExecutionJournal()
    journal.log_order_submitted(_order())
    event = journal.all_events()[0]
    assert event.event_type == EventType.ORDER_SUBMITTED


def test_log_order_submitted_position_id_is_none():
    journal = ExecutionJournal()
    journal.log_order_submitted(_order())
    assert journal.all_events()[0].position_id is None


def test_log_order_submitted_reason_text_non_empty():
    journal = ExecutionJournal()
    journal.log_order_submitted(_order())
    assert len(journal.all_events()[0].reason_text) > 10


def test_log_order_submitted_details_contain_risk_amount():
    journal = ExecutionJournal()
    journal.log_order_submitted(_order())
    details = journal.all_events()[0].details
    assert details["risk_amount"] == pytest.approx(100.0)


# ─────────────────────────────────────────────────────────────────────────────
# log_order_filled
# ─────────────────────────────────────────────────────────────────────────────


def test_log_order_filled_appends_event():
    journal = ExecutionJournal()
    pos = _open_pos()
    journal.log_order_filled(pos.fill, pos)
    assert journal.event_count == 1


def test_log_order_filled_event_type():
    journal = ExecutionJournal()
    pos = _open_pos()
    journal.log_order_filled(pos.fill, pos)
    assert journal.all_events()[0].event_type == EventType.ORDER_FILLED


def test_log_order_filled_carries_position_id():
    journal = ExecutionJournal()
    pos = _open_pos()
    journal.log_order_filled(pos.fill, pos)
    assert journal.all_events()[0].position_id == pos.position_id


def test_log_order_filled_reason_text_non_empty():
    journal = ExecutionJournal()
    pos = _open_pos()
    journal.log_order_filled(pos.fill, pos)
    assert len(journal.all_events()[0].reason_text) > 10


# ─────────────────────────────────────────────────────────────────────────────
# log_position_closed
# ─────────────────────────────────────────────────────────────────────────────


def test_log_position_closed_appends_event():
    journal = ExecutionJournal()
    pos = _open_pos()
    journal.log_position_closed(_closed_pos(pos))
    assert journal.event_count == 1


def test_log_position_closed_event_type():
    journal = ExecutionJournal()
    pos = _open_pos()
    journal.log_position_closed(_closed_pos(pos))
    assert journal.all_events()[0].event_type == EventType.POSITION_CLOSED


def test_log_position_closed_details_pnl():
    journal = ExecutionJournal()
    pos = _open_pos()
    journal.log_position_closed(_closed_pos(pos))
    details = journal.all_events()[0].details
    assert details["pnl_amount"] == pytest.approx(200.0)
    assert details["r_multiple"] == pytest.approx(2.0)


def test_log_position_closed_reason_text_is_closed_reason_text():
    journal = ExecutionJournal()
    pos = _open_pos()
    closed = _closed_pos(pos)
    journal.log_position_closed(closed)
    assert journal.all_events()[0].reason_text == closed.reason_text


# ─────────────────────────────────────────────────────────────────────────────
# Queries
# ─────────────────────────────────────────────────────────────────────────────


def test_events_for_returns_only_matching_position():
    journal = ExecutionJournal()
    pos1 = _open_pos()
    pos2 = _open_pos()
    journal.log_order_filled(pos1.fill, pos1)
    journal.log_order_filled(pos2.fill, pos2)
    journal.log_position_closed(_closed_pos(pos1))

    events_1 = journal.events_for(pos1.position_id)
    assert len(events_1) == 2
    assert all(e.position_id == pos1.position_id for e in events_1)


def test_events_by_type_filters_correctly():
    journal = ExecutionJournal()
    order = _order()
    pos = _open_pos(order)
    journal.log_order_submitted(order)
    journal.log_order_filled(pos.fill, pos)
    journal.log_position_closed(_closed_pos(pos))

    submitted = journal.events_by_type(EventType.ORDER_SUBMITTED)
    filled = journal.events_by_type(EventType.ORDER_FILLED)
    closed_events = journal.events_by_type(EventType.POSITION_CLOSED)

    assert len(submitted) == 1
    assert len(filled) == 1
    assert len(closed_events) == 1


def test_all_events_returns_all():
    journal = ExecutionJournal()
    order = _order()
    pos = _open_pos(order)
    journal.log_order_submitted(order)
    journal.log_order_filled(pos.fill, pos)
    journal.log_position_closed(_closed_pos(pos))
    assert len(journal.all_events()) == 3


def test_log_raw_event():
    journal = ExecutionJournal()
    event = ExecutionEvent(
        event_type=EventType.ORDER_SUBMITTED,
        order_id="ord_custom",
        position_id=None,
        strategy_id="custom_strat",
        symbol="EURUSD",
        timestamp_utc=_TS,
        reason_text="Custom event for testing.",
    )
    journal.log(event)
    assert journal.event_count == 1
    assert journal.all_events()[0].order_id == "ord_custom"


def test_event_count_property():
    journal = ExecutionJournal()
    assert journal.event_count == 0
    journal.log_order_submitted(_order())
    assert journal.event_count == 1
