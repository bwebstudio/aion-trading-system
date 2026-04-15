"""
aion/execution/journal.py
──────────────────────────
Event journal for the paper execution engine.

Records every significant event in the order/position lifecycle:
  ORDER_SUBMITTED  — an ExecutionOrder was created and submitted
  ORDER_FILLED     — the order was filled (position opened at fill_price)
  POSITION_CLOSED  — position was closed (stop, target, timeout, or manual)

The journal is append-only.  Events are immutable Pydantic models and are
never modified after logging.

reason_text on each event is non-technical, suitable for a dashboard.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from aion.core.ids import new_snapshot_id
from aion.execution.models import ClosedPosition, ExecutionOrder, FillResult, OpenPosition


def _new_event_id() -> str:
    return f"evt_{new_snapshot_id()[5:]}"


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class EventType(str, Enum):
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    POSITION_CLOSED = "POSITION_CLOSED"


# ─────────────────────────────────────────────────────────────────────────────
# Event model
# ─────────────────────────────────────────────────────────────────────────────


class ExecutionEvent(BaseModel, frozen=True):
    """
    A single event in the execution lifecycle.

    reason_text is written for a non-technical user — it describes what
    happened in plain language, suitable for a dashboard or trade log.
    details carries structured data for machine consumption (logging, analytics).
    """

    event_id: str = Field(default_factory=_new_event_id)
    event_type: EventType
    order_id: str
    position_id: str | None
    """None for ORDER_SUBMITTED (position not yet opened)."""

    strategy_id: str
    symbol: str
    timestamp_utc: datetime
    reason_text: str
    """Clear, non-technical explanation of what happened."""

    details: dict[str, Any] = Field(default_factory=dict)
    """Structured data for logging or debugging."""


# ─────────────────────────────────────────────────────────────────────────────
# Journal
# ─────────────────────────────────────────────────────────────────────────────


class ExecutionJournal:
    """
    Append-only log of execution events.

    Usage:
      journal = ExecutionJournal()
      journal.log_order_submitted(order)
      journal.log_order_filled(fill, position)
      journal.log_position_closed(closed)
    """

    def __init__(self) -> None:
        self._events: list[ExecutionEvent] = []

    # ── Logging helpers ───────────────────────────────────────────────────────

    def log_order_submitted(self, order: ExecutionOrder) -> None:
        """Log that an order was created and submitted for execution."""
        dir_label = "buy" if order.direction.value == "LONG" else "sell"
        target_text = (
            f", target at {order.target_price:.5f}" if order.target_price is not None else ""
        )
        self._events.append(
            ExecutionEvent(
                event_type=EventType.ORDER_SUBMITTED,
                order_id=order.order_id,
                position_id=None,
                strategy_id=order.strategy_id,
                symbol=order.symbol,
                timestamp_utc=order.created_at,
                reason_text=(
                    f"New {dir_label} order submitted for {order.symbol}: "
                    f"{order.position_size} lot(s), entry near {order.entry_price:.5f}, "
                    f"stop at {order.stop_price:.5f}{target_text}. "
                    f"Risk: ${order.risk_amount:.2f}."
                ),
                details={
                    "position_size": order.position_size,
                    "entry_price": order.entry_price,
                    "stop_price": order.stop_price,
                    "target_price": order.target_price,
                    "risk_amount": order.risk_amount,
                    "direction": order.direction.value,
                },
            )
        )

    def log_order_filled(self, fill: FillResult, position: OpenPosition) -> None:
        """Log that an order was filled and a position was opened."""
        order = position.order
        dir_label = "buy" if order.direction.value == "LONG" else "sell"
        self._events.append(
            ExecutionEvent(
                event_type=EventType.ORDER_FILLED,
                order_id=fill.order_id,
                position_id=position.position_id,
                strategy_id=order.strategy_id,
                symbol=order.symbol,
                timestamp_utc=fill.fill_timestamp,
                reason_text=(
                    f"{order.symbol} {dir_label} order filled at {fill.fill_price:.5f}. "
                    f"Position opened: {order.position_size} lot(s), "
                    f"risk ${order.risk_amount:.2f}."
                ),
                details={
                    "fill_price": fill.fill_price,
                    "position_size": order.position_size,
                    "risk_amount": order.risk_amount,
                    "slippage_points": fill.slippage_points,
                },
            )
        )

    def log_position_closed(self, closed: ClosedPosition) -> None:
        """Log that a position was closed."""
        order = closed.open_position.order
        self._events.append(
            ExecutionEvent(
                event_type=EventType.POSITION_CLOSED,
                order_id=order.order_id,
                position_id=closed.open_position.position_id,
                strategy_id=order.strategy_id,
                symbol=order.symbol,
                timestamp_utc=closed.close_timestamp,
                reason_text=closed.reason_text,
                details={
                    "close_price": closed.close_price,
                    "close_reason": closed.close_reason.value,
                    "pnl_amount": closed.pnl_amount,
                    "r_multiple": closed.r_multiple,
                    "bars_held": closed.bars_held,
                },
            )
        )

    def log(self, event: ExecutionEvent) -> None:
        """Append a raw ExecutionEvent (for custom events)."""
        self._events.append(event)

    # ── Queries ───────────────────────────────────────────────────────────────

    def all_events(self) -> list[ExecutionEvent]:
        return list(self._events)

    def events_for(self, position_id: str) -> list[ExecutionEvent]:
        """All events associated with a specific position_id."""
        return [e for e in self._events if e.position_id == position_id]

    def events_by_type(self, event_type: EventType) -> list[ExecutionEvent]:
        """All events of a given type."""
        return [e for e in self._events if e.event_type == event_type]

    @property
    def event_count(self) -> int:
        return len(self._events)
