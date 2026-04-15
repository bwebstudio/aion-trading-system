"""
aion.execution
───────────────
Execution Engine v1 — paper trading mode.

Public API
----------
  PaperExecutionEngine  — create orders, simulate fills, evaluate bars
  ExecutionState        — mutable portfolio state (open / closed positions)
  ExecutionJournal      — append-only event log

  ExecutionOrder        — risk-approved order (entry, stop, target, sizing)
  FillResult            — order fill confirmation (fill_price, timestamp)
  OpenPosition          — active paper position
  ClosedPosition        — completed position with P&L and R-multiple
  CloseReason           — STOP_LOSS | TAKE_PROFIT | TIMEOUT | MANUAL

  ExecutionEvent        — journal entry (frozen)
  EventType             — ORDER_SUBMITTED | ORDER_FILLED | POSITION_CLOSED
"""

from aion.execution.journal import EventType, ExecutionEvent, ExecutionJournal
from aion.execution.models import (
    CloseReason,
    ClosedPosition,
    ExecutionOrder,
    FillResult,
    OpenPosition,
)
from aion.execution.paper import PaperExecutionEngine
from aion.execution.state import ExecutionState

__all__ = [
    "PaperExecutionEngine",
    "ExecutionState",
    "ExecutionJournal",
    "ExecutionOrder",
    "FillResult",
    "OpenPosition",
    "ClosedPosition",
    "CloseReason",
    "ExecutionEvent",
    "EventType",
]
