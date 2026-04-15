"""
aion/execution/models.py
─────────────────────────
Domain models for Execution Engine v1.

Four core models:
  ExecutionOrder  — risk-approved order, ready to be sent to the broker.
  FillResult      — confirmation of an order fill (simulated or real).
  OpenPosition    — live position derived from a filled order.
  ClosedPosition  — completed position with P&L and close context.

All models are immutable (frozen=True).

Design notes:
  - stop_price and target_price are in the instrument's native price units.
  - P&L is computed using the R-multiple formula:
      r_multiple = direction_sign * (close - fill) / |stop - fill|
      pnl_amount = r_multiple * risk_amount
    This avoids needing pip_multiplier or InstrumentSpec at close time.
    At a full stop loss: r_multiple = -1.0, pnl_amount = -risk_amount exactly.
  - reason_text on ClosedPosition is non-technical, suitable for a dashboard.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from aion.core.enums import TradeDirection
from aion.core.ids import new_snapshot_id


# ─────────────────────────────────────────────────────────────────────────────
# ID factories
# ─────────────────────────────────────────────────────────────────────────────


def _new_position_id() -> str:
    return f"pos_{new_snapshot_id()[5:]}"


def _new_order_id() -> str:
    return f"ord_{new_snapshot_id()[5:]}"


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class CloseReason(str, Enum):
    """Why a position was closed."""

    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TIMEOUT = "TIMEOUT"
    MANUAL = "MANUAL"


# ─────────────────────────────────────────────────────────────────────────────
# Order
# ─────────────────────────────────────────────────────────────────────────────


class ExecutionOrder(BaseModel, frozen=True):
    """
    A risk-approved order, ready to be executed.

    Created by PaperExecutionEngine.create_order() from a RiskDecision
    and CandidateSetup.  Contains all information needed to manage the
    full position lifecycle.

    Price levels (entry_price, stop_price, target_price) are in the
    instrument's native price units.
    """

    order_id: str = Field(default_factory=_new_order_id)

    setup_id: str
    """CandidateSetup.setup_id — for tracing back to the strategy signal."""

    strategy_id: str
    symbol: str
    direction: TradeDirection

    entry_price: float
    """Reference entry price (OR high/low from CandidateSetup.entry_reference)."""

    stop_price: float
    """Stop loss price level in instrument price units."""

    target_price: float | None = None
    """Take profit price level.  None if no target was specified."""

    position_size: float
    """Position size in lots."""

    risk_amount: float
    """Maximum monetary amount at risk (account currency)."""

    stop_distance_points: float
    """Stop distance in strategy-native units (pips for forex, points for indices)."""

    target_distance_points: float | None = None
    """Target distance in the same units.  None if no target."""

    created_at: datetime
    """When this order was created (UTC)."""


# ─────────────────────────────────────────────────────────────────────────────
# Fill
# ─────────────────────────────────────────────────────────────────────────────


class FillResult(BaseModel, frozen=True):
    """
    Confirmation of an order fill.

    In paper trading, fill_price is the open of the bar following the signal
    bar (next-bar-open model), and slippage_points is always 0.
    """

    order_id: str
    fill_price: float
    """Actual fill price (bar.open of the fill bar)."""

    fill_timestamp: datetime
    slippage_points: float = 0.0
    """Slippage in the same units as stop_distance_points.  Always 0 in paper mode."""


# ─────────────────────────────────────────────────────────────────────────────
# Open Position
# ─────────────────────────────────────────────────────────────────────────────


class OpenPosition(BaseModel, frozen=True):
    """
    An active paper position opened after a fill.

    Immutable snapshot — the engine creates new instances rather than
    mutating.  ExecutionState tracks the current live set.
    """

    position_id: str = Field(default_factory=_new_position_id)
    order: ExecutionOrder
    fill: FillResult

    opened_at: datetime
    """UTC timestamp when the position was opened (= fill timestamp)."""

    bars_open: int = 0
    """Number of bars evaluated since this position was opened."""


# ─────────────────────────────────────────────────────────────────────────────
# Closed Position
# ─────────────────────────────────────────────────────────────────────────────


class ClosedPosition(BaseModel, frozen=True):
    """
    A position that has been closed, with full P&L context.

    P&L is computed using the R-multiple formula:
      r_multiple = direction_sign * (close_price - fill_price) / |stop_price - fill_price|
      pnl_amount = r_multiple * risk_amount

    A stop loss always yields r_multiple = -1.0.
    A 2:1 target yields r_multiple = +2.0.
    These values are set by PaperExecutionEngine._close() — not computed here.
    """

    open_position: OpenPosition
    close_price: float
    close_timestamp: datetime
    close_reason: CloseReason

    pnl_amount: float
    """Net P&L in account currency (USD).  Positive = profit, negative = loss."""

    r_multiple: float
    """
    P&L as a multiple of the initial risk.
    -1.0 = full stop loss.  +2.0 = 2R win.
    """

    bars_held: int
    """Number of bars the position was held."""

    reason_text: str
    """Non-technical explanation, suitable for a user-facing dashboard."""
