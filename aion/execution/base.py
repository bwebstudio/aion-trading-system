"""
aion.execution.base
────────────────────
Abstract base class for execution adapters.

STUB — not yet implemented.

Two concrete adapters will be built:
  - PaperAdapter  → simulates fills with slippage model (no real money)
  - Mt5Adapter    → sends orders to MetaTrader5 (live mode only)

The adapter selected at runtime depends on SystemMode (PAPER / LIVE).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ApprovedOrder:
    """A risk-approved order ready to be sent to the broker.  (Placeholder)"""

    order_id: str
    symbol: str
    direction: str
    lot_size: float
    entry_price: float | None  # None = market order
    stop_loss: float
    take_profit: float
    timestamp_utc: datetime


@dataclass(frozen=True)
class FillEvent:
    """A confirmed order fill returned by the execution adapter.  (Placeholder)"""

    order_id: str
    symbol: str
    fill_price: float
    lot_size: float
    timestamp_utc: datetime
    slippage_points: float


class ExecutionAdapter(ABC):
    """Abstract execution adapter.  Subclass for paper and live."""

    @property
    @abstractmethod
    def adapter_id(self) -> str:
        """Unique identifier (e.g. 'paper_v1', 'mt5_live')."""

    @abstractmethod
    def submit(self, order: ApprovedOrder) -> FillEvent:
        """Submit an order and return the resulting fill."""

    @abstractmethod
    def cancel(self, order_id: str) -> bool:
        """Cancel a pending order.  Returns True if successfully cancelled."""
