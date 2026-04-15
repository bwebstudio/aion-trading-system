"""
aion/execution/state.py
────────────────────────
Mutable portfolio state for the paper execution engine.

ExecutionState is the single source of truth for open and closed positions.
It also bridges to the risk allocator by generating an immutable PortfolioState
snapshot, so the same risk rules that govern the live system apply in paper mode.

Design:
  - Mutable class (not frozen) — it must be updated as positions open/close.
  - Internal storage uses a dict keyed by position_id for O(1) lookups.
  - to_portfolio_state() returns an immutable PortfolioState for the risk
    allocator.  Caller passes risk_per_trade_pct so daily_risk_used_pct is
    computed correctly from the current open position count.
"""

from __future__ import annotations

from aion.execution.models import ClosedPosition, OpenPosition
from aion.risk.models import PortfolioState


class ExecutionState:
    """
    Mutable state for paper execution: open and closed positions.

    Usage:
      state = ExecutionState()
      state.add_position(open_pos)

      closed = engine.evaluate_bar(open_pos, bar, ...)
      if closed:
          state.close_position(open_pos.position_id, closed)

      snapshot = state.to_portfolio_state(risk_per_trade_pct=1.0)
    """

    def __init__(self) -> None:
        self._open: dict[str, OpenPosition] = {}
        self._closed: list[ClosedPosition] = []

    # ── Mutations ─────────────────────────────────────────────────────────────

    def add_position(self, position: OpenPosition) -> None:
        """Register a newly opened position."""
        self._open[position.position_id] = position

    def close_position(self, position_id: str, closed: ClosedPosition) -> None:
        """
        Move a position from open to closed.

        Raises
        ------
        KeyError
            If position_id is not in the open set.
        """
        if position_id not in self._open:
            raise KeyError(f"Position '{position_id}' not found in open positions.")
        del self._open[position_id]
        self._closed.append(closed)

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_open(self, position_id: str) -> OpenPosition | None:
        """Return an open position by ID, or None if not found."""
        return self._open.get(position_id)

    def all_open(self) -> list[OpenPosition]:
        """Snapshot of all currently open positions."""
        return list(self._open.values())

    def all_closed(self) -> list[ClosedPosition]:
        """All closed positions in chronological close order."""
        return list(self._closed)

    @property
    def open_count(self) -> int:
        return len(self._open)

    @property
    def closed_count(self) -> int:
        return len(self._closed)

    @property
    def total_realized_pnl(self) -> float:
        """Sum of all closed position P&L (account currency)."""
        return sum(c.pnl_amount for c in self._closed)

    # ── Risk module bridge ────────────────────────────────────────────────────

    def to_portfolio_state(self, risk_per_trade_pct: float = 1.0) -> PortfolioState:
        """
        Generate an immutable PortfolioState snapshot for the risk allocator.

        daily_risk_used_pct is estimated as open_count * risk_per_trade_pct,
        which is exact for fixed-fractional sizing.

        Parameters
        ----------
        risk_per_trade_pct:
            Risk percentage per trade (from RiskProfile.max_risk_per_trade_pct).
        """
        by_strategy: dict[str, int] = {}
        by_direction: dict[str, int] = {}

        for pos in self._open.values():
            sid = pos.order.strategy_id
            by_strategy[sid] = by_strategy.get(sid, 0) + 1

            dir_key = pos.order.direction.value  # "LONG" or "SHORT"
            by_direction[dir_key] = by_direction.get(dir_key, 0) + 1

        daily_realized = sum(c.pnl_amount for c in self._closed)
        daily_risk_used = len(self._open) * risk_per_trade_pct

        return PortfolioState(
            open_positions_count=len(self._open),
            open_positions_by_strategy=by_strategy,
            open_positions_by_direction=by_direction,
            daily_realized_pnl=daily_realized,
            daily_unrealized_pnl=0.0,  # unrealized P&L not tracked in v1
            daily_risk_used_pct=daily_risk_used,
        )
