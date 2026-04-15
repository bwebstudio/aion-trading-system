"""
aion/execution/paper.py
────────────────────────
Paper Execution Engine v1.

Simulates order fills and manages the position lifecycle without real money.
All fills are reproducible and deterministic.

Fill model (next-bar-open):
  - Order is filled at the open price of the bar following the signal.
  - No slippage in paper mode (slippage_points = 0).
  - Stop and target are evaluated intrabar using bar high/low.

Stop / target trigger logic:
  LONG position:
    - Stop triggered if bar.low  <= stop_price  -> closes at stop_price
    - Target triggered if bar.high >= target_price -> closes at target_price
  SHORT position:
    - Stop triggered if bar.high >= stop_price  -> closes at stop_price
    - Target triggered if bar.low  <= target_price -> closes at target_price

  If both stop and target are triggered on the same bar, stop wins (conservative).
  If timeout is reached (bar_index >= max_bars_open - 1), closes at bar.close.

P&L formula (R-multiple, no InstrumentSpec needed at close):
  r_multiple = direction_sign * (close_price - fill_price) / |stop_price - fill_price|
  pnl_amount = r_multiple * risk_amount

This engine is stateless — it creates and evaluates objects without holding
any mutable state.  ExecutionState handles persistence.
"""

from __future__ import annotations

from datetime import datetime

from aion.core.enums import TradeDirection
from aion.core.models import MarketBar
from aion.execution.models import (
    CloseReason,
    ClosedPosition,
    ExecutionOrder,
    FillResult,
    OpenPosition,
)
from aion.risk.models import RiskDecision
from aion.strategies.models import CandidateSetup


class PaperExecutionEngine:
    """
    Stateless engine for paper order execution and position management.

    Usage:
      engine  = PaperExecutionEngine()
      order   = engine.create_order(decision, candidate, stop_price=1.1010)
      fill, position = engine.fill_order(order, next_bar)
      closed  = engine.evaluate_bar(position, bar, bar_index=0, max_bars_open=20)
    """

    def create_order(
        self,
        decision: RiskDecision,
        candidate: CandidateSetup,
        stop_price: float,
        target_price: float | None = None,
    ) -> ExecutionOrder:
        """
        Create an execution order from a risk-approved decision.

        Parameters
        ----------
        decision:
            Approved RiskDecision.  Must have approved=True.
        candidate:
            The CandidateSetup that generated the signal.
        stop_price:
            Stop loss price in instrument price units.
            For LONG: below entry.  For SHORT: above entry.
        target_price:
            Take profit price in instrument price units.  Optional.
            For LONG: above entry.  For SHORT: below entry.

        Raises
        ------
        ValueError
            If decision.approved is False or sizing fields are missing.
        """
        if not decision.approved:
            raise ValueError(
                f"Cannot create an order from a rejected RiskDecision "
                f"(reason: {decision.reason_code})."
            )
        if decision.position_size is None or decision.risk_amount is None:
            raise ValueError(
                "Approved RiskDecision must have position_size and risk_amount."
            )

        return ExecutionOrder(
            setup_id=candidate.setup_id,
            strategy_id=candidate.strategy_id,
            symbol=candidate.symbol,
            direction=candidate.direction,
            entry_price=candidate.entry_reference,
            stop_price=stop_price,
            target_price=target_price,
            position_size=decision.position_size,
            risk_amount=decision.risk_amount,
            stop_distance_points=decision.stop_distance_points or 0.0,
            target_distance_points=decision.target_distance_points,
            created_at=candidate.timestamp_utc,
        )

    def fill_order(
        self,
        order: ExecutionOrder,
        fill_bar: MarketBar,
        slippage_points: float = 0.0,
    ) -> tuple[FillResult, OpenPosition]:
        """
        Simulate an order fill at the open of fill_bar.

        In paper trading, fill_bar is the bar immediately following
        the signal bar (next-bar-open model).

        Parameters
        ----------
        slippage_points:
            Execution slippage in price points.  Applied adversely:
            LONG: fill = bar.open + slippage (worse entry, higher price)
            SHORT: fill = bar.open - slippage (worse entry, lower price)
            Default 0.0 preserves current behaviour.

        Returns
        -------
        (FillResult, OpenPosition)
        """
        if slippage_points != 0.0:
            from aion.core.enums import TradeDirection
            if order.direction == TradeDirection.LONG:
                fill_price = fill_bar.open + slippage_points
            else:
                fill_price = fill_bar.open - slippage_points
        else:
            fill_price = fill_bar.open

        fill = FillResult(
            order_id=order.order_id,
            fill_price=fill_price,
            fill_timestamp=fill_bar.timestamp_utc,
            slippage_points=slippage_points,
        )
        position = OpenPosition(
            order=order,
            fill=fill,
            opened_at=fill_bar.timestamp_utc,
            bars_open=0,
        )
        return fill, position

    def evaluate_bar(
        self,
        position: OpenPosition,
        bar: MarketBar,
        bar_index: int,
        max_bars_open: int | None = None,
    ) -> ClosedPosition | None:
        """
        Check whether a position should close on the given bar.

        Evaluation order (first match wins):
          1. Stop loss   — conservative, checked before target
          2. Take profit
          3. Timeout     — only if max_bars_open is set

        Returns None if the position should remain open.

        Parameters
        ----------
        position:
            The open position to evaluate.
        bar:
            The current bar being evaluated.
        bar_index:
            0-based index of evaluated bars since the fill bar.
            bar_index=0 is the first full bar after fill.
        max_bars_open:
            If set, close at bar.close when bar_index >= max_bars_open - 1.
        """
        order = position.order

        # ── 1. Stop loss ──────────────────────────────────────────────────────
        stop_hit = (
            order.direction == TradeDirection.LONG and bar.low <= order.stop_price
        ) or (
            order.direction == TradeDirection.SHORT and bar.high >= order.stop_price
        )
        if stop_hit:
            return self._close(
                position,
                close_price=order.stop_price,
                close_ts=bar.timestamp_utc,
                close_reason=CloseReason.STOP_LOSS,
                bars_held=bar_index + 1,
            )

        # ── 2. Take profit ────────────────────────────────────────────────────
        if order.target_price is not None:
            target_hit = (
                order.direction == TradeDirection.LONG
                and bar.high >= order.target_price
            ) or (
                order.direction == TradeDirection.SHORT
                and bar.low <= order.target_price
            )
            if target_hit:
                return self._close(
                    position,
                    close_price=order.target_price,
                    close_ts=bar.timestamp_utc,
                    close_reason=CloseReason.TAKE_PROFIT,
                    bars_held=bar_index + 1,
                )

        # ── 3. Timeout ────────────────────────────────────────────────────────
        if max_bars_open is not None and bar_index >= max_bars_open - 1:
            return self._close(
                position,
                close_price=bar.close,
                close_ts=bar.timestamp_utc,
                close_reason=CloseReason.TIMEOUT,
                bars_held=bar_index + 1,
            )

        return None

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _close(
        self,
        position: OpenPosition,
        close_price: float,
        close_ts: datetime,
        close_reason: CloseReason,
        bars_held: int,
    ) -> ClosedPosition:
        """Compute P&L and build a ClosedPosition."""
        order = position.order
        fill_price = position.fill.fill_price
        direction_sign = 1.0 if order.direction == TradeDirection.LONG else -1.0

        stop_distance_price = abs(order.stop_price - fill_price)
        if stop_distance_price == 0.0:
            r_multiple = 0.0
        else:
            r_multiple = (
                direction_sign * (close_price - fill_price) / stop_distance_price
            )

        pnl_amount = r_multiple * order.risk_amount

        reason_texts = {
            CloseReason.STOP_LOSS: (
                f"Trade closed at stop loss ({close_price:.5f}). "
                f"Loss: ${abs(pnl_amount):.2f}."
            ),
            CloseReason.TAKE_PROFIT: (
                f"Trade closed at take profit ({close_price:.5f}). "
                f"Profit: ${pnl_amount:.2f}."
            ),
            CloseReason.TIMEOUT: (
                f"Trade closed after {bars_held} bar(s) — timeout. "
                f"{'Profit' if pnl_amount >= 0 else 'Loss'}: ${abs(pnl_amount):.2f}."
            ),
            CloseReason.MANUAL: (
                f"Trade closed manually at {close_price:.5f}. "
                f"{'Profit' if pnl_amount >= 0 else 'Loss'}: ${abs(pnl_amount):.2f}."
            ),
        }

        return ClosedPosition(
            open_position=position,
            close_price=close_price,
            close_timestamp=close_ts,
            close_reason=close_reason,
            pnl_amount=round(pnl_amount, 2),
            r_multiple=round(r_multiple, 4),
            bars_held=bars_held,
            reason_text=reason_texts[close_reason],
        )
