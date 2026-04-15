"""
aion/app/loop.py
─────────────────
Paper trading loop — orchestrates all AION components sequentially.

run_paper_loop() is the main entry point.  It receives a list of
MarketSnapshot objects (oldest first) and runs them through the full
paper trading pipeline:

  For each snapshot:
    1. Fill pending orders (filled at this bar's open — next-bar-open model)
    2. Evaluate all open positions against this bar's high/low/close
       (stop loss, take profit, timeout)
    3. Run each strategy engine against this snapshot
    4. For each CANDIDATE: run risk allocation
    5. For each APPROVED: compute price levels, create order, queue for fill

Design notes
─────────────
  - bar_index for evaluate_bar is computed as (i - fill_snapshot_index),
    where i is the current snapshot index and fill_snapshot_index is the
    index at which the position was filled.  This starts at 0 on the fill
    bar itself, allowing stop/target to trigger on the same bar as fill.

  - Risk check uses a portfolio snapshot BEFORE new orders from the
    current iteration are counted.  If two strategies both generate
    signals in the same snapshot, both see the same portfolio state.
    This may allow one extra position in edge cases — known v1 limitation.

  - Pending orders at the end of the sequence (no more data) are counted
    as risk_approved but NOT as executed.

  - No lookahead: strategy engines receive only snapshot[i], which by
    design contains only bars[0..i].
"""

from __future__ import annotations

from datetime import datetime, timezone

from pathlib import Path

from aion.app.orchestrator import PaperTradingConfig
from aion.app.summary import PaperTradingResult, PaperTradingSummary, StrategyBreakdown
from aion.core.enums import TradeDirection
from aion.core.ids import new_pipeline_run_id
from aion.core.models import MarketSnapshot
from aion.execution.execution_model import ExecutionModel, detect_session
from aion.execution.models import ExecutionOrder
from aion.execution.paper import PaperExecutionEngine
from aion.execution.state import ExecutionState
from aion.execution.journal import ExecutionJournal
from aion.risk.allocator import evaluate as risk_evaluate
from aion.strategies.base import StrategyEngine
from aion.strategies.models import StrategyOutcome


_EXECUTION_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "execution_config.yaml"
)


def run_paper_loop(
    snapshots: list[MarketSnapshot],
    engines: list[StrategyEngine],
    config: PaperTradingConfig,
) -> PaperTradingResult:
    """
    Run the full paper trading loop over a sequence of snapshots.

    Parameters
    ----------
    snapshots:
        Historical snapshots in chronological order, oldest first.
        Each snapshot must contain only bars up to its own timestamp
        (no-lookahead guarantee is the caller's responsibility).
    engines:
        List of strategy engines to evaluate on each snapshot.
        Each engine is evaluated independently.  Filters and wrappers
        (QualityFilter, SessionFilter, etc.) should be applied before
        passing engines here.
    config:
        Complete run configuration: risk profile, instrument, stop/target
        distances, pip size, and optional bar timeout.

    Returns
    -------
    PaperTradingResult
        Contains the run summary, live ExecutionState, and ExecutionJournal.
    """
    run_id = new_pipeline_run_id()
    start_time = datetime.now(timezone.utc)

    exec_engine = PaperExecutionEngine()
    state = ExecutionState()
    journal = ExecutionJournal()

    try:
        import random as _random
        execution_model: ExecutionModel | None = ExecutionModel.from_config(
            _EXECUTION_CONFIG_PATH
        )
        # Seed deterministically so paper backtests are reproducible.
        execution_model._rng = _random.Random(0)
    except Exception:
        execution_model = None

    # Per-strategy signal counters
    signals: dict[str, int] = {}
    risk_approved: dict[str, int] = {}
    executed: dict[str, int] = {}

    # Orders created this snapshot; filled at the NEXT snapshot's bar open
    pending_orders: list[ExecutionOrder] = []

    # Per-order execution context captured at signal time
    # order_id -> {"entry_type": str, "atr_1m": float | None}
    order_exec_ctx: dict[str, dict[str, object]] = {}

    # snapshot index when each position was filled  (for bar_index calculation)
    fill_snapshot_idx: dict[str, int] = {}  # position_id -> snapshot index

    for i, snapshot in enumerate(snapshots):
        bar = snapshot.latest_bar

        # ── 1. Fill pending orders at this bar's open ─────────────────────────
        for order in pending_orders:
            slip_points = config.slippage_points
            if execution_model is not None:
                try:
                    ctx = order_exec_ctx.get(order.order_id, {})
                    entry_type = str(ctx.get("entry_type", "retest"))
                    atr_1m = ctx.get("atr_1m")
                    session_label = detect_session(bar.timestamp_utc)
                    spread = execution_model.estimate_spread(
                        order.symbol,
                        atr_1m if isinstance(atr_1m, (int, float)) else None,
                    )
                    slippage = execution_model.estimate_slippage(
                        bar,
                        session=session_label,
                        entry_type=entry_type,
                        symbol=order.symbol,
                    )
                    slip_points = spread / 2.0 + slippage
                except Exception:
                    slip_points = 0.0

            fill, position = exec_engine.fill_order(
                order, bar, slippage_points=slip_points,
            )
            state.add_position(position)
            journal.log_order_filled(fill, position)
            fill_snapshot_idx[position.position_id] = i
            executed[order.strategy_id] = executed.get(order.strategy_id, 0) + 1
            order_exec_ctx.pop(order.order_id, None)
        pending_orders.clear()

        # ── 2. Evaluate all open positions against this bar ───────────────────
        # Includes positions just filled in step 1 (bar_index=0 on fill bar).
        for position in list(state.all_open()):
            bar_idx = i - fill_snapshot_idx[position.position_id]
            closed = exec_engine.evaluate_bar(
                position,
                bar,
                bar_index=bar_idx,
                max_bars_open=config.max_bars_open,
            )
            if closed is not None:
                state.close_position(position.position_id, closed)
                journal.log_position_closed(closed)

        # ── 3. Evaluate strategies, apply risk, queue new orders ──────────────
        portfolio_snapshot = state.to_portfolio_state(
            config.risk_profile.max_risk_per_trade_pct
        )

        for strategy_engine in engines:
            result = strategy_engine.evaluate(snapshot)

            if result.outcome != StrategyOutcome.CANDIDATE:
                continue

            candidate = result.candidate
            sid = candidate.strategy_id
            signals[sid] = signals.get(sid, 0) + 1

            # Risk check against current portfolio state
            decision = risk_evaluate(
                candidate,
                config.risk_profile,
                portfolio_snapshot,
                config.instrument,
                stop_distance_points=config.stop_distance_points,
                target_distance_points=config.target_distance_points,
            )

            if not decision.approved:
                continue

            risk_approved[sid] = risk_approved.get(sid, 0) + 1

            # Price levels: prefer strategy_detail overrides, fall back to config
            sd = candidate.strategy_detail
            if "stop_price" in sd and "target_price" in sd:
                stop_price = sd["stop_price"]
                target_price = sd["target_price"]
            else:
                pip = config.pip_size
                if candidate.direction == TradeDirection.LONG:
                    stop_price = (
                        candidate.entry_reference - config.stop_distance_points * pip
                    )
                    target_price = (
                        candidate.entry_reference + config.target_distance_points * pip
                        if config.target_distance_points is not None
                        else None
                    )
                else:  # SHORT
                    stop_price = (
                        candidate.entry_reference + config.stop_distance_points * pip
                    )
                    target_price = (
                        candidate.entry_reference - config.target_distance_points * pip
                        if config.target_distance_points is not None
                        else None
                    )

            order = exec_engine.create_order(
                decision,
                candidate,
                stop_price=stop_price,
                target_price=target_price,
            )
            journal.log_order_submitted(order)
            pending_orders.append(order)

            entry_type = str(sd.get("entry_type", "retest"))
            order_exec_ctx[order.order_id] = {
                "entry_type": entry_type,
                "atr_1m": candidate.atr_14,
            }

    end_time = datetime.now(timezone.utc)
    elapsed = (end_time - start_time).total_seconds()

    # ── Build summary ─────────────────────────────────────────────────────────
    closed_positions = state.all_closed()

    win_count = sum(1 for c in closed_positions if c.pnl_amount > 0)
    loss_count = sum(1 for c in closed_positions if c.pnl_amount < 0)

    r_values = [c.r_multiple for c in closed_positions]
    avg_r: float | None = (
        round(sum(r_values) / len(r_values), 4) if r_values else None
    )

    all_strategy_ids = sorted(
        set(signals) | set(risk_approved) | set(executed)
    )

    def _strategy_closed(sid: str) -> int:
        return sum(
            1 for c in closed_positions
            if c.open_position.order.strategy_id == sid
        )

    def _strategy_pnl(sid: str) -> float:
        return sum(
            c.pnl_amount for c in closed_positions
            if c.open_position.order.strategy_id == sid
        )

    def _strategy_wins(sid: str) -> int:
        return sum(
            1 for c in closed_positions
            if c.open_position.order.strategy_id == sid and c.pnl_amount > 0
        )

    def _strategy_losses(sid: str) -> int:
        return sum(
            1 for c in closed_positions
            if c.open_position.order.strategy_id == sid and c.pnl_amount < 0
        )

    breakdown = [
        StrategyBreakdown(
            strategy_id=sid,
            signals=signals.get(sid, 0),
            risk_approved=risk_approved.get(sid, 0),
            executed=executed.get(sid, 0),
            closed=_strategy_closed(sid),
            pnl=round(_strategy_pnl(sid), 2),
            win_count=_strategy_wins(sid),
            loss_count=_strategy_losses(sid),
        )
        for sid in all_strategy_ids
    ]

    summary = PaperTradingSummary(
        run_id=run_id,
        snapshots_evaluated=len(snapshots),
        total_signals=sum(signals.values()),
        risk_approved=sum(risk_approved.values()),
        total_executed=sum(executed.values()),
        positions_closed=len(closed_positions),
        positions_still_open=state.open_count,
        total_pnl=round(state.total_realized_pnl, 2),
        win_count=win_count,
        loss_count=loss_count,
        avg_r_multiple=avg_r,
        strategy_breakdown=breakdown,
        elapsed_seconds=elapsed,
    )

    return PaperTradingResult(summary=summary, state=state, journal=journal)
