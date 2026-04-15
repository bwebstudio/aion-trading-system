"""
scripts/run_paper_execution_demo.py
────────────────────────────────────
Demonstration of Execution Engine v1 in paper trading mode.

Shows two scenarios:
  1. Target hit  -- EURUSD OR LONG, price reaches take profit after 3 bars.
  2. Stop hit    -- EURUSD OR SHORT, price hits stop on bar 2.

Each scenario shows the full lifecycle:
  Risk Allocation -> Order Creation -> Fill -> Bar Evaluation -> Close -> Journal

Usage:
  python scripts/run_paper_execution_demo.py

Output is plain text, readable by a non-technical user.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aion.core.enums import AssetClass, DataSource, SessionName, Timeframe, TradeDirection
from aion.core.models import InstrumentSpec, MarketBar
from aion.execution import (
    ExecutionJournal,
    ExecutionState,
    PaperExecutionEngine,
)
from aion.risk import PortfolioState, RiskProfile, evaluate
from aion.strategies.models import CandidateSetup

_UTC = timezone.utc
_TS = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_UTC)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _eurusd() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="EURUSD",
        broker_symbol="EURUSD",
        asset_class=AssetClass.FOREX,
        price_timezone="Etc/UTC",
        market_timezone="Etc/UTC",
        broker_timezone="Etc/UTC",
        tick_size=0.00001,
        point_value=10.0,
        contract_size=100_000.0,
        min_lot=0.01,
        lot_step=0.01,
        currency_profit="USD",
        currency_margin="USD",
        session_calendar="forex_standard",
        trading_hours_label="Sun 22:00 - Fri 22:00 UTC",
    )


def _bar(
    open_: float,
    high: float,
    low: float,
    close: float,
    offset_minutes: int = 1,
) -> MarketBar:
    ts = _TS + timedelta(minutes=offset_minutes)
    return MarketBar(
        symbol="EURUSD",
        timestamp_utc=ts,
        timestamp_market=ts,
        timeframe=Timeframe.M1,
        open=open_,
        high=high,
        low=low,
        close=close,
        tick_volume=150.0,
        real_volume=0.0,
        spread=1.0,
        source=DataSource.CSV,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────


def _section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _bar_line(i: int, bar: MarketBar, status: str) -> None:
    print(
        f"  Bar {i + 1}: O={bar.open:.5f} H={bar.high:.5f} "
        f"L={bar.low:.5f} C={bar.close:.5f}  -> {status}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1: Target hit (LONG)
# ─────────────────────────────────────────────────────────────────────────────


def scenario_1_target_hit() -> None:
    """
    EURUSD OR LONG breakout.
    Stop: 10 pips below entry (1.1010).
    Target: 20 pips above entry (1.1040, 2R).
    Price drifts up and hits target on bar 3.
    """
    _section("Scenario 1 - EURUSD LONG: Target Hit (2R)")

    instrument = _eurusd()
    profile = RiskProfile(
        account_equity=10_000.0,
        max_risk_per_trade_pct=1.0,
        max_daily_risk_pct=2.0,
        max_concurrent_positions=3,
        max_positions_per_strategy=2,
    )
    state = ExecutionState()
    journal = ExecutionJournal()
    engine = PaperExecutionEngine()

    # --- Risk Allocation ---
    candidate = CandidateSetup(
        strategy_id="or_london_v1",
        strategy_version="1.0.0",
        symbol="EURUSD",
        timestamp_utc=_TS,
        direction=TradeDirection.LONG,
        entry_reference=1.1020,
        range_high=1.1020,
        range_low=1.1000,
        range_size_pips=20.0,
        session_name=SessionName.LONDON.value,
        quality_score=1.0,
        atr_14=0.00015,
    )
    portfolio_snapshot = state.to_portfolio_state(risk_per_trade_pct=1.0)
    decision = evaluate(
        candidate, profile, portfolio_snapshot, instrument,
        stop_distance_points=10.0,
        target_distance_points=20.0,
    )

    print(f"  Risk decision  : {'APPROVED' if decision.approved else 'REJECTED'}")
    print(f"  Position size  : {decision.position_size} lot(s)")
    print(f"  Risk amount    : ${decision.risk_amount:.2f}")

    # --- Order ---
    # LONG: stop 10 pips below entry, target 20 pips above entry
    # 1 pip = 0.0001 for EURUSD (tick_size * 10)
    stop_price = 1.1020 - 10 * 0.0001   # 1.1010
    target_price = 1.1020 + 20 * 0.0001  # 1.1040
    order = engine.create_order(decision, candidate, stop_price=stop_price, target_price=target_price)
    journal.log_order_submitted(order)
    print(f"  Entry ref      : {order.entry_price:.5f}")
    print(f"  Stop price     : {order.stop_price:.5f}")
    print(f"  Target price   : {order.target_price:.5f}")

    # --- Fill (next bar open) ---
    fill_bar = _bar(open_=1.1021, high=1.1025, low=1.1018, close=1.1023, offset_minutes=1)
    fill, position = engine.fill_order(order, fill_bar)
    state.add_position(position)
    journal.log_order_filled(fill, position)
    print(f"  Filled at      : {fill.fill_price:.5f} (bar open)")

    # --- Bar evaluation ---
    eval_bars = [
        _bar(1.1023, 1.1030, 1.1018, 1.1028, offset_minutes=2),  # holds
        _bar(1.1028, 1.1035, 1.1022, 1.1032, offset_minutes=3),  # holds
        _bar(1.1030, 1.1042, 1.1025, 1.1040, offset_minutes=4),  # target hit
    ]

    print()
    closed = None
    for i, bar in enumerate(eval_bars):
        result = engine.evaluate_bar(position, bar, bar_index=i)
        if result is not None:
            _bar_line(i, bar, f"CLOSED - {result.close_reason.value}")
            closed = result
            break
        else:
            _bar_line(i, bar, "open")

    if closed:
        state.close_position(position.position_id, closed)
        journal.log_position_closed(closed)
        print()
        print(f"  Close price    : {closed.close_price:.5f}")
        print(f"  P&L            : ${closed.pnl_amount:+.2f}")
        print(f"  R-multiple     : {closed.r_multiple:+.2f}R")
        print(f"  Bars held      : {closed.bars_held}")
        print(f"  Reason         : {closed.reason_text}")

    # --- Journal ---
    print()
    print("  Journal events:")
    for e in journal.all_events():
        print(f"    [{e.event_type.value}] {e.reason_text}")

    # --- Final state ---
    print()
    print(f"  Open positions  : {state.open_count}")
    print(f"  Closed positions: {state.closed_count}")
    print(f"  Total P&L       : ${state.total_realized_pnl:+.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2: Stop hit (SHORT)
# ─────────────────────────────────────────────────────────────────────────────


def scenario_2_stop_hit() -> None:
    """
    EURUSD OR SHORT breakout.
    Stop: 10 pips above entry (1.1030).
    Target: 20 pips below entry (1.1000, 2R).
    Price reverses and hits stop on bar 2.
    """
    _section("Scenario 2 - EURUSD SHORT: Stop Hit (-1R)")

    instrument = _eurusd()
    profile = RiskProfile(
        account_equity=10_000.0,
        max_risk_per_trade_pct=1.0,
        max_daily_risk_pct=2.0,
        max_concurrent_positions=3,
        max_positions_per_strategy=2,
        allow_same_direction_multiple=True,
    )
    state = ExecutionState()
    journal = ExecutionJournal()
    engine = PaperExecutionEngine()

    # --- Risk Allocation ---
    candidate = CandidateSetup(
        strategy_id="or_london_v1",
        strategy_version="1.0.0",
        symbol="EURUSD",
        timestamp_utc=_TS,
        direction=TradeDirection.SHORT,
        entry_reference=1.1020,
        range_high=1.1040,
        range_low=1.1020,
        range_size_pips=20.0,
        session_name=SessionName.LONDON.value,
        quality_score=1.0,
        atr_14=0.00015,
    )
    portfolio_snapshot = state.to_portfolio_state(risk_per_trade_pct=1.0)
    decision = evaluate(
        candidate, profile, portfolio_snapshot, instrument,
        stop_distance_points=10.0,
        target_distance_points=20.0,
    )

    print(f"  Risk decision  : {'APPROVED' if decision.approved else 'REJECTED'}")
    print(f"  Position size  : {decision.position_size} lot(s)")
    print(f"  Risk amount    : ${decision.risk_amount:.2f}")

    # --- Order ---
    # SHORT: stop 10 pips above entry, target 20 pips below entry
    stop_price = 1.1020 + 10 * 0.0001   # 1.1030
    target_price = 1.1020 - 20 * 0.0001  # 1.1000
    order = engine.create_order(decision, candidate, stop_price=stop_price, target_price=target_price)
    journal.log_order_submitted(order)
    print(f"  Entry ref      : {order.entry_price:.5f}")
    print(f"  Stop price     : {order.stop_price:.5f}")
    print(f"  Target price   : {order.target_price:.5f}")

    # --- Fill ---
    fill_bar = _bar(open_=1.1019, high=1.1022, low=1.1015, close=1.1017, offset_minutes=1)
    fill, position = engine.fill_order(order, fill_bar)
    state.add_position(position)
    journal.log_order_filled(fill, position)
    print(f"  Filled at      : {fill.fill_price:.5f} (bar open)")

    # --- Bar evaluation ---
    eval_bars = [
        _bar(1.1017, 1.1025, 1.1012, 1.1020, offset_minutes=2),  # holds
        _bar(1.1020, 1.1032, 1.1018, 1.1030, offset_minutes=3),  # stop hit
    ]

    print()
    closed = None
    for i, bar in enumerate(eval_bars):
        result = engine.evaluate_bar(position, bar, bar_index=i)
        if result is not None:
            _bar_line(i, bar, f"CLOSED - {result.close_reason.value}")
            closed = result
            break
        else:
            _bar_line(i, bar, "open")

    if closed:
        state.close_position(position.position_id, closed)
        journal.log_position_closed(closed)
        print()
        print(f"  Close price    : {closed.close_price:.5f}")
        print(f"  P&L            : ${closed.pnl_amount:+.2f}")
        print(f"  R-multiple     : {closed.r_multiple:+.2f}R")
        print(f"  Bars held      : {closed.bars_held}")
        print(f"  Reason         : {closed.reason_text}")

    # --- Journal ---
    print()
    print("  Journal events:")
    for e in journal.all_events():
        print(f"    [{e.event_type.value}] {e.reason_text}")

    # --- Final state ---
    print()
    print(f"  Open positions  : {state.open_count}")
    print(f"  Closed positions: {state.closed_count}")
    print(f"  Total P&L       : ${state.total_realized_pnl:+.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print()
    print("Execution Engine v1 - Paper Trading Demo")
    print("=" * 45)
    print("Account: $10,000 | Risk: 1% per trade | Instrument: EURUSD")
    print("Fill model: next-bar-open | Slippage: 0")

    scenario_1_target_hit()
    scenario_2_stop_hit()

    print()


if __name__ == "__main__":
    main()
