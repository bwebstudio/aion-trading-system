"""
scripts/run_risk_allocation_demo.py
-------------------------------------
Demonstration of Risk Allocation v1.

Shows five scenarios:
  1. Approved  -- EURUSD, clean portfolio, sizing computed.
  2. Approved  -- US100.cash, index points sizing.
  3. Rejected  -- portfolio at the concurrent position limit.
  4. Rejected  -- daily risk budget exhausted.
  5. Rejected  -- same direction not allowed.

Usage:
  python scripts/run_risk_allocation_demo.py

Output is plain text, readable by a non-technical user.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aion.core.enums import AssetClass, SessionName, TradeDirection
from aion.core.models import InstrumentSpec
from aion.risk import PortfolioState, RiskDecision, RiskProfile, evaluate
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


def _us100() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="US100.cash",
        broker_symbol="US100.cash",
        asset_class=AssetClass.INDICES,
        price_timezone="America/New_York",
        market_timezone="America/New_York",
        broker_timezone="Etc/UTC",
        tick_size=0.01,
        point_value=1.0,
        contract_size=1.0,
        min_lot=0.1,
        lot_step=0.1,
        currency_profit="USD",
        currency_margin="USD",
        session_calendar="us_equity",
        trading_hours_label="Mon-Fri 09:30-16:00 ET",
    )


def _profile(
    equity: float = 10_000.0,
    risk_pct: float = 1.0,
    daily_limit: float = 2.0,
    max_positions: int = 3,
    max_per_strategy: int = 2,
    allow_same_direction_multiple: bool = False,
) -> RiskProfile:
    return RiskProfile(
        account_equity=equity,
        max_risk_per_trade_pct=risk_pct,
        max_daily_risk_pct=daily_limit,
        max_concurrent_positions=max_positions,
        max_positions_per_strategy=max_per_strategy,
        allow_same_direction_multiple=allow_same_direction_multiple,
    )


def _candidate(
    strategy_id: str = "or_london_v1",
    direction: TradeDirection = TradeDirection.LONG,
    entry: float = 1.1020,
    or_high: float = 1.1020,
    or_low: float = 1.1000,
) -> CandidateSetup:
    return CandidateSetup(
        strategy_id=strategy_id,
        strategy_version="1.0.0",
        symbol="EURUSD",
        timestamp_utc=_TS,
        direction=direction,
        entry_reference=entry,
        range_high=or_high,
        range_low=or_low,
        range_size_pips=20.0,
        session_name=SessionName.LONDON.value,
        quality_score=1.0,
        atr_14=0.00015,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────


def _section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _print_decision(decision: RiskDecision, label: str) -> None:
    verdict = "APPROVED" if decision.approved else "REJECTED"
    print(f"  [{verdict}] {label}")
    print(f"    reason_code     : {decision.reason_code}")
    print(f"    reason_text     : {decision.reason_text}")
    if decision.approved:
        print(f"    position_size   : {decision.position_size} lot(s)")
        print(f"    risk_amount     : ${decision.risk_amount:.2f}")
        print(f"    stop            : {decision.stop_distance_points} pips")
        if decision.target_distance_points is not None:
            print(f"    target          : {decision.target_distance_points} pips")


def _print_profile(profile: RiskProfile) -> None:
    print(f"  account_equity         : ${profile.account_equity:,.2f}")
    print(f"  max_risk_per_trade_pct : {profile.max_risk_per_trade_pct}%")
    print(f"  max_daily_risk_pct     : {profile.max_daily_risk_pct}%")
    print(f"  max_concurrent_pos     : {profile.max_concurrent_positions}")
    print(f"  max_per_strategy       : {profile.max_positions_per_strategy}")
    print(f"  allow_same_direction   : {profile.allow_same_direction_multiple}")


def _print_state(state: PortfolioState) -> None:
    print(f"  open_positions         : {state.open_positions_count}")
    print(f"  daily_risk_used        : {state.daily_risk_used_pct}%")
    if state.open_positions_by_strategy:
        for sid, count in state.open_positions_by_strategy.items():
            print(f"    {sid}: {count} open")


# ─────────────────────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────────────────────


def scenario_1_approved_eurusd() -> None:
    """
    Clean state: EURUSD OR breakout, $10 000 account, 1% risk, 10-pip stop.
    Expected: approved, 1.00 lot, $100 at risk.
    """
    _section("Scenario 1 -EURUSD: Approved trade")

    profile = _profile(equity=10_000.0, risk_pct=1.0)
    state = PortfolioState()
    candidate = _candidate(strategy_id="or_london_v1", direction=TradeDirection.LONG)

    print("  Risk profile:")
    _print_profile(profile)
    print("  Portfolio state:")
    _print_state(state)
    print("  Signal: OR LONG breakout, entry=1.1020, stop=10 pips, target=20 pips")

    decision = evaluate(
        candidate,
        profile,
        state,
        _eurusd(),
        stop_distance_points=10.0,
        target_distance_points=20.0,
    )
    print()
    _print_decision(decision, "EURUSD OR LONG, 10-pip stop")


def scenario_2_approved_us100() -> None:
    """
    US100.cash OR breakout, $25 000 account, 0.5% risk, 10-point stop.
    Expected: approved. Sizing: $125 / (10 * $1) = 12.5 lots (floored to 12.5).
    """
    _section("Scenario 2 -US100.cash: Approved trade (index points)")

    profile = RiskProfile(
        account_equity=25_000.0,
        max_risk_per_trade_pct=0.5,
        max_daily_risk_pct=1.5,
        max_concurrent_positions=3,
        max_positions_per_strategy=2,
        allow_same_direction_multiple=False,
    )
    state = PortfolioState()
    candidate = CandidateSetup(
        strategy_id="or_us100_v1",
        strategy_version="1.0.0",
        symbol="US100.cash",
        timestamp_utc=_TS,
        direction=TradeDirection.LONG,
        entry_reference=18_020.0,
        range_high=18_020.0,
        range_low=18_000.0,
        range_size_pips=20.0,
        session_name=SessionName.NEW_YORK.value,
        quality_score=1.0,
        atr_14=50.0,
    )

    print("  Risk profile:")
    _print_profile(profile)
    print("  Portfolio state:")
    _print_state(state)
    print("  Signal: US100 OR LONG breakout, entry=18020, stop=10 points, target=20 points")
    print("  Sizing: $25000 * 0.5% / (10 pts * $1/pt/lot) = 12.5 lots")

    decision = evaluate(
        candidate,
        profile,
        state,
        _us100(),
        stop_distance_points=10.0,
        target_distance_points=20.0,
    )
    print()
    _print_decision(decision, "US100 OR LONG, 10-point stop")


def scenario_3_rejected_max_positions() -> None:
    """
    Portfolio already has 3 open positions (at the limit).
    Expected: rejected with MAX_POSITIONS_REACHED.
    """
    _section("Scenario 3 -Rejected: too many open positions")

    profile = _profile(max_positions=3)
    state = PortfolioState(
        open_positions_count=3,
        open_positions_by_strategy={
            "or_london_v1": 2,
            "vwap_fade_london_v1": 1,
        },
        daily_risk_used_pct=3.0,
    )
    candidate = _candidate(strategy_id="or_london_v1")

    print("  Risk profile:")
    _print_profile(profile)
    print("  Portfolio state (already full):")
    _print_state(state)

    decision = evaluate(candidate, profile, state, _eurusd(), stop_distance_points=10.0)
    print()
    _print_decision(decision, "EURUSD OR LONG -portfolio full")


def scenario_4_rejected_daily_risk() -> None:
    """
    Daily risk budget almost exhausted (1.5% used, 2% limit, new trade needs 1%).
    Expected: rejected with MAX_DAILY_RISK_REACHED.
    """
    _section("Scenario 4 -Rejected: daily risk budget exhausted")

    profile = _profile(risk_pct=1.0, daily_limit=2.0, max_positions=5)
    state = PortfolioState(
        open_positions_count=1,
        open_positions_by_strategy={"or_london_v1": 1},
        daily_risk_used_pct=1.5,   # 1.5% used + 1% new = 2.5% > 2% limit
        daily_realized_pnl=-80.0,
    )
    candidate = _candidate(strategy_id="or_london_v1")

    print("  Risk profile:")
    _print_profile(profile)
    print("  Portfolio state:")
    _print_state(state)
    print("  Adding 1% risk would bring total to 2.5% (limit: 2%)")

    decision = evaluate(candidate, profile, state, _eurusd(), stop_distance_points=10.0)
    print()
    _print_decision(decision, "EURUSD OR LONG -daily budget exhausted")


def scenario_5_rejected_same_direction() -> None:
    """
    Already have an open LONG, allow_same_direction_multiple=False.
    Expected: rejected with SAME_DIRECTION_NOT_ALLOWED.
    """
    _section("Scenario 5 -Rejected: same direction not allowed")

    profile = _profile(allow_same_direction_multiple=False)
    state = PortfolioState(
        open_positions_count=1,
        open_positions_by_strategy={"vwap_fade_london_v1": 1},
        open_positions_by_direction={"LONG": 1},
    )
    candidate = _candidate(strategy_id="or_london_v1", direction=TradeDirection.LONG)

    print("  Risk profile:")
    _print_profile(profile)
    print("  Portfolio state:")
    _print_state(state)
    print("  Already have 1 LONG open. New signal is also LONG.")

    decision = evaluate(candidate, profile, state, _eurusd(), stop_distance_points=10.0)
    print()
    _print_decision(decision, "EURUSD OR LONG -direction blocked")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print()
    print("Risk Allocation v1 -Demo")
    print("=" * 40)
    print("Account: $10 000 | Risk: 1% per trade | Daily limit: 2%")
    print("Instrument: EURUSD (point_value=$10/pip/lot, lot_step=0.01)")

    scenario_1_approved_eurusd()
    scenario_2_approved_us100()
    scenario_3_rejected_max_positions()
    scenario_4_rejected_daily_risk()
    scenario_5_rejected_same_direction()

    print()


if __name__ == "__main__":
    main()
