"""
aion/risk/rules.py
───────────────────
Pure rule-check functions for Risk Allocation v1.

Each function receives the inputs it needs and returns either:
  (reason_code, reason_text)  if the rule is violated  →  trade blocked
  None                        if the check passes       →  continue

Functions are stateless and have no side effects.  They can be called
independently for testing or used by the allocator in sequence.

Evaluation order used by the allocator:
  1. check_equity                  — account is funded
  2. check_stop_distance           — stop is a valid positive number
  3. check_max_positions           — concurrent position limit not exceeded
  4. check_max_strategy_positions  — per-strategy limit not exceeded
  5. check_same_direction          — direction constraint (if configured)
  6. check_daily_risk              — daily risk budget not exhausted

reason_text is intentionally non-technical: it will be shown on a
user-facing dashboard and must be clear to a non-expert trader.
"""

from __future__ import annotations

from aion.core.enums import TradeDirection
from aion.risk.models import PortfolioState, RiskProfile


def check_equity(profile: RiskProfile) -> tuple[str, str] | None:
    """Equity must be positive to size any trade.

    This rule guards against a degenerate account state (equity wiped out
    or negative due to an extreme loss event).
    """
    if profile.account_equity <= 0:
        return (
            "INVALID_EQUITY",
            f"The account balance ({profile.account_equity:.2f}) is zero or negative. "
            "No new trades can be opened until the account is funded.",
        )
    return None


def check_stop_distance(stop_distance_points: float) -> tuple[str, str] | None:
    """Stop distance must be a positive number to compute a valid position size.

    A zero or negative stop makes risk calculation impossible and likely
    indicates a configuration error in the calling code.
    """
    if stop_distance_points <= 0:
        return (
            "INVALID_STOP_DISTANCE",
            f"The stop distance ({stop_distance_points}) is not valid. "
            "A positive stop distance is required to calculate the trade size.",
        )
    return None


def check_max_positions(
    state: PortfolioState,
    profile: RiskProfile,
) -> tuple[str, str] | None:
    """Total open positions must not exceed the configured maximum.

    Limits portfolio-wide exposure regardless of strategy or direction.
    """
    if state.open_positions_count >= profile.max_concurrent_positions:
        return (
            "MAX_POSITIONS_REACHED",
            f"You already have {state.open_positions_count} open trade(s) "
            f"(the maximum is {profile.max_concurrent_positions}). "
            "Wait for one to close before opening a new position.",
        )
    return None


def check_max_strategy_positions(
    state: PortfolioState,
    profile: RiskProfile,
    strategy_id: str,
) -> tuple[str, str] | None:
    """Open positions for a single strategy must not exceed the configured maximum.

    Prevents over-concentration in one strategy even when the total
    position count is within the global limit.
    """
    count = state.open_positions_by_strategy.get(strategy_id, 0)
    if count >= profile.max_positions_per_strategy:
        return (
            "MAX_STRATEGY_POSITIONS_REACHED",
            f"This strategy already has {count} open trade(s) "
            f"(the maximum per strategy is {profile.max_positions_per_strategy}). "
            "Wait for one of its trades to close before taking a new signal.",
        )
    return None


def check_same_direction(
    state: PortfolioState,
    profile: RiskProfile,
    direction: TradeDirection,
) -> tuple[str, str] | None:
    """If allow_same_direction_multiple=False, block a second trade in the same direction.

    This rule enforces a simple directional discipline: one active LONG and
    one active SHORT at most (across all strategies combined).
    """
    if profile.allow_same_direction_multiple:
        return None  # rule disabled by configuration
    count = state.open_positions_by_direction.get(direction.value, 0)
    if count > 0:
        dir_label = "long (buy)" if direction == TradeDirection.LONG else "short (sell)"
        return (
            "SAME_DIRECTION_NOT_ALLOWED",
            f"You already have an open {dir_label} position. "
            "Your current settings do not allow multiple trades in the same "
            "direction at the same time.",
        )
    return None


def check_daily_risk(
    state: PortfolioState,
    profile: RiskProfile,
    new_risk_pct: float,
) -> tuple[str, str] | None:
    """Adding this trade must not push the day's total risk over the daily limit.

    daily_risk_used_pct tracks risk already committed in open positions.
    new_risk_pct is the risk this new trade would add (typically max_risk_per_trade_pct).
    """
    projected = state.daily_risk_used_pct + new_risk_pct
    if projected > profile.max_daily_risk_pct:
        return (
            "MAX_DAILY_RISK_REACHED",
            f"Opening this trade would bring your total daily risk to {projected:.1f}% "
            f"of your account (the daily limit is {profile.max_daily_risk_pct:.1f}%). "
            "No more trades can be opened today.",
        )
    return None
