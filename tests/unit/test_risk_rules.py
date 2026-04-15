"""
tests/unit/test_risk_rules.py
──────────────────────────────
Unit tests for aion.risk.rules.

Each rule function is tested independently:
  check_equity               — passes/fails on equity sign
  check_stop_distance        — passes/fails on stop sign
  check_max_positions        — passes/fails on position count vs limit
  check_max_strategy_positions — passes/fails per strategy
  check_same_direction       — passes when disabled, passes/fails when enabled
  check_daily_risk           — passes/fails on projected daily risk vs limit
"""

from __future__ import annotations

import pytest

from aion.core.enums import TradeDirection
from aion.risk.models import PortfolioState, RiskProfile
from aion.risk.rules import (
    check_daily_risk,
    check_equity,
    check_max_positions,
    check_max_strategy_positions,
    check_same_direction,
    check_stop_distance,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _profile(**overrides) -> RiskProfile:
    defaults = dict(
        account_equity=10_000.0,
        max_risk_per_trade_pct=1.0,
        max_daily_risk_pct=2.0,
        max_concurrent_positions=3,
        max_positions_per_strategy=2,
        allow_same_direction_multiple=False,
    )
    defaults.update(overrides)
    return RiskProfile(**defaults)


def _state(**overrides) -> PortfolioState:
    return PortfolioState(**overrides)


# ─────────────────────────────────────────────────────────────────────────────
# check_equity
# ─────────────────────────────────────────────────────────────────────────────


def test_check_equity_passes_with_positive_equity():
    """Standard positive equity passes."""
    profile = _profile(account_equity=10_000.0)
    assert check_equity(profile) is None


def test_check_equity_fails_with_zero_equity():
    """Equity of exactly 0 is rejected.
    Note: RiskProfile validator also rejects 0, so we test via a workaround."""
    # We test the rule function directly by bypassing the model validator
    # using model_construct (no validation)
    profile = RiskProfile.model_construct(account_equity=0.0)
    result = check_equity(profile)
    assert result is not None
    code, text = result
    assert code == "INVALID_EQUITY"
    assert len(text) > 0


def test_check_equity_fails_with_negative_equity():
    profile = RiskProfile.model_construct(account_equity=-500.0)
    result = check_equity(profile)
    assert result is not None
    assert result[0] == "INVALID_EQUITY"


# ─────────────────────────────────────────────────────────────────────────────
# check_stop_distance
# ─────────────────────────────────────────────────────────────────────────────


def test_check_stop_distance_passes_with_positive_stop():
    assert check_stop_distance(10.0) is None


def test_check_stop_distance_passes_with_small_positive():
    assert check_stop_distance(0.001) is None


def test_check_stop_distance_fails_with_zero():
    result = check_stop_distance(0.0)
    assert result is not None
    assert result[0] == "INVALID_STOP_DISTANCE"


def test_check_stop_distance_fails_with_negative():
    result = check_stop_distance(-5.0)
    assert result is not None
    assert result[0] == "INVALID_STOP_DISTANCE"


def test_check_stop_distance_reason_text_is_readable():
    _, text = check_stop_distance(-5.0)
    assert len(text) > 10  # non-trivial message


# ─────────────────────────────────────────────────────────────────────────────
# check_max_positions
# ─────────────────────────────────────────────────────────────────────────────


def test_check_max_positions_passes_when_below_limit():
    state = _state(open_positions_count=2)
    profile = _profile(max_concurrent_positions=3)
    assert check_max_positions(state, profile) is None


def test_check_max_positions_passes_when_zero_positions():
    state = _state(open_positions_count=0)
    profile = _profile(max_concurrent_positions=3)
    assert check_max_positions(state, profile) is None


def test_check_max_positions_fails_at_limit():
    """Exactly at the limit (not below) is rejected."""
    state = _state(open_positions_count=3)
    profile = _profile(max_concurrent_positions=3)
    result = check_max_positions(state, profile)
    assert result is not None
    assert result[0] == "MAX_POSITIONS_REACHED"


def test_check_max_positions_fails_above_limit():
    state = _state(open_positions_count=5)
    profile = _profile(max_concurrent_positions=3)
    result = check_max_positions(state, profile)
    assert result is not None
    assert result[0] == "MAX_POSITIONS_REACHED"


def test_check_max_positions_reason_text_mentions_counts():
    state = _state(open_positions_count=3)
    profile = _profile(max_concurrent_positions=3)
    _, text = check_max_positions(state, profile)
    assert "3" in text


# ─────────────────────────────────────────────────────────────────────────────
# check_max_strategy_positions
# ─────────────────────────────────────────────────────────────────────────────


def test_check_max_strategy_positions_passes_when_no_positions():
    state = _state(open_positions_by_strategy={})
    profile = _profile(max_positions_per_strategy=2)
    assert check_max_strategy_positions(state, profile, "or_london_v1") is None


def test_check_max_strategy_positions_passes_below_limit():
    state = _state(open_positions_by_strategy={"or_london_v1": 1})
    profile = _profile(max_positions_per_strategy=2)
    assert check_max_strategy_positions(state, profile, "or_london_v1") is None


def test_check_max_strategy_positions_fails_at_limit():
    state = _state(open_positions_by_strategy={"or_london_v1": 2})
    profile = _profile(max_positions_per_strategy=2)
    result = check_max_strategy_positions(state, profile, "or_london_v1")
    assert result is not None
    assert result[0] == "MAX_STRATEGY_POSITIONS_REACHED"


def test_check_max_strategy_positions_other_strategy_not_affected():
    """Limit on strategy A does not block strategy B."""
    state = _state(open_positions_by_strategy={"or_london_v1": 2})
    profile = _profile(max_positions_per_strategy=2)
    assert check_max_strategy_positions(state, profile, "vwap_fade_v1") is None


# ─────────────────────────────────────────────────────────────────────────────
# check_same_direction
# ─────────────────────────────────────────────────────────────────────────────


def test_check_same_direction_disabled_when_allow_multiple():
    """allow_same_direction_multiple=True disables the rule entirely."""
    state = _state(open_positions_by_direction={"LONG": 3})
    profile = _profile(allow_same_direction_multiple=True)
    assert check_same_direction(state, profile, TradeDirection.LONG) is None


def test_check_same_direction_passes_when_no_existing_position():
    state = _state(open_positions_by_direction={})
    profile = _profile(allow_same_direction_multiple=False)
    assert check_same_direction(state, profile, TradeDirection.LONG) is None


def test_check_same_direction_passes_for_opposite_direction():
    """Having a LONG open does not block a SHORT."""
    state = _state(open_positions_by_direction={"LONG": 1})
    profile = _profile(allow_same_direction_multiple=False)
    assert check_same_direction(state, profile, TradeDirection.SHORT) is None


def test_check_same_direction_fails_long_already_open():
    state = _state(open_positions_by_direction={"LONG": 1})
    profile = _profile(allow_same_direction_multiple=False)
    result = check_same_direction(state, profile, TradeDirection.LONG)
    assert result is not None
    assert result[0] == "SAME_DIRECTION_NOT_ALLOWED"


def test_check_same_direction_fails_short_already_open():
    state = _state(open_positions_by_direction={"SHORT": 1})
    profile = _profile(allow_same_direction_multiple=False)
    result = check_same_direction(state, profile, TradeDirection.SHORT)
    assert result is not None
    assert result[0] == "SAME_DIRECTION_NOT_ALLOWED"


def test_check_same_direction_reason_text_mentions_direction():
    state = _state(open_positions_by_direction={"LONG": 1})
    profile = _profile(allow_same_direction_multiple=False)
    _, text = check_same_direction(state, profile, TradeDirection.LONG)
    assert "long" in text.lower() or "buy" in text.lower()


# ─────────────────────────────────────────────────────────────────────────────
# check_daily_risk
# ─────────────────────────────────────────────────────────────────────────────


def test_check_daily_risk_passes_when_budget_available():
    """0% used + 1% new trade = 1% < 2% limit."""
    state = _state(daily_risk_used_pct=0.0)
    profile = _profile(max_daily_risk_pct=2.0)
    assert check_daily_risk(state, profile, new_risk_pct=1.0) is None


def test_check_daily_risk_passes_exactly_at_budget():
    """1% used + 1% new = 2% == 2% limit — allowed (not strictly over)."""
    state = _state(daily_risk_used_pct=1.0)
    profile = _profile(max_daily_risk_pct=2.0)
    assert check_daily_risk(state, profile, new_risk_pct=1.0) is None


def test_check_daily_risk_fails_when_would_exceed():
    """1.5% used + 1% new = 2.5% > 2% limit."""
    state = _state(daily_risk_used_pct=1.5)
    profile = _profile(max_daily_risk_pct=2.0)
    result = check_daily_risk(state, profile, new_risk_pct=1.0)
    assert result is not None
    assert result[0] == "MAX_DAILY_RISK_REACHED"


def test_check_daily_risk_fails_when_budget_already_full():
    """2% already used, any new trade is blocked."""
    state = _state(daily_risk_used_pct=2.0)
    profile = _profile(max_daily_risk_pct=2.0)
    result = check_daily_risk(state, profile, new_risk_pct=0.5)
    assert result is not None
    assert result[0] == "MAX_DAILY_RISK_REACHED"


def test_check_daily_risk_reason_text_mentions_percentages():
    state = _state(daily_risk_used_pct=1.5)
    profile = _profile(max_daily_risk_pct=2.0)
    _, text = check_daily_risk(state, profile, new_risk_pct=1.0)
    assert "%" in text
