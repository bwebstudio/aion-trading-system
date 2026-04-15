"""
tests/unit/test_risk_models.py
───────────────────────────────
Unit tests for aion.risk.models.

Tests verify:
  RiskProfile:
    - default field values
    - is frozen (immutable)
    - rejects non-positive equity
    - rejects out-of-range percentage values
    - rejects position limits below 1

  PortfolioState:
    - default fields produce a zero/empty state
    - is frozen
    - dict fields are accessible and correct

  RiskDecision:
    - approved decision has position_size and risk_amount populated
    - rejected decision has None for sizing fields
    - is frozen
    - reason_code and reason_text are always present
"""

from __future__ import annotations

import pytest

from aion.risk.models import PortfolioState, RiskDecision, RiskProfile


# ─────────────────────────────────────────────────────────────────────────────
# RiskProfile
# ─────────────────────────────────────────────────────────────────────────────


def test_risk_profile_default_values():
    profile = RiskProfile(account_equity=10_000.0)
    assert profile.max_risk_per_trade_pct == pytest.approx(1.0)
    assert profile.max_daily_risk_pct == pytest.approx(2.0)
    assert profile.max_concurrent_positions == 3
    assert profile.max_positions_per_strategy == 2
    assert profile.allow_same_direction_multiple is False


def test_risk_profile_custom_values():
    profile = RiskProfile(
        account_equity=50_000.0,
        max_risk_per_trade_pct=0.5,
        max_daily_risk_pct=1.5,
        max_concurrent_positions=5,
        max_positions_per_strategy=3,
        allow_same_direction_multiple=True,
    )
    assert profile.account_equity == pytest.approx(50_000.0)
    assert profile.max_risk_per_trade_pct == pytest.approx(0.5)
    assert profile.allow_same_direction_multiple is True


def test_risk_profile_is_frozen():
    profile = RiskProfile(account_equity=10_000.0)
    with pytest.raises(Exception):
        profile.account_equity = 20_000.0  # type: ignore[misc]


def test_risk_profile_rejects_zero_equity():
    with pytest.raises(Exception):
        RiskProfile(account_equity=0.0)


def test_risk_profile_rejects_negative_equity():
    with pytest.raises(Exception):
        RiskProfile(account_equity=-1000.0)


def test_risk_profile_rejects_zero_risk_pct():
    with pytest.raises(Exception):
        RiskProfile(account_equity=10_000.0, max_risk_per_trade_pct=0.0)


def test_risk_profile_rejects_risk_pct_above_100():
    with pytest.raises(Exception):
        RiskProfile(account_equity=10_000.0, max_risk_per_trade_pct=101.0)


def test_risk_profile_rejects_zero_max_concurrent():
    with pytest.raises(Exception):
        RiskProfile(account_equity=10_000.0, max_concurrent_positions=0)


def test_risk_profile_rejects_zero_max_per_strategy():
    with pytest.raises(Exception):
        RiskProfile(account_equity=10_000.0, max_positions_per_strategy=0)


# ─────────────────────────────────────────────────────────────────────────────
# PortfolioState
# ─────────────────────────────────────────────────────────────────────────────


def test_portfolio_state_default_is_empty():
    state = PortfolioState()
    assert state.open_positions_count == 0
    assert state.open_positions_by_strategy == {}
    assert state.open_positions_by_direction == {}
    assert state.daily_realized_pnl == pytest.approx(0.0)
    assert state.daily_unrealized_pnl == pytest.approx(0.0)
    assert state.daily_risk_used_pct == pytest.approx(0.0)


def test_portfolio_state_is_frozen():
    state = PortfolioState()
    with pytest.raises(Exception):
        state.open_positions_count = 5  # type: ignore[misc]


def test_portfolio_state_strategy_dict_accessible():
    state = PortfolioState(
        open_positions_count=2,
        open_positions_by_strategy={"or_london_v1": 2},
    )
    assert state.open_positions_by_strategy["or_london_v1"] == 2
    assert state.open_positions_by_strategy.get("vwap_fade_v1", 0) == 0


def test_portfolio_state_direction_dict_accessible():
    state = PortfolioState(
        open_positions_by_direction={"LONG": 1, "SHORT": 0},
    )
    assert state.open_positions_by_direction["LONG"] == 1


def test_portfolio_state_with_daily_pnl():
    state = PortfolioState(
        daily_realized_pnl=250.0,
        daily_unrealized_pnl=-50.0,
        daily_risk_used_pct=1.5,
    )
    assert state.daily_realized_pnl == pytest.approx(250.0)
    assert state.daily_unrealized_pnl == pytest.approx(-50.0)
    assert state.daily_risk_used_pct == pytest.approx(1.5)


# ─────────────────────────────────────────────────────────────────────────────
# RiskDecision
# ─────────────────────────────────────────────────────────────────────────────


def _approved_decision() -> RiskDecision:
    return RiskDecision(
        approved=True,
        reason_code="APPROVED",
        reason_text="All checks passed.",
        candidate_setup_id="setup_abc",
        strategy_id="or_london_v1",
        position_size=1.0,
        risk_amount=100.0,
        stop_distance_points=10.0,
        target_distance_points=20.0,
    )


def _rejected_decision(reason_code: str = "MAX_POSITIONS_REACHED") -> RiskDecision:
    return RiskDecision(
        approved=False,
        reason_code=reason_code,
        reason_text="Too many open positions.",
        candidate_setup_id="setup_abc",
        strategy_id="or_london_v1",
        stop_distance_points=10.0,
    )


def test_approved_decision_has_sizing():
    d = _approved_decision()
    assert d.approved is True
    assert d.position_size == pytest.approx(1.0)
    assert d.risk_amount == pytest.approx(100.0)


def test_rejected_decision_has_no_sizing():
    d = _rejected_decision()
    assert d.approved is False
    assert d.position_size is None
    assert d.risk_amount is None


def test_decision_is_frozen():
    d = _approved_decision()
    with pytest.raises(Exception):
        d.approved = False  # type: ignore[misc]


def test_decision_reason_code_always_present():
    d = _rejected_decision("MAX_DAILY_RISK_REACHED")
    assert d.reason_code == "MAX_DAILY_RISK_REACHED"


def test_decision_reason_text_always_present():
    d = _rejected_decision()
    assert d.reason_text != ""


def test_decision_target_distance_optional():
    d = _rejected_decision()
    assert d.target_distance_points is None


def test_decision_stop_distance_carried_on_rejection():
    d = _rejected_decision()
    assert d.stop_distance_points == pytest.approx(10.0)


def test_decision_stop_and_target_carried_on_approval():
    d = _approved_decision()
    assert d.stop_distance_points == pytest.approx(10.0)
    assert d.target_distance_points == pytest.approx(20.0)
