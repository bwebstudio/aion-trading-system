"""
tests/unit/test_risk_allocator.py
───────────────────────────────────
Integration tests for aion.risk.allocator.evaluate().

Tests verify end-to-end behaviour of the allocator:
  - full approval: all rules pass, RiskDecision has position_size + risk_amount
  - sizing is correct on approval
  - rejected by MAX_POSITIONS_REACHED
  - rejected by MAX_STRATEGY_POSITIONS_REACHED
  - rejected by MAX_DAILY_RISK_REACHED
  - rejected by INVALID_STOP_DISTANCE
  - rejected by SAME_DIRECTION_NOT_ALLOWED
  - output is always a RiskDecision instance
  - output is frozen (immutable)
  - candidate_setup_id and strategy_id carried through correctly
  - target_distance_points carried through on both approval and rejection
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aion.core.enums import AssetClass, DataSource, SessionName, Timeframe, TradeDirection
from aion.core.ids import new_snapshot_id
from aion.core.models import InstrumentSpec
from aion.risk.allocator import evaluate
from aion.risk.models import PortfolioState, RiskDecision, RiskProfile
from aion.strategies.models import CandidateSetup

_UTC = timezone.utc
_TS = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_UTC)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _instrument() -> InstrumentSpec:
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


def _candidate(
    strategy_id: str = "or_london_v1",
    direction: TradeDirection = TradeDirection.LONG,
) -> CandidateSetup:
    return CandidateSetup(
        strategy_id=strategy_id,
        strategy_version="1.0.0",
        symbol="EURUSD",
        timestamp_utc=_TS,
        direction=direction,
        entry_reference=1.1020,
        range_high=1.1020,
        range_low=1.1000,
        range_size_pips=20.0,
        session_name=SessionName.LONDON.value,
        quality_score=1.0,
        atr_14=0.00015,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Return type and structure
# ─────────────────────────────────────────────────────────────────────────────


def test_evaluate_returns_risk_decision():
    result = evaluate(_candidate(), _profile(), _state(), _instrument(), 10.0)
    assert isinstance(result, RiskDecision)


def test_evaluate_result_is_frozen():
    result = evaluate(_candidate(), _profile(), _state(), _instrument(), 10.0)
    with pytest.raises(Exception):
        result.approved = False  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# Approval
# ─────────────────────────────────────────────────────────────────────────────


def test_full_approval_with_clean_state():
    """Empty portfolio + valid profile + 10-pip stop -> approved."""
    result = evaluate(_candidate(), _profile(), _state(), _instrument(), 10.0)
    assert result.approved is True
    assert result.reason_code == "APPROVED"


def test_approved_decision_has_position_size():
    result = evaluate(_candidate(), _profile(), _state(), _instrument(), 10.0)
    assert result.position_size is not None
    assert result.position_size > 0


def test_approved_decision_has_risk_amount():
    result = evaluate(_candidate(), _profile(), _state(), _instrument(), 10.0)
    assert result.risk_amount is not None
    assert result.risk_amount == pytest.approx(100.0)  # 10 000 * 1%


def test_approved_position_size_correct():
    """$10 000 * 1% / (10 pips * $10) = 1.00 lot."""
    result = evaluate(_candidate(), _profile(), _state(), _instrument(), 10.0)
    assert result.position_size == pytest.approx(1.0)


def test_approved_position_size_scales_with_stop():
    """Wider stop -> smaller position."""
    r10 = evaluate(_candidate(), _profile(), _state(), _instrument(), 10.0)
    r20 = evaluate(_candidate(), _profile(), _state(), _instrument(), 20.0)
    assert r10.position_size > r20.position_size


def test_approved_carries_candidate_id():
    candidate = _candidate()
    result = evaluate(candidate, _profile(), _state(), _instrument(), 10.0)
    assert result.candidate_setup_id == candidate.setup_id


def test_approved_carries_strategy_id():
    result = evaluate(_candidate("my_strategy"), _profile(), _state(), _instrument(), 10.0)
    assert result.strategy_id == "my_strategy"


def test_approved_carries_target_distance():
    result = evaluate(_candidate(), _profile(), _state(), _instrument(), 10.0, 20.0)
    assert result.target_distance_points == pytest.approx(20.0)


# ─────────────────────────────────────────────────────────────────────────────
# Rejection — MAX_POSITIONS_REACHED
# ─────────────────────────────────────────────────────────────────────────────


def test_rejected_when_max_positions_reached():
    state = _state(open_positions_count=3)
    profile = _profile(max_concurrent_positions=3)
    result = evaluate(_candidate(), profile, state, _instrument(), 10.0)
    assert result.approved is False
    assert result.reason_code == "MAX_POSITIONS_REACHED"


def test_rejected_max_positions_has_no_sizing():
    state = _state(open_positions_count=3)
    profile = _profile(max_concurrent_positions=3)
    result = evaluate(_candidate(), profile, state, _instrument(), 10.0)
    assert result.position_size is None
    assert result.risk_amount is None


# ─────────────────────────────────────────────────────────────────────────────
# Rejection — MAX_STRATEGY_POSITIONS_REACHED
# ─────────────────────────────────────────────────────────────────────────────


def test_rejected_when_max_strategy_positions_reached():
    state = _state(
        open_positions_count=1,
        open_positions_by_strategy={"or_london_v1": 2},
    )
    profile = _profile(max_positions_per_strategy=2)
    result = evaluate(_candidate("or_london_v1"), profile, state, _instrument(), 10.0)
    assert result.approved is False
    assert result.reason_code == "MAX_STRATEGY_POSITIONS_REACHED"


def test_different_strategy_not_blocked_by_other_strategy_limit():
    """Hitting the limit for strategy A does not block strategy B."""
    state = _state(
        open_positions_count=1,
        open_positions_by_strategy={"or_london_v1": 2},
    )
    profile = _profile(max_positions_per_strategy=2)
    result = evaluate(_candidate("vwap_fade_v1"), profile, state, _instrument(), 10.0)
    # vwap_fade_v1 has 0 positions → should pass this rule (may fail others)
    assert result.reason_code != "MAX_STRATEGY_POSITIONS_REACHED"


# ─────────────────────────────────────────────────────────────────────────────
# Rejection — MAX_DAILY_RISK_REACHED
# ─────────────────────────────────────────────────────────────────────────────


def test_rejected_when_daily_risk_exhausted():
    """1.5% already used + 1% new trade = 2.5% > 2% daily limit."""
    state = _state(daily_risk_used_pct=1.5)
    profile = _profile(max_risk_per_trade_pct=1.0, max_daily_risk_pct=2.0)
    result = evaluate(_candidate(), profile, state, _instrument(), 10.0)
    assert result.approved is False
    assert result.reason_code == "MAX_DAILY_RISK_REACHED"


def test_approved_when_daily_risk_not_exhausted():
    """0.5% used + 1% new = 1.5% < 2% limit -> approved."""
    state = _state(daily_risk_used_pct=0.5)
    profile = _profile(max_risk_per_trade_pct=1.0, max_daily_risk_pct=2.0)
    result = evaluate(_candidate(), profile, state, _instrument(), 10.0)
    assert result.approved is True


# ─────────────────────────────────────────────────────────────────────────────
# Rejection — INVALID_STOP_DISTANCE
# ─────────────────────────────────────────────────────────────────────────────


def test_rejected_when_stop_is_zero():
    result = evaluate(_candidate(), _profile(), _state(), _instrument(), 0.0)
    assert result.approved is False
    assert result.reason_code == "INVALID_STOP_DISTANCE"


def test_rejected_when_stop_is_negative():
    result = evaluate(_candidate(), _profile(), _state(), _instrument(), -5.0)
    assert result.approved is False
    assert result.reason_code == "INVALID_STOP_DISTANCE"


# ─────────────────────────────────────────────────────────────────────────────
# Rejection — SAME_DIRECTION_NOT_ALLOWED
# ─────────────────────────────────────────────────────────────────────────────


def test_rejected_when_same_direction_not_allowed():
    state = _state(open_positions_by_direction={"LONG": 1})
    profile = _profile(allow_same_direction_multiple=False)
    result = evaluate(_candidate(direction=TradeDirection.LONG), profile, state, _instrument(), 10.0)
    assert result.approved is False
    assert result.reason_code == "SAME_DIRECTION_NOT_ALLOWED"


def test_approved_when_same_direction_allowed():
    state = _state(open_positions_by_direction={"LONG": 1})
    profile = _profile(allow_same_direction_multiple=True)
    result = evaluate(_candidate(direction=TradeDirection.LONG), profile, state, _instrument(), 10.0)
    assert result.reason_code != "SAME_DIRECTION_NOT_ALLOWED"


def test_opposite_direction_not_blocked():
    """Having a LONG open does not block a SHORT (even with allow_same=False)."""
    state = _state(open_positions_by_direction={"LONG": 1})
    profile = _profile(allow_same_direction_multiple=False)
    result = evaluate(_candidate(direction=TradeDirection.SHORT), profile, state, _instrument(), 10.0)
    assert result.reason_code != "SAME_DIRECTION_NOT_ALLOWED"


# ─────────────────────────────────────────────────────────────────────────────
# Context always carried through
# ─────────────────────────────────────────────────────────────────────────────


def test_stop_distance_carried_on_rejection():
    state = _state(open_positions_count=3)
    result = evaluate(_candidate(), _profile(max_concurrent_positions=3), state, _instrument(), 15.0)
    assert result.stop_distance_points == pytest.approx(15.0)


def test_target_distance_carried_on_rejection():
    state = _state(open_positions_count=3)
    result = evaluate(_candidate(), _profile(max_concurrent_positions=3),
                      state, _instrument(), 15.0, 30.0)
    assert result.target_distance_points == pytest.approx(30.0)


def test_target_distance_none_when_not_provided():
    result = evaluate(_candidate(), _profile(), _state(), _instrument(), 10.0)
    assert result.target_distance_points is None


# ─────────────────────────────────────────────────────────────────────────────
# Output consistency
# ─────────────────────────────────────────────────────────────────────────────


def test_same_inputs_produce_same_output():
    """Allocator is deterministic."""
    candidate = _candidate()
    profile = _profile()
    state = _state()
    instrument = _instrument()
    r1 = evaluate(candidate, profile, state, instrument, 10.0)
    r2 = evaluate(candidate, profile, state, instrument, 10.0)
    assert r1.approved == r2.approved
    assert r1.position_size == r2.position_size
    assert r1.risk_amount == r2.risk_amount
