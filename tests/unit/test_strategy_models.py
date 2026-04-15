"""
tests/unit/test_strategy_models.py
────────────────────────────────────
Unit tests for aion.strategies.models.

Tests verify:
  - OpeningRangeDefinition field validation (positive ranges, max > min)
  - pip/price conversion helpers
  - CandidateSetup derived properties (range_size, is_long, is_short)
  - setup_id is unique per instance
  - StrategyEvaluationResult outcome helpers (has_setup, is_no_trade, etc.)
  - CANDIDATE result has candidate, no no_trade
  - NO_TRADE result has no_trade, no candidate
  - INSUFFICIENT_DATA result has neither
  - All models are frozen (immutable)
  - All models serialise to JSON without error
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aion.core.enums import TradeDirection
from aion.strategies.models import (
    CandidateSetup,
    NoTradeDecision,
    OpeningRangeDefinition,
    OpeningRangeState,
    StrategyEvaluationResult,
    StrategyOutcome,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

_TS = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


def make_defn(
    min_range_pips: float = 5.0,
    max_range_pips: float = 40.0,
    session_name: str = "LONDON",
    direction_bias: TradeDirection | None = None,
    require_completed_range: bool = True,
) -> OpeningRangeDefinition:
    return OpeningRangeDefinition(
        strategy_id="or_london_v1",
        session_name=session_name,
        min_range_pips=min_range_pips,
        max_range_pips=max_range_pips,
        direction_bias=direction_bias,
        require_completed_range=require_completed_range,
    )


def make_candidate(
    direction: TradeDirection = TradeDirection.LONG,
    range_high: float = 1.1050,
    range_low: float = 1.1000,
    range_size_pips: float = 50.0,
) -> CandidateSetup:
    return CandidateSetup(
        strategy_id="or_london_v1",
        strategy_version="1.0.0",
        symbol="EURUSD",
        timestamp_utc=_TS,
        direction=direction,
        entry_reference=range_high if direction == TradeDirection.LONG else range_low,
        range_high=range_high,
        range_low=range_low,
        range_size_pips=range_size_pips,
        session_name="LONDON",
        quality_score=0.98,
        atr_14=0.00015,
    )


def make_no_trade(reason_code: str = "RANGE_TOO_TIGHT") -> NoTradeDecision:
    return NoTradeDecision(
        strategy_id="or_london_v1",
        symbol="EURUSD",
        timestamp_utc=_TS,
        reason_code=reason_code,
        reason_detail="Range was 3.0 pips; minimum is 5.0 pips.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# OpeningRangeDefinition validation
# ─────────────────────────────────────────────────────────────────────────────


def test_valid_definition_builds():
    defn = make_defn()
    assert defn.strategy_id == "or_london_v1"
    assert defn.session_name == "LONDON"


def test_negative_min_range_raises():
    with pytest.raises(Exception):
        OpeningRangeDefinition(
            strategy_id="x",
            session_name="LONDON",
            min_range_pips=-1.0,
            max_range_pips=40.0,
        )


def test_zero_min_range_raises():
    with pytest.raises(Exception):
        OpeningRangeDefinition(
            strategy_id="x",
            session_name="LONDON",
            min_range_pips=0.0,
            max_range_pips=40.0,
        )


def test_max_range_below_min_raises():
    with pytest.raises(Exception):
        OpeningRangeDefinition(
            strategy_id="x",
            session_name="LONDON",
            min_range_pips=20.0,
            max_range_pips=10.0,
        )


def test_max_range_equal_to_min_raises():
    with pytest.raises(Exception):
        OpeningRangeDefinition(
            strategy_id="x",
            session_name="LONDON",
            min_range_pips=10.0,
            max_range_pips=10.0,
        )


def test_definition_is_frozen():
    defn = make_defn()
    with pytest.raises(Exception):
        defn.session_name = "NEW_YORK"  # type: ignore[misc]


def test_definition_default_version():
    defn = make_defn()
    assert defn.version == "1.0.0"


def test_definition_default_pip_multiplier():
    defn = make_defn()
    assert defn.pip_multiplier == 10.0


def test_direction_bias_none_by_default():
    defn = make_defn()
    assert defn.direction_bias is None


def test_require_completed_range_true_by_default():
    defn = make_defn()
    assert defn.require_completed_range is True


# ─────────────────────────────────────────────────────────────────────────────
# Pip / price conversion
# ─────────────────────────────────────────────────────────────────────────────


def test_pips_to_price_eurusd():
    """1 pip EURUSD (tick_size=0.00001, pip_multiplier=10) = 0.0001."""
    defn = make_defn()
    tick_size = 0.00001  # 5-decimal instrument
    price = defn.pips_to_price(pips=1.0, tick_size=tick_size)
    assert price == pytest.approx(0.0001)


def test_pips_to_price_10_pips():
    defn = make_defn()
    price = defn.pips_to_price(pips=10.0, tick_size=0.00001)
    assert price == pytest.approx(0.0010)


def test_price_to_pips_eurusd():
    defn = make_defn()
    pips = defn.price_to_pips(price_distance=0.0010, tick_size=0.00001)
    assert pips == pytest.approx(10.0)


def test_pip_round_trip():
    """pips_to_price and price_to_pips are inverse operations."""
    defn = make_defn()
    tick_size = 0.00001
    original_pips = 17.5
    price = defn.pips_to_price(original_pips, tick_size)
    recovered = defn.price_to_pips(price, tick_size)
    assert recovered == pytest.approx(original_pips)


# ─────────────────────────────────────────────────────────────────────────────
# CandidateSetup
# ─────────────────────────────────────────────────────────────────────────────


def test_candidate_setup_id_generated():
    c = make_candidate()
    assert c.setup_id.startswith("setup_")
    assert len(c.setup_id) > 6


def test_candidate_setup_ids_are_unique():
    c1 = make_candidate()
    c2 = make_candidate()
    assert c1.setup_id != c2.setup_id


def test_candidate_is_long_true():
    c = make_candidate(direction=TradeDirection.LONG)
    assert c.is_long is True
    assert c.is_short is False


def test_candidate_is_short_true():
    c = make_candidate(direction=TradeDirection.SHORT)
    assert c.is_short is True
    assert c.is_long is False


def test_candidate_range_size_property():
    c = make_candidate(range_high=1.1050, range_low=1.1000)
    assert c.range_size == pytest.approx(0.0050)


def test_candidate_entry_reference_is_range_high_for_long():
    c = make_candidate(direction=TradeDirection.LONG, range_high=1.1050)
    assert c.entry_reference == pytest.approx(1.1050)


def test_candidate_entry_reference_is_range_low_for_short():
    c = make_candidate(direction=TradeDirection.SHORT, range_low=1.1000)
    assert c.entry_reference == pytest.approx(1.1000)


def test_candidate_is_frozen():
    c = make_candidate()
    with pytest.raises(Exception):
        c.direction = TradeDirection.SHORT  # type: ignore[misc]


def test_candidate_serialises_to_json():
    c = make_candidate()
    json_str = c.model_dump_json()
    assert "setup_id" in json_str
    assert "EURUSD" in json_str


def test_candidate_atr_can_be_none():
    c = CandidateSetup(
        strategy_id="x",
        strategy_version="1.0.0",
        symbol="EURUSD",
        timestamp_utc=_TS,
        direction=TradeDirection.LONG,
        entry_reference=1.1050,
        range_high=1.1050,
        range_low=1.1000,
        range_size_pips=50.0,
        session_name="LONDON",
        quality_score=0.95,
        atr_14=None,
    )
    assert c.atr_14 is None


# ─────────────────────────────────────────────────────────────────────────────
# NoTradeDecision
# ─────────────────────────────────────────────────────────────────────────────


def test_no_trade_reason_code_preserved():
    nt = make_no_trade("RANGE_TOO_WIDE")
    assert nt.reason_code == "RANGE_TOO_WIDE"


def test_no_trade_is_frozen():
    nt = make_no_trade()
    with pytest.raises(Exception):
        nt.reason_code = "X"  # type: ignore[misc]


def test_no_trade_optional_fields_default_none():
    nt = make_no_trade()
    assert nt.or_high is None
    assert nt.or_low is None
    assert nt.or_state is None


def test_no_trade_with_or_levels():
    nt = NoTradeDecision(
        strategy_id="x",
        symbol="EURUSD",
        timestamp_utc=_TS,
        reason_code="RANGE_TOO_TIGHT",
        reason_detail="too tight",
        or_high=1.1050,
        or_low=1.1045,
        or_state=OpeningRangeState.COMPLETED,
    )
    assert nt.or_high == pytest.approx(1.1050)
    assert nt.or_state == OpeningRangeState.COMPLETED


# ─────────────────────────────────────────────────────────────────────────────
# StrategyEvaluationResult
# ─────────────────────────────────────────────────────────────────────────────


def _make_candidate_result() -> StrategyEvaluationResult:
    return StrategyEvaluationResult(
        outcome=StrategyOutcome.CANDIDATE,
        strategy_id="or_london_v1",
        symbol="EURUSD",
        timestamp_utc=_TS,
        candidate=make_candidate(),
    )


def _make_no_trade_result() -> StrategyEvaluationResult:
    return StrategyEvaluationResult(
        outcome=StrategyOutcome.NO_TRADE,
        strategy_id="or_london_v1",
        symbol="EURUSD",
        timestamp_utc=_TS,
        no_trade=make_no_trade(),
    )


def _make_insufficient_result() -> StrategyEvaluationResult:
    return StrategyEvaluationResult(
        outcome=StrategyOutcome.INSUFFICIENT_DATA,
        strategy_id="or_london_v1",
        symbol="EURUSD",
        timestamp_utc=_TS,
        reason_detail="Quality score too low.",
    )


def test_candidate_result_has_setup():
    r = _make_candidate_result()
    assert r.has_setup is True
    assert r.is_no_trade is False
    assert r.is_insufficient_data is False


def test_candidate_result_candidate_populated():
    r = _make_candidate_result()
    assert r.candidate is not None
    assert r.no_trade is None


def test_no_trade_result_is_no_trade():
    r = _make_no_trade_result()
    assert r.has_setup is False
    assert r.is_no_trade is True
    assert r.is_insufficient_data is False


def test_no_trade_result_no_trade_populated():
    r = _make_no_trade_result()
    assert r.no_trade is not None
    assert r.candidate is None


def test_insufficient_data_result():
    r = _make_insufficient_result()
    assert r.is_insufficient_data is True
    assert r.has_setup is False
    assert r.is_no_trade is False
    assert r.candidate is None
    assert r.no_trade is None


def test_insufficient_data_has_reason_detail():
    r = _make_insufficient_result()
    assert r.reason_detail is not None
    assert len(r.reason_detail) > 0


def test_result_is_frozen():
    r = _make_candidate_result()
    with pytest.raises(Exception):
        r.outcome = StrategyOutcome.NO_TRADE  # type: ignore[misc]


def test_result_serialises_to_json():
    r = _make_candidate_result()
    json_str = r.model_dump_json()
    assert "CANDIDATE" in json_str
    assert "EURUSD" in json_str


# ─────────────────────────────────────────────────────────────────────────────
# OpeningRangeState enum
# ─────────────────────────────────────────────────────────────────────────────


def test_or_state_values_are_strings():
    """All enum values are strings for JSON serialisation compatibility."""
    for state in OpeningRangeState:
        assert isinstance(state.value, str)


def test_or_outcome_values_are_strings():
    for outcome in StrategyOutcome:
        assert isinstance(outcome.value, str)
