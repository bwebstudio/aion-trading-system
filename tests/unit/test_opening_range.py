"""
tests/unit/test_opening_range.py
──────────────────────────────────
Unit tests for aion.strategies.opening_range.OpeningRangeEngine.

Tests verify:
  - CANDIDATE returned when all conditions are met (LONG and SHORT biases)
  - NO_TRADE: wrong session
  - NO_TRADE: OR not completed (require_completed_range=True)
  - NO_TRADE: OR active (require_completed_range=True)
  - NO_TRADE: OR levels are None
  - NO_TRADE: range too tight
  - NO_TRADE: range too wide
  - INSUFFICIENT_DATA: low quality score
  - Direction bias: LONG bias → LONG candidate
  - Direction bias: SHORT bias → SHORT candidate
  - require_completed_range=False allows ACTIVE OR
  - CandidateSetup fields populated correctly
  - NoTradeDecision includes or_high / or_low / or_state when available
  - Engine strategy_id and version match definition
  - OFF_HOURS → NO_TRADE with NOT_IN_TARGET_SESSION
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from aion.core.constants import (
    ATR_PERIOD,
    FEATURE_SET_VERSION,
    MIN_QUALITY_SCORE,
    SNAPSHOT_VERSION,
)
from aion.core.enums import AssetClass, DataSource, SessionName, Timeframe, TradeDirection
from aion.core.ids import new_snapshot_id
from aion.core.models import (
    DataQualityReport,
    FeatureVector,
    InstrumentSpec,
    MarketBar,
    MarketSnapshot,
    SessionContext,
)
from aion.strategies.models import (
    OpeningRangeDefinition,
    OpeningRangeState,
    StrategyOutcome,
)
from aion.strategies.opening_range import OpeningRangeEngine


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot builder helpers
# ─────────────────────────────────────────────────────────────────────────────

_UTC = timezone.utc
_BASE_TS = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_UTC)


def make_instrument() -> InstrumentSpec:
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


def make_bar(ts: datetime) -> MarketBar:
    return MarketBar(
        symbol="EURUSD",
        timestamp_utc=ts,
        timestamp_market=ts,
        timeframe=Timeframe.M1,
        open=1.1000,
        high=1.1010,
        low=1.0990,
        close=1.1005,
        tick_volume=100.0,
        real_volume=0.0,
        spread=2.0,
        source=DataSource.CSV,
    )


def make_session(
    session_name: SessionName = SessionName.LONDON,
    opening_range_active: bool = False,
    opening_range_completed: bool = True,
    is_session_open: bool = True,
    ts: datetime = _BASE_TS,
) -> SessionContext:
    session_open = ts.replace(hour=8, minute=0, second=0)
    session_close = ts.replace(hour=16, minute=30, second=0)
    is_london = session_name in (SessionName.LONDON, SessionName.OVERLAP_LONDON_NY)
    is_ny = session_name in (SessionName.NEW_YORK, SessionName.OVERLAP_LONDON_NY)
    is_asia = session_name == SessionName.ASIA
    return SessionContext(
        trading_day=ts.date(),
        broker_time=ts,
        market_time=ts,
        local_time=ts,
        is_asia=is_asia,
        is_london=is_london,
        is_new_york=is_ny,
        is_session_open_window=is_session_open,
        opening_range_active=opening_range_active,
        opening_range_completed=opening_range_completed,
        session_name=session_name,
        session_open_utc=session_open if is_session_open else None,
        session_close_utc=session_close if is_session_open else None,
    )


def make_feature_vector(
    or_high: float | None = 1.1020,
    or_low: float | None = 1.1000,
    quality_score: float = 1.0,
    atr_14: float | None = 0.00015,
    ts: datetime = _BASE_TS,
) -> FeatureVector:
    return FeatureVector(
        symbol="EURUSD",
        timestamp_utc=ts,
        timeframe=Timeframe.M1,
        atr_14=atr_14,
        rolling_range_10=0.001,
        rolling_range_20=0.002,
        volatility_percentile_20=0.5,
        session_high=1.1060,
        session_low=1.0990,
        opening_range_high=or_high,
        opening_range_low=or_low,
        vwap_session=1.1025,
        spread_mean_20=2.0,
        spread_zscore_20=0.0,
        return_1=0.0001,
        return_5=0.0005,
        candle_body=0.0005,
        upper_wick=0.0005,
        lower_wick=0.001,
        distance_to_session_high=-0.0055,
        distance_to_session_low=0.0015,
        feature_set_version=FEATURE_SET_VERSION,
    )


def make_quality_report(quality_score: float = 1.0) -> DataQualityReport:
    return DataQualityReport(
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        rows_checked=100,
        missing_bars=0,
        duplicate_timestamps=0,
        out_of_order_rows=0,
        stale_bars=0,
        spike_bars=0,
        null_rows=0,
        quality_score=quality_score,
        warnings=[],
    )


def make_snapshot(
    session: SessionContext | None = None,
    feature_vector: FeatureVector | None = None,
    quality_score: float = 1.0,
    ts: datetime = _BASE_TS,
) -> MarketSnapshot:
    bar = make_bar(ts)
    sess = session or make_session(ts=ts)
    fv = feature_vector or make_feature_vector(ts=ts)
    qr = make_quality_report(quality_score)
    return MarketSnapshot(
        snapshot_id=new_snapshot_id(),
        symbol="EURUSD",
        timestamp_utc=ts,
        base_timeframe=Timeframe.M1,
        instrument=make_instrument(),
        session_context=sess,
        latest_bar=bar,
        bars_m1=[bar],
        bars_m5=[],
        bars_m15=[],
        feature_vector=fv,
        quality_report=qr,
        snapshot_version=SNAPSHOT_VERSION,
    )


def make_engine(
    min_range_pips: float = 5.0,
    max_range_pips: float = 40.0,
    session_name: str = "LONDON",
    direction_bias: TradeDirection | None = None,
    require_completed_range: bool = True,
    min_quality: float = MIN_QUALITY_SCORE,
    max_retest_penetration_points: float | None = None,
) -> OpeningRangeEngine:
    defn = OpeningRangeDefinition(
        strategy_id="or_london_v1",
        session_name=session_name,
        min_range_pips=min_range_pips,
        max_range_pips=max_range_pips,
        direction_bias=direction_bias,
        require_completed_range=require_completed_range,
        max_retest_penetration_points=max_retest_penetration_points,
    )
    return OpeningRangeEngine(defn, min_quality_score=min_quality)


def make_snap_for_retest(
    close: float,
    or_high: float = 1.1020,
    or_low: float = 1.1000,
) -> MarketSnapshot:
    """Snapshot where the bar's close is set explicitly for retest testing.

    OR range is 20 pips (or_high=1.1020, or_low=1.1000).
    Penetration for LONG  = max(0, or_high - close) converted to pips.
    Penetration for SHORT = max(0, close - or_low)  converted to pips.

    For EURUSD (tick_size=0.00001, pip_multiplier=10):
      close=1.1015 → LONG penetration = 5 pips, SHORT penetration = 15 pips
      close=1.1005 → LONG penetration = 15 pips, SHORT penetration = 5 pips
      close=1.1025 → LONG penetration = 0 (above OR high, no retest)
    """
    ts = _BASE_TS
    bar = MarketBar(
        symbol="EURUSD",
        timestamp_utc=ts,
        timestamp_market=ts,
        timeframe=Timeframe.M1,
        open=close,
        high=round(close + 0.0005, 5),
        low=round(close - 0.0005, 5),
        close=close,
        tick_volume=100.0,
        real_volume=0.0,
        spread=2.0,
        source=DataSource.CSV,
    )
    sess = make_session(ts=ts)
    fv = make_feature_vector(or_high=or_high, or_low=or_low, ts=ts)
    qr = make_quality_report()
    return MarketSnapshot(
        snapshot_id=new_snapshot_id(),
        symbol="EURUSD",
        timestamp_utc=ts,
        base_timeframe=Timeframe.M1,
        instrument=make_instrument(),
        session_context=sess,
        latest_bar=bar,
        bars_m1=[bar],
        bars_m5=[],
        bars_m15=[],
        feature_vector=fv,
        quality_report=qr,
        snapshot_version=SNAPSHOT_VERSION,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Engine identity
# ─────────────────────────────────────────────────────────────────────────────


def test_engine_strategy_id_matches_definition():
    engine = make_engine()
    assert engine.strategy_id == "or_london_v1"


def test_engine_version_matches_definition():
    engine = make_engine()
    assert engine.version == "1.0.0"


def test_engine_definition_accessible():
    engine = make_engine()
    assert engine.definition.session_name == "LONDON"


# ─────────────────────────────────────────────────────────────────────────────
# Happy path — CANDIDATE
# ─────────────────────────────────────────────────────────────────────────────


def test_candidate_returned_with_clean_snapshot():
    """All conditions met → CANDIDATE."""
    snap = make_snapshot(
        session=make_session(
            session_name=SessionName.LONDON,
            opening_range_completed=True,
            opening_range_active=False,
        ),
        feature_vector=make_feature_vector(or_high=1.1020, or_low=1.1000),
    )
    engine = make_engine()
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE


def test_candidate_setup_populated():
    snap = make_snapshot()
    engine = make_engine()
    result = engine.evaluate(snap)
    assert result.candidate is not None
    assert result.no_trade is None


def test_candidate_symbol_matches():
    snap = make_snapshot()
    engine = make_engine()
    result = engine.evaluate(snap)
    assert result.candidate.symbol == "EURUSD"


def test_candidate_strategy_id_matches():
    snap = make_snapshot()
    engine = make_engine()
    result = engine.evaluate(snap)
    assert result.candidate.strategy_id == "or_london_v1"


def test_candidate_direction_is_long_by_default():
    """With no direction_bias, engine defaults to LONG."""
    snap = make_snapshot()
    engine = make_engine()
    result = engine.evaluate(snap)
    assert result.candidate.direction == TradeDirection.LONG


def test_candidate_entry_reference_is_or_high_for_long():
    snap = make_snapshot(feature_vector=make_feature_vector(or_high=1.1020, or_low=1.1000))
    engine = make_engine(direction_bias=TradeDirection.LONG)
    result = engine.evaluate(snap)
    assert result.candidate.entry_reference == pytest.approx(1.1020)


def test_candidate_entry_reference_is_or_low_for_short():
    snap = make_snapshot(feature_vector=make_feature_vector(or_high=1.1020, or_low=1.1000))
    engine = make_engine(direction_bias=TradeDirection.SHORT)
    result = engine.evaluate(snap)
    assert result.candidate.entry_reference == pytest.approx(1.1000)


def test_candidate_range_high_and_low_set():
    # 30-pip range: well within 5–40 pip limits, no floating-point boundary risk
    fv = make_feature_vector(or_high=1.1030, or_low=1.1000)
    snap = make_snapshot(feature_vector=fv)
    engine = make_engine()
    result = engine.evaluate(snap)
    assert result.candidate.range_high == pytest.approx(1.1030)
    assert result.candidate.range_low == pytest.approx(1.1000)


def test_candidate_range_size_pips_computed():
    """or_high - or_low = 0.0040 = 40 pips for EURUSD (tick_size=0.00001)."""
    fv = make_feature_vector(or_high=1.1040, or_low=1.1000)
    snap = make_snapshot(feature_vector=fv)
    engine = make_engine(min_range_pips=5.0, max_range_pips=50.0)
    result = engine.evaluate(snap)
    assert result.candidate.range_size_pips == pytest.approx(40.0)


def test_candidate_quality_score_carried():
    snap = make_snapshot(quality_score=0.95)
    engine = make_engine()
    result = engine.evaluate(snap)
    assert result.candidate.quality_score == pytest.approx(0.95)


def test_candidate_atr_14_carried():
    fv = make_feature_vector(atr_14=0.00020)
    snap = make_snapshot(feature_vector=fv)
    engine = make_engine()
    result = engine.evaluate(snap)
    assert result.candidate.atr_14 == pytest.approx(0.00020)


def test_candidate_atr_14_none_when_unavailable():
    fv = make_feature_vector(atr_14=None)
    snap = make_snapshot(feature_vector=fv)
    engine = make_engine()
    result = engine.evaluate(snap)
    assert result.candidate.atr_14 is None


# ─────────────────────────────────────────────────────────────────────────────
# Direction bias
# ─────────────────────────────────────────────────────────────────────────────


def test_long_bias_produces_long_candidate():
    snap = make_snapshot()
    engine = make_engine(direction_bias=TradeDirection.LONG)
    result = engine.evaluate(snap)
    assert result.candidate.direction == TradeDirection.LONG


def test_short_bias_produces_short_candidate():
    snap = make_snapshot()
    engine = make_engine(direction_bias=TradeDirection.SHORT)
    result = engine.evaluate(snap)
    assert result.candidate.direction == TradeDirection.SHORT


# ─────────────────────────────────────────────────────────────────────────────
# NO_TRADE — wrong session
# ─────────────────────────────────────────────────────────────────────────────


def test_wrong_session_is_no_trade():
    sess = make_session(session_name=SessionName.ASIA)
    snap = make_snapshot(session=sess)
    engine = make_engine(session_name="LONDON")
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.NO_TRADE


def test_wrong_session_reason_code():
    sess = make_session(session_name=SessionName.ASIA)
    snap = make_snapshot(session=sess)
    engine = make_engine(session_name="LONDON")
    result = engine.evaluate(snap)
    assert result.no_trade.reason_code == "NOT_IN_TARGET_SESSION"


def test_off_hours_is_no_trade():
    sess = make_session(
        session_name=SessionName.OFF_HOURS,
        is_session_open=False,
        opening_range_active=False,
        opening_range_completed=False,
    )
    snap = make_snapshot(session=sess)
    engine = make_engine(session_name="LONDON")
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "NOT_IN_TARGET_SESSION"


def test_overlap_london_ny_counts_for_london_target():
    """OVERLAP_LONDON_NY should satisfy a LONDON-targeted engine."""
    sess = make_session(
        session_name=SessionName.OVERLAP_LONDON_NY,
        opening_range_completed=True,
    )
    snap = make_snapshot(session=sess)
    engine = make_engine(session_name="LONDON")
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE


def test_overlap_london_ny_counts_for_new_york_target():
    """OVERLAP_LONDON_NY should also satisfy a NEW_YORK-targeted engine."""
    sess = make_session(
        session_name=SessionName.OVERLAP_LONDON_NY,
        opening_range_completed=True,
    )
    snap = make_snapshot(session=sess)
    engine = make_engine(session_name="NEW_YORK")
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE


def test_new_york_engine_rejects_london_only():
    sess = make_session(session_name=SessionName.LONDON)
    snap = make_snapshot(session=sess)
    engine = make_engine(session_name="NEW_YORK")
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.NO_TRADE


# ─────────────────────────────────────────────────────────────────────────────
# NO_TRADE — OR not completed
# ─────────────────────────────────────────────────────────────────────────────


def test_or_active_is_no_trade_when_require_completed():
    sess = make_session(
        session_name=SessionName.LONDON,
        opening_range_active=True,
        opening_range_completed=False,
    )
    snap = make_snapshot(session=sess)
    engine = make_engine(require_completed_range=True)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "OR_NOT_COMPLETED"


def test_or_active_is_candidate_when_not_require_completed():
    """require_completed_range=False → engine evaluates even while OR is active."""
    sess = make_session(
        session_name=SessionName.LONDON,
        opening_range_active=True,
        opening_range_completed=False,
    )
    snap = make_snapshot(
        session=sess,
        feature_vector=make_feature_vector(or_high=1.1020, or_low=1.1000),
    )
    engine = make_engine(require_completed_range=False)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE


# ─────────────────────────────────────────────────────────────────────────────
# NO_TRADE — OR levels unavailable
# ─────────────────────────────────────────────────────────────────────────────


def test_or_high_none_is_no_trade():
    fv = make_feature_vector(or_high=None, or_low=1.1000)
    snap = make_snapshot(feature_vector=fv)
    engine = make_engine()
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "OR_UNAVAILABLE"


def test_or_low_none_is_no_trade():
    fv = make_feature_vector(or_high=1.1050, or_low=None)
    snap = make_snapshot(feature_vector=fv)
    engine = make_engine()
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "OR_UNAVAILABLE"


def test_or_high_equals_or_low_is_no_trade():
    """Zero-width range → UNAVAILABLE."""
    fv = make_feature_vector(or_high=1.1000, or_low=1.1000)
    snap = make_snapshot(feature_vector=fv)
    engine = make_engine()
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "OR_UNAVAILABLE"


# ─────────────────────────────────────────────────────────────────────────────
# NO_TRADE — range size violations
# ─────────────────────────────────────────────────────────────────────────────


def test_range_too_tight():
    """or_high - or_low = 0.0002 = 2 pips, below min_range_pips=5."""
    fv = make_feature_vector(or_high=1.1002, or_low=1.1000)
    snap = make_snapshot(feature_vector=fv)
    engine = make_engine(min_range_pips=5.0, max_range_pips=40.0)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "RANGE_TOO_TIGHT"


def test_range_too_tight_includes_levels_in_decision():
    fv = make_feature_vector(or_high=1.1002, or_low=1.1000)
    snap = make_snapshot(feature_vector=fv)
    engine = make_engine(min_range_pips=5.0)
    result = engine.evaluate(snap)
    assert result.no_trade.or_high == pytest.approx(1.1002)
    assert result.no_trade.or_low == pytest.approx(1.1000)
    assert result.no_trade.or_state == OpeningRangeState.COMPLETED


def test_range_too_wide():
    """or_high - or_low = 0.0060 = 60 pips, above max_range_pips=40."""
    fv = make_feature_vector(or_high=1.1060, or_low=1.1000)
    snap = make_snapshot(feature_vector=fv)
    engine = make_engine(min_range_pips=5.0, max_range_pips=40.0)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "RANGE_TOO_WIDE"


def test_range_well_above_minimum_is_candidate():
    """10-pip range is well above the 5-pip minimum — should produce CANDIDATE."""
    # 10 pips: or_high - or_low = 0.0010 → 10 / (0.00001 * 10) = 10 pips
    fv = make_feature_vector(or_high=1.1010, or_low=1.1000)
    snap = make_snapshot(feature_vector=fv)
    engine = make_engine(min_range_pips=5.0, max_range_pips=40.0)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE


def test_range_well_below_maximum_is_candidate():
    """20-pip range is well below the 40-pip maximum — should produce CANDIDATE."""
    fv = make_feature_vector(or_high=1.1020, or_low=1.1000)
    snap = make_snapshot(feature_vector=fv)
    engine = make_engine(min_range_pips=5.0, max_range_pips=40.0)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE


# ─────────────────────────────────────────────────────────────────────────────
# INSUFFICIENT_DATA — low quality score
# ─────────────────────────────────────────────────────────────────────────────


def test_low_quality_score_is_insufficient_data():
    snap = make_snapshot(quality_score=0.50)
    engine = make_engine(min_quality=MIN_QUALITY_SCORE)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.INSUFFICIENT_DATA


def test_low_quality_score_reason_detail_mentions_score():
    snap = make_snapshot(quality_score=0.50)
    engine = make_engine(min_quality=MIN_QUALITY_SCORE)
    result = engine.evaluate(snap)
    assert result.reason_detail is not None
    assert "0.5" in result.reason_detail


def test_quality_at_threshold_is_not_rejected():
    """Exactly at threshold → sufficient."""
    snap = make_snapshot(quality_score=MIN_QUALITY_SCORE)
    engine = make_engine(min_quality=MIN_QUALITY_SCORE)
    result = engine.evaluate(snap)
    assert result.outcome != StrategyOutcome.INSUFFICIENT_DATA


# ─────────────────────────────────────────────────────────────────────────────
# Result structure integrity
# ─────────────────────────────────────────────────────────────────────────────


def test_no_trade_result_has_no_candidate():
    sess = make_session(session_name=SessionName.ASIA)
    snap = make_snapshot(session=sess)
    engine = make_engine(session_name="LONDON")
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.candidate is None


def test_insufficient_data_result_has_neither():
    snap = make_snapshot(quality_score=0.0)
    engine = make_engine(min_quality=MIN_QUALITY_SCORE)
    result = engine.evaluate(snap)
    assert result.candidate is None
    assert result.no_trade is None


def test_result_timestamp_matches_snapshot():
    snap = make_snapshot(ts=_BASE_TS)
    engine = make_engine()
    result = engine.evaluate(snap)
    assert result.timestamp_utc == _BASE_TS


def test_result_symbol_matches_snapshot():
    snap = make_snapshot()
    engine = make_engine()
    result = engine.evaluate(snap)
    assert result.symbol == "EURUSD"


# ─────────────────────────────────────────────────────────────────────────────
# NO_TRADE — retest too deep
# ─────────────────────────────────────────────────────────────────────────────


def test_retest_none_skips_check():
    """max_retest_penetration_points=None disables the retest check entirely."""
    # close=1.1005: 15 pips below OR high — deeply retesting, but filter is off
    snap = make_snap_for_retest(close=1.1005)
    engine = make_engine(max_retest_penetration_points=None)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE


def test_retest_within_limit_is_candidate():
    """LONG: close 5 pips below OR high -> penetration=5 < limit=10 -> CANDIDATE."""
    # or_high=1.1020, close=1.1015 -> penetration = (1.1020-1.1015)/0.0001 = 5 pips
    snap = make_snap_for_retest(close=1.1015)
    engine = make_engine(max_retest_penetration_points=10.0)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE


def test_retest_too_deep_is_no_trade():
    """LONG: close 15 pips below OR high -> penetration=15 > limit=10 -> NO_TRADE."""
    # or_high=1.1020, close=1.1005 -> penetration = (1.1020-1.1005)/0.0001 = 15 pips
    snap = make_snap_for_retest(close=1.1005)
    engine = make_engine(max_retest_penetration_points=10.0)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.NO_TRADE


def test_retest_too_deep_reason_code():
    """RETEST_TOO_DEEP reason code is set when penetration exceeds limit."""
    snap = make_snap_for_retest(close=1.1005)
    engine = make_engine(max_retest_penetration_points=10.0)
    result = engine.evaluate(snap)
    assert result.no_trade.reason_code == "RETEST_TOO_DEEP"


def test_retest_too_deep_carries_or_levels():
    """NoTradeDecision includes or_high and or_low when RETEST_TOO_DEEP fires."""
    snap = make_snap_for_retest(close=1.1005, or_high=1.1020, or_low=1.1000)
    engine = make_engine(max_retest_penetration_points=10.0)
    result = engine.evaluate(snap)
    assert result.no_trade.or_high == pytest.approx(1.1020)
    assert result.no_trade.or_low == pytest.approx(1.1000)


def test_retest_above_entry_is_always_candidate():
    """LONG: close above OR high -> no retest (clean breakout) -> CANDIDATE."""
    # close=1.1025 > or_high=1.1020 -> penetration_price < 0 -> check skipped
    snap = make_snap_for_retest(close=1.1025)
    engine = make_engine(max_retest_penetration_points=5.0)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE


def test_retest_short_within_limit_is_candidate():
    """SHORT: close 5 pips above OR low -> penetration=5 < limit=10 -> CANDIDATE."""
    # or_low=1.1000, close=1.1005 -> penetration = (1.1005-1.1000)/0.0001 = 5 pips
    snap = make_snap_for_retest(close=1.1005, or_high=1.1020, or_low=1.1000)
    engine = make_engine(direction_bias=TradeDirection.SHORT, max_retest_penetration_points=10.0)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE


def test_retest_short_too_deep_is_no_trade():
    """SHORT: close 15 pips above OR low -> penetration=15 > limit=10 -> NO_TRADE."""
    # or_low=1.1000, close=1.1015 -> penetration = (1.1015-1.1000)/0.0001 = 15 pips
    snap = make_snap_for_retest(close=1.1015, or_high=1.1020, or_low=1.1000)
    engine = make_engine(direction_bias=TradeDirection.SHORT, max_retest_penetration_points=10.0)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.NO_TRADE


def test_retest_clearly_within_limit_is_candidate():
    """Penetration clearly below the limit passes (8 pips < 10 pip limit)."""
    # or_high=1.1020, close=1.1012 -> penetration = 8 pips < limit=10
    snap = make_snap_for_retest(close=1.1012)
    engine = make_engine(max_retest_penetration_points=10.0)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE
