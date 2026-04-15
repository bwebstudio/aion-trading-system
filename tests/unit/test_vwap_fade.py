"""
tests/unit/test_vwap_fade.py
──────────────────────────────
Unit tests for aion.strategies.vwap_fade.

Tests verify:
  - LONG candidate when close < VWAP by sufficient pips
  - SHORT candidate when close > VWAP by sufficient pips
  - INSUFFICIENT_DATA when quality score is too low
  - INSUFFICIENT_DATA when vwap_session is None
  - NO_TRADE(EXTENSION_TOO_SMALL) when distance < min
  - NO_TRADE(EXTENSION_TOO_LARGE) when distance > max
  - NO_TRADE(NOT_IN_TARGET_SESSION) for wrong session
  - NO_TRADE(DIRECTION_BIAS_MISMATCH) when signal contradicts bias
  - NO_TRADE(NO_REJECTION_SIGNAL) when require_rejection=True but no signal
  - Rejection: bullish bar allows LONG fade with require_rejection
  - Rejection: bearish bar allows SHORT fade with require_rejection
  - Rejection: lower wick > body allows LONG fade with require_rejection
  - Rejection: upper wick > body allows SHORT fade with require_rejection
  - session_name="ALL" bypasses session check
  - OVERLAP_LONDON_NY counts as LONDON target
  - Candidate entry_reference == bar.close
  - Candidate range_high/low correct for LONG and SHORT
  - Candidate range_size_pips == distance to VWAP
  - Candidate direction LONG when below VWAP, SHORT when above
  - strategy_id and version propagate correctly
  - VWAPFadeDefinition validation rejects invalid configs
  - min_distance boundary: exactly at min → CANDIDATE
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aion.core.constants import FEATURE_SET_VERSION, MIN_QUALITY_SCORE
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
from aion.strategies.models import StrategyOutcome
from aion.strategies.vwap_fade import VWAPFadeDefinition, VWAPFadeEngine

_UTC = timezone.utc
_TS = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_UTC)

# ─────────────────────────────────────────────────────────────────────────────
# Snapshot factory
# ─────────────────────────────────────────────────────────────────────────────

_INSTRUMENT = InstrumentSpec(
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


def _make_bar(
    *,
    open_: float = 1.1022,
    high: float = 1.1030,
    low: float = 1.1020,
    close: float = 1.1025,
) -> MarketBar:
    return MarketBar(
        symbol="EURUSD",
        timestamp_utc=_TS,
        timestamp_market=_TS,
        timeframe=Timeframe.M1,
        open=open_,
        high=high,
        low=low,
        close=close,
        tick_volume=100.0,
        real_volume=0.0,
        spread=2.0,
        source=DataSource.SYNTHETIC,
    )


def _make_session(session: SessionName = SessionName.LONDON) -> SessionContext:
    session_open = _TS.replace(hour=8, minute=0, second=0)
    session_close = _TS.replace(hour=16, minute=30, second=0)
    return SessionContext(
        trading_day=_TS.date(),
        broker_time=_TS,
        market_time=_TS,
        local_time=_TS,
        is_asia=session == SessionName.ASIA,
        is_london=session in (SessionName.LONDON, SessionName.OVERLAP_LONDON_NY),
        is_new_york=session in (SessionName.NEW_YORK, SessionName.OVERLAP_LONDON_NY),
        is_session_open_window=True,
        opening_range_active=False,
        opening_range_completed=True,
        session_name=session,
        session_open_utc=session_open,
        session_close_utc=session_close,
    )


def _make_fv(
    *,
    vwap: float | None = 1.1010,
    atr_14: float | None = 0.00015,
    spread_mean_20: float | None = 2.0,
) -> FeatureVector:
    return FeatureVector(
        symbol="EURUSD",
        timestamp_utc=_TS,
        timeframe=Timeframe.M1,
        atr_14=atr_14,
        rolling_range_10=0.0010,
        rolling_range_20=0.0012,
        volatility_percentile_20=0.50,
        session_high=1.1060,
        session_low=1.0990,
        opening_range_high=1.1020,
        opening_range_low=1.1000,
        vwap_session=vwap,
        spread_mean_20=spread_mean_20,
        spread_zscore_20=0.0,
        return_1=0.0001,
        return_5=0.0003,
        candle_body=0.00005,
        upper_wick=0.00005,
        lower_wick=0.00005,
        distance_to_session_high=-0.0040,
        distance_to_session_low=0.0010,
        feature_set_version=FEATURE_SET_VERSION,
    )


def _make_quality(quality_score: float = 1.0) -> DataQualityReport:
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
    *,
    close: float = 1.1025,
    open_: float = 1.1022,
    high: float = 1.1030,
    low: float = 1.1020,
    vwap: float | None = 1.1010,
    quality_score: float = 1.0,
    session: SessionName = SessionName.LONDON,
    atr_14: float | None = 0.00015,
) -> MarketSnapshot:
    bar = _make_bar(open_=open_, high=high, low=low, close=close)
    return MarketSnapshot(
        snapshot_id=new_snapshot_id(),
        symbol="EURUSD",
        timestamp_utc=_TS,
        base_timeframe=Timeframe.M1,
        instrument=_INSTRUMENT,
        session_context=_make_session(session),
        latest_bar=bar,
        bars_m1=[bar],
        bars_m5=[],
        bars_m15=[],
        feature_vector=_make_fv(vwap=vwap, atr_14=atr_14),
        quality_report=_make_quality(quality_score),
        snapshot_version="1.0.0",
    )


def make_engine(
    *,
    session_name: str = "LONDON",
    min_distance: float = 10.0,
    max_distance: float = 50.0,
    require_rejection: bool = False,
    direction_bias: TradeDirection | None = None,
    min_quality: float = MIN_QUALITY_SCORE,
) -> VWAPFadeEngine:
    defn = VWAPFadeDefinition(
        strategy_id="vwap_fade_v1",
        session_name=session_name,
        min_distance_to_vwap_pips=min_distance,
        max_distance_to_vwap_pips=max_distance,
        require_rejection=require_rejection,
        direction_bias=direction_bias,
    )
    return VWAPFadeEngine(defn, min_quality_score=min_quality)


# ─────────────────────────────────────────────────────────────────────────────
# Happy paths: candidate generation
# ─────────────────────────────────────────────────────────────────────────────


def test_short_candidate_when_above_vwap():
    """close=1.1025, vwap=1.1010 → 15 pips above → SHORT CANDIDATE."""
    engine = make_engine()
    snap = make_snapshot(close=1.1025, vwap=1.1010)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE
    assert result.candidate.direction == TradeDirection.SHORT


def test_long_candidate_when_below_vwap():
    """close=1.0995, vwap=1.1010 → 15 pips below → LONG CANDIDATE."""
    engine = make_engine()
    snap = make_snapshot(close=1.0995, vwap=1.1010, low=1.0990, high=1.1000)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE
    assert result.candidate.direction == TradeDirection.LONG


def test_candidate_entry_reference_is_bar_close():
    engine = make_engine()
    snap = make_snapshot(close=1.1025, vwap=1.1010)
    result = engine.evaluate(snap)
    assert result.candidate.entry_reference == pytest.approx(1.1025)


def test_short_candidate_range_high_is_close():
    """SHORT fade: close > vwap → range_high = close."""
    engine = make_engine()
    snap = make_snapshot(close=1.1025, vwap=1.1010)
    c = engine.evaluate(snap).candidate
    assert c.range_high == pytest.approx(1.1025)
    assert c.range_low == pytest.approx(1.1010)


def test_long_candidate_range_high_is_vwap():
    """LONG fade: close < vwap → range_high = vwap."""
    engine = make_engine()
    snap = make_snapshot(close=1.0995, vwap=1.1010, low=1.0990, high=1.1000)
    c = engine.evaluate(snap).candidate
    assert c.range_high == pytest.approx(1.1010)
    assert c.range_low == pytest.approx(1.0995)


def test_candidate_range_size_pips_equals_distance():
    """range_size_pips should equal the pip distance from close to VWAP."""
    engine = make_engine()
    snap = make_snapshot(close=1.1025, vwap=1.1010)
    # distance = |1.1025 - 1.1010| / 0.0001 = 15 pips
    c = engine.evaluate(snap).candidate
    assert c.range_size_pips == pytest.approx(15.0, abs=0.01)


def test_candidate_strategy_id_propagated():
    engine = make_engine()
    snap = make_snapshot(close=1.1025, vwap=1.1010)
    result = engine.evaluate(snap)
    assert result.strategy_id == "vwap_fade_v1"
    assert result.candidate.strategy_id == "vwap_fade_v1"


def test_candidate_version_propagated():
    engine = make_engine()
    snap = make_snapshot(close=1.1025, vwap=1.1010)
    result = engine.evaluate(snap)
    assert result.candidate.strategy_version == "1.0.0"


def test_candidate_session_name_from_context():
    engine = make_engine(session_name="ALL")
    snap = make_snapshot(close=1.1025, vwap=1.1010, session=SessionName.NEW_YORK)
    c = engine.evaluate(snap).candidate
    assert c.session_name == "NEW_YORK"


def test_candidate_quality_score_propagated():
    engine = make_engine()
    snap = make_snapshot(close=1.1025, vwap=1.1010, quality_score=0.95)
    c = engine.evaluate(snap).candidate
    assert c.quality_score == pytest.approx(0.95)


def test_candidate_atr_14_propagated():
    engine = make_engine()
    snap = make_snapshot(close=1.1025, vwap=1.1010, atr_14=0.00020)
    c = engine.evaluate(snap).candidate
    assert c.atr_14 == pytest.approx(0.00020)


def test_candidate_atr_14_none_allowed():
    """atr_14=None in feature vector should pass through to CandidateSetup."""
    engine = make_engine()
    snap = make_snapshot(close=1.1025, vwap=1.1010, atr_14=None)
    c = engine.evaluate(snap).candidate
    assert c.atr_14 is None


# ─────────────────────────────────────────────────────────────────────────────
# Guard: data quality
# ─────────────────────────────────────────────────────────────────────────────


def test_insufficient_data_when_quality_too_low():
    engine = make_engine(min_quality=0.90)
    snap = make_snapshot(close=1.1025, vwap=1.1010, quality_score=0.85)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.INSUFFICIENT_DATA


def test_quality_exactly_at_threshold_is_accepted():
    engine = make_engine(min_quality=0.90)
    snap = make_snapshot(close=1.1025, vwap=1.1010, quality_score=0.90)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE


# ─────────────────────────────────────────────────────────────────────────────
# Guard: VWAP availability
# ─────────────────────────────────────────────────────────────────────────────


def test_insufficient_data_when_vwap_is_none():
    engine = make_engine()
    snap = make_snapshot(close=1.1025, vwap=None)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.INSUFFICIENT_DATA


# ─────────────────────────────────────────────────────────────────────────────
# Guard: session
# ─────────────────────────────────────────────────────────────────────────────


def test_no_trade_when_wrong_session():
    engine = make_engine(session_name="LONDON")
    snap = make_snapshot(close=1.1025, vwap=1.1010, session=SessionName.NEW_YORK)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "NOT_IN_TARGET_SESSION"


def test_session_all_bypasses_check():
    """session_name='ALL' should accept any session."""
    engine = make_engine(session_name="ALL")
    snap = make_snapshot(close=1.1025, vwap=1.1010, session=SessionName.ASIA)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE


def test_overlap_london_ny_counts_as_london_target():
    engine = make_engine(session_name="LONDON")
    snap = make_snapshot(
        close=1.1025, vwap=1.1010, session=SessionName.OVERLAP_LONDON_NY
    )
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE


def test_overlap_london_ny_counts_as_new_york_target():
    engine = make_engine(session_name="NEW_YORK")
    snap = make_snapshot(
        close=1.1025, vwap=1.1010, session=SessionName.OVERLAP_LONDON_NY
    )
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE


# ─────────────────────────────────────────────────────────────────────────────
# Guard: distance bounds
# ─────────────────────────────────────────────────────────────────────────────


def test_no_trade_when_extension_too_small():
    """close=1.1013, vwap=1.1010 → 3 pips; min=10 → EXTENSION_TOO_SMALL."""
    engine = make_engine(min_distance=10.0)
    snap = make_snapshot(close=1.1013, vwap=1.1010)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "EXTENSION_TOO_SMALL"


def test_no_trade_when_extension_too_large():
    """close=1.1070, vwap=1.1010 → 60 pips; max=50 → EXTENSION_TOO_LARGE."""
    engine = make_engine(max_distance=50.0)
    snap = make_snapshot(close=1.1070, vwap=1.1010, high=1.1075, low=1.1060)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "EXTENSION_TOO_LARGE"


def test_min_distance_exactly_at_boundary_is_candidate():
    """distance == min_distance_to_vwap_pips should produce a CANDIDATE."""
    engine = make_engine(min_distance=15.0)
    # close=1.1025, vwap=1.1010 → exactly 15 pips
    snap = make_snapshot(close=1.1025, vwap=1.1010)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE


def test_within_distance_range_is_candidate():
    """distance within [min, max] should produce a CANDIDATE."""
    engine = make_engine(min_distance=10.0, max_distance=20.0)
    # close=1.1025, vwap=1.1010 → ~15 pips, clearly within 10-20 range
    snap = make_snapshot(close=1.1025, vwap=1.1010)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE


# ─────────────────────────────────────────────────────────────────────────────
# Guard: direction bias
# ─────────────────────────────────────────────────────────────────────────────


def test_direction_bias_long_blocks_short_signal():
    """Signal is SHORT but bias=LONG → DIRECTION_BIAS_MISMATCH."""
    engine = make_engine(direction_bias=TradeDirection.LONG)
    snap = make_snapshot(close=1.1025, vwap=1.1010)  # above VWAP → SHORT
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "DIRECTION_BIAS_MISMATCH"


def test_direction_bias_short_blocks_long_signal():
    """Signal is LONG but bias=SHORT → DIRECTION_BIAS_MISMATCH."""
    engine = make_engine(direction_bias=TradeDirection.SHORT)
    snap = make_snapshot(close=1.0995, vwap=1.1010, low=1.0990, high=1.1000)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "DIRECTION_BIAS_MISMATCH"


def test_direction_bias_short_allows_short_signal():
    engine = make_engine(direction_bias=TradeDirection.SHORT)
    snap = make_snapshot(close=1.1025, vwap=1.1010)  # above VWAP → SHORT
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE


def test_direction_bias_long_allows_long_signal():
    engine = make_engine(direction_bias=TradeDirection.LONG)
    snap = make_snapshot(close=1.0995, vwap=1.1010, low=1.0990, high=1.1000)
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE


# ─────────────────────────────────────────────────────────────────────────────
# Guard: rejection signal
# ─────────────────────────────────────────────────────────────────────────────


def test_no_rejection_signal_blocks_when_required():
    """Flat bar (open==close, no wicks) → NO_REJECTION_SIGNAL."""
    engine = make_engine(require_rejection=True)
    # SHORT fade: flat bar has no bearish signal and no upper wick
    snap = make_snapshot(
        close=1.1025, open_=1.1025, high=1.1025, low=1.1025, vwap=1.1010
    )
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "NO_REJECTION_SIGNAL"


def test_bearish_bar_allows_short_fade_with_rejection():
    """Bearish bar (close < open) → valid rejection for SHORT fade."""
    engine = make_engine(require_rejection=True)
    snap = make_snapshot(
        close=1.1025, open_=1.1035, high=1.1040, low=1.1020, vwap=1.1010
    )
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE


def test_upper_wick_allows_short_fade_with_rejection():
    """Upper wick > body on doji → valid rejection for SHORT fade."""
    engine = make_engine(require_rejection=True)
    # doji (open==close=1.1025) with upper wick = 1.1035 - 1.1025 = 10 pips > body=0
    snap = make_snapshot(
        close=1.1025, open_=1.1025, high=1.1035, low=1.1020, vwap=1.1010
    )
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE


def test_bullish_bar_allows_long_fade_with_rejection():
    """Bullish bar (close > open) → valid rejection for LONG fade."""
    engine = make_engine(require_rejection=True)
    snap = make_snapshot(
        close=1.0995, open_=1.0985, high=1.0997, low=1.0983, vwap=1.1010
    )
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE


def test_lower_wick_allows_long_fade_with_rejection():
    """Lower wick > body on doji → valid rejection for LONG fade."""
    engine = make_engine(require_rejection=True)
    # doji at 1.0995 with lower wick = 1.0995 - 1.0980 = 15 pips > body=0
    snap = make_snapshot(
        close=1.0995, open_=1.0995, high=1.0997, low=1.0980, vwap=1.1010
    )
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE


def test_rejection_not_required_flat_bar_still_passes():
    """require_rejection=False (default) → flat bar produces CANDIDATE."""
    engine = make_engine(require_rejection=False)
    snap = make_snapshot(
        close=1.1025, open_=1.1025, high=1.1025, low=1.1025, vwap=1.1010
    )
    result = engine.evaluate(snap)
    assert result.outcome == StrategyOutcome.CANDIDATE


# ─────────────────────────────────────────────────────────────────────────────
# VWAPFadeDefinition validation
# ─────────────────────────────────────────────────────────────────────────────


def test_definition_rejects_zero_min_distance():
    with pytest.raises(Exception):
        VWAPFadeDefinition(
            strategy_id="x",
            min_distance_to_vwap_pips=0.0,
            max_distance_to_vwap_pips=50.0,
        )


def test_definition_rejects_max_less_than_min():
    with pytest.raises(Exception):
        VWAPFadeDefinition(
            strategy_id="x",
            min_distance_to_vwap_pips=30.0,
            max_distance_to_vwap_pips=20.0,
        )


def test_definition_pip_size_property():
    defn = VWAPFadeDefinition(
        strategy_id="x",
        tick_size=0.00001,
        pip_multiplier=10.0,
    )
    assert defn.pip_size == pytest.approx(0.0001)


def test_definition_pips_to_price():
    defn = VWAPFadeDefinition(strategy_id="x")
    assert defn.pips_to_price(15.0) == pytest.approx(0.0015)


def test_definition_price_to_pips():
    defn = VWAPFadeDefinition(strategy_id="x")
    assert defn.price_to_pips(0.0015) == pytest.approx(15.0)


# ─────────────────────────────────────────────────────────────────────────────
# Engine properties
# ─────────────────────────────────────────────────────────────────────────────


def test_engine_strategy_id():
    engine = make_engine()
    assert engine.strategy_id == "vwap_fade_v1"


def test_engine_version():
    engine = make_engine()
    assert engine.version == "1.0.0"


def test_engine_definition_property():
    engine = make_engine()
    assert engine.definition.session_name == "LONDON"
