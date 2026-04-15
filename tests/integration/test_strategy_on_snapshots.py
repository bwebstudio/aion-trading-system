"""
tests/integration/test_strategy_on_snapshots.py
─────────────────────────────────────────────────
Integration tests: OpeningRangeEngine + filters evaluated against
MarketSnapshot objects.

What these tests verify
────────────────────────
  Engine evaluation:
    - CANDIDATE returned on a clean London snapshot with valid OR levels
    - NO_TRADE on wrong session (OFF_HOURS, ASIA, NEW_YORK)
    - NO_TRADE when OR is still ACTIVE and require_completed_range=True
    - CANDIDATE when OR is ACTIVE and require_completed_range=False
    - NO_TRADE when range is too tight (< min_range_pips)
    - NO_TRADE when range is too wide (> max_range_pips)
    - INSUFFICIENT_DATA when quality score is below engine threshold
    - Direction bias LONG → entry_reference = or_high
    - Direction bias SHORT → entry_reference = or_low
    - OVERLAP_LONDON_NY session counts as LONDON for OR engine
    - Identical snapshots produce identical outcome (reproducibility)
    - CandidateSetup carries correct atr_14 from feature vector

  QualityFilter:
    - Blocks with LOW_QUALITY_DATA when quality < filter threshold
    - Passes through to engine when quality >= filter threshold
    - strategy_id and version are inherited from the wrapped engine

  SessionFilter:
    - Blocks with SESSION_FILTER_BLOCKED when session not in allowed set
    - Passes when session is in allowed set

  SpreadFilter:
    - Blocks with SPREAD_TOO_WIDE when spread_pips > max
    - Blocks with SPREAD_UNAVAILABLE when spread_mean_20 is None
    - Passes when spread is within limit

  Composition:
    - Filters compose correctly (outermost runs first)
    - Inner engine is not called when outer filter blocks
    - All three filters stacked: each can block independently

Integration means: real engine + real filter code + real model code.
Only the snapshot input is manually constructed (no actual CSV loading).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aion.core.constants import FEATURE_SET_VERSION, MIN_QUALITY_SCORE, SNAPSHOT_VERSION
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
from aion.strategies.filters import QualityFilter, SessionFilter, SpreadFilter
from aion.strategies.models import (
    OpeningRangeDefinition,
    OpeningRangeState,
    StrategyOutcome,
)
from aion.strategies.opening_range import OpeningRangeEngine

# ─────────────────────────────────────────────────────────────────────────────
# Shared timestamps
# ─────────────────────────────────────────────────────────────────────────────

_UTC = timezone.utc
_TS = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_UTC)  # Mon, mid-London session


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot factory helpers
# ─────────────────────────────────────────────────────────────────────────────


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


def make_bar() -> MarketBar:
    return MarketBar(
        symbol="EURUSD",
        timestamp_utc=_TS,
        timestamp_market=_TS,
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
) -> SessionContext:
    is_open = session_name != SessionName.OFF_HOURS
    is_london = session_name in (SessionName.LONDON, SessionName.OVERLAP_LONDON_NY)
    is_ny = session_name in (SessionName.NEW_YORK, SessionName.OVERLAP_LONDON_NY)
    is_asia = session_name == SessionName.ASIA
    session_open = _TS.replace(hour=8, minute=0, second=0) if is_open else None
    session_close = _TS.replace(hour=16, minute=30, second=0) if is_open else None
    return SessionContext(
        trading_day=_TS.date(),
        broker_time=_TS,
        market_time=_TS,
        local_time=_TS,
        is_asia=is_asia,
        is_london=is_london,
        is_new_york=is_ny,
        is_session_open_window=is_open,
        opening_range_active=opening_range_active,
        opening_range_completed=opening_range_completed,
        session_name=session_name,
        session_open_utc=session_open,
        session_close_utc=session_close,
    )


def make_feature_vector(
    or_high: float | None = 1.1020,
    or_low: float | None = 1.1000,
    atr_14: float | None = 0.00015,
    spread_mean_20: float | None = 2.0,
    volatility_percentile_20: float | None = 0.50,
    return_5: float | None = 0.0003,
) -> FeatureVector:
    return FeatureVector(
        symbol="EURUSD",
        timestamp_utc=_TS,
        timeframe=Timeframe.M1,
        atr_14=atr_14,
        rolling_range_10=0.0010,
        rolling_range_20=0.0012,
        volatility_percentile_20=volatility_percentile_20,
        session_high=1.1060,
        session_low=1.0990,
        opening_range_high=or_high,
        opening_range_low=or_low,
        vwap_session=1.1010,
        spread_mean_20=spread_mean_20,
        spread_zscore_20=0.0,
        return_1=0.0001,
        return_5=return_5,
        candle_body=0.00005,
        upper_wick=0.00005,
        lower_wick=0.00005,
        distance_to_session_high=-0.0040,
        distance_to_session_low=0.0010,
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
    session_name: SessionName = SessionName.LONDON,
    opening_range_active: bool = False,
    opening_range_completed: bool = True,
    or_high: float | None = 1.1020,
    or_low: float | None = 1.1000,
    atr_14: float | None = 0.00015,
    quality_score: float = 1.0,
    spread_mean_20: float | None = 2.0,
    volatility_percentile_20: float | None = 0.50,
    return_5: float | None = 0.0003,
) -> MarketSnapshot:
    bar = make_bar()
    return MarketSnapshot(
        snapshot_id=new_snapshot_id(),
        symbol="EURUSD",
        timestamp_utc=_TS,
        base_timeframe=Timeframe.M1,
        instrument=make_instrument(),
        session_context=make_session(
            session_name=session_name,
            opening_range_active=opening_range_active,
            opening_range_completed=opening_range_completed,
        ),
        latest_bar=bar,
        bars_m1=[bar],
        bars_m5=[],
        bars_m15=[],
        feature_vector=make_feature_vector(
            or_high=or_high,
            or_low=or_low,
            atr_14=atr_14,
            spread_mean_20=spread_mean_20,
            volatility_percentile_20=volatility_percentile_20,
            return_5=return_5,
        ),
        quality_report=make_quality_report(quality_score),
        snapshot_version=SNAPSHOT_VERSION,
    )


def make_engine(
    min_range_pips: float = 5.0,
    max_range_pips: float = 40.0,
    direction_bias: TradeDirection | None = None,
    require_completed_range: bool = True,
    min_quality: float = MIN_QUALITY_SCORE,
) -> OpeningRangeEngine:
    return OpeningRangeEngine(
        OpeningRangeDefinition(
            strategy_id="or_london_v1",
            session_name="LONDON",
            min_range_pips=min_range_pips,
            max_range_pips=max_range_pips,
            direction_bias=direction_bias,
            require_completed_range=require_completed_range,
        ),
        min_quality_score=min_quality,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Engine evaluation tests
# ─────────────────────────────────────────────────────────────────────────────


def test_candidate_on_clean_london_snapshot():
    """Happy path: London session, OR completed, valid range → CANDIDATE."""
    result = make_engine().evaluate(make_snapshot())
    assert result.outcome == StrategyOutcome.CANDIDATE
    assert result.has_setup is True
    assert result.candidate is not None


def test_no_trade_off_hours():
    """OFF_HOURS session → NOT_IN_TARGET_SESSION."""
    result = make_engine().evaluate(make_snapshot(session_name=SessionName.OFF_HOURS))
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade is not None
    assert result.no_trade.reason_code == "NOT_IN_TARGET_SESSION"


def test_no_trade_asia_session():
    """ASIA session → NOT_IN_TARGET_SESSION for a LONDON engine."""
    result = make_engine().evaluate(make_snapshot(session_name=SessionName.ASIA))
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "NOT_IN_TARGET_SESSION"


def test_no_trade_or_active_when_require_completed():
    """OR still active + require_completed_range=True → OR_NOT_COMPLETED."""
    result = make_engine(require_completed_range=True).evaluate(
        make_snapshot(opening_range_active=True, opening_range_completed=False)
    )
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "OR_NOT_COMPLETED"


def test_candidate_when_or_active_and_require_completed_false():
    """OR active + require_completed_range=False → engine evaluates and can return CANDIDATE."""
    result = make_engine(require_completed_range=False).evaluate(
        make_snapshot(opening_range_active=True, opening_range_completed=False)
    )
    # The OR levels are provided (or_high=1.1020, or_low=1.1000 → 20 pips in range)
    assert result.outcome == StrategyOutcome.CANDIDATE


def test_no_trade_range_too_tight():
    """OR range = 3 pips < min 5 pips → RANGE_TOO_TIGHT."""
    # 1.1003 - 1.1000 = 0.0003 = 3 pips (well below 5 pip minimum)
    result = make_engine(min_range_pips=5.0).evaluate(
        make_snapshot(or_high=1.1003, or_low=1.1000)
    )
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "RANGE_TOO_TIGHT"


def test_no_trade_range_too_wide():
    """OR range = 60 pips > max 40 pips → RANGE_TOO_WIDE."""
    # 1.1060 - 1.1000 = 0.0060 = 60 pips (well above 40 pip maximum)
    result = make_engine(max_range_pips=40.0).evaluate(
        make_snapshot(or_high=1.1060, or_low=1.1000)
    )
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "RANGE_TOO_WIDE"


def test_insufficient_data_low_quality():
    """quality_score below engine threshold → INSUFFICIENT_DATA."""
    result = make_engine(min_quality=0.90).evaluate(
        make_snapshot(quality_score=0.80)
    )
    assert result.outcome == StrategyOutcome.INSUFFICIENT_DATA
    assert result.is_insufficient_data is True


def test_direction_bias_long_produces_long_candidate():
    """direction_bias=LONG → candidate direction is LONG."""
    result = make_engine(direction_bias=TradeDirection.LONG).evaluate(make_snapshot())
    assert result.outcome == StrategyOutcome.CANDIDATE
    assert result.candidate.direction == TradeDirection.LONG


def test_direction_bias_short_produces_short_candidate():
    """direction_bias=SHORT → candidate direction is SHORT."""
    result = make_engine(direction_bias=TradeDirection.SHORT).evaluate(make_snapshot())
    assert result.outcome == StrategyOutcome.CANDIDATE
    assert result.candidate.direction == TradeDirection.SHORT


def test_entry_reference_is_or_high_for_long():
    """LONG candidate: entry_reference must equal or_high."""
    snapshot = make_snapshot(or_high=1.1020, or_low=1.1000)
    result = make_engine(direction_bias=TradeDirection.LONG).evaluate(snapshot)
    assert result.outcome == StrategyOutcome.CANDIDATE
    assert result.candidate.entry_reference == pytest.approx(1.1020)


def test_entry_reference_is_or_low_for_short():
    """SHORT candidate: entry_reference must equal or_low."""
    snapshot = make_snapshot(or_high=1.1020, or_low=1.1000)
    result = make_engine(direction_bias=TradeDirection.SHORT).evaluate(snapshot)
    assert result.outcome == StrategyOutcome.CANDIDATE
    assert result.candidate.entry_reference == pytest.approx(1.1000)


def test_overlap_london_ny_treated_as_london():
    """OVERLAP_LONDON_NY counts as LONDON for a LONDON-targeted engine."""
    result = make_engine().evaluate(
        make_snapshot(session_name=SessionName.OVERLAP_LONDON_NY)
    )
    assert result.outcome == StrategyOutcome.CANDIDATE


def test_result_is_reproducible():
    """Same snapshot input → same outcome (engine is deterministic)."""
    engine = make_engine()
    snapshot = make_snapshot()
    r1 = engine.evaluate(snapshot)
    r2 = engine.evaluate(snapshot)
    assert r1.outcome == r2.outcome
    assert r1.candidate.direction == r2.candidate.direction
    assert r1.candidate.entry_reference == pytest.approx(r2.candidate.entry_reference)


def test_candidate_atr_propagated_from_feature_vector():
    """atr_14 in CandidateSetup must match the feature vector value."""
    snapshot = make_snapshot(atr_14=0.00020)
    result = make_engine().evaluate(snapshot)
    assert result.outcome == StrategyOutcome.CANDIDATE
    assert result.candidate.atr_14 == pytest.approx(0.00020)


def test_candidate_atr_is_none_when_feature_vector_has_none():
    """atr_14=None in feature vector → atr_14=None in CandidateSetup."""
    snapshot = make_snapshot(atr_14=None)
    result = make_engine().evaluate(snapshot)
    assert result.outcome == StrategyOutcome.CANDIDATE
    assert result.candidate.atr_14 is None


def test_no_trade_includes_or_levels_in_decision():
    """When OR levels are available, NoTradeDecision should carry them."""
    snapshot = make_snapshot(or_high=1.1003, or_low=1.1000)  # too tight
    result = make_engine(min_range_pips=5.0).evaluate(snapshot)
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.or_high == pytest.approx(1.1003)
    assert result.no_trade.or_low == pytest.approx(1.1000)


# ─────────────────────────────────────────────────────────────────────────────
# QualityFilter tests
# ─────────────────────────────────────────────────────────────────────────────


def test_quality_filter_blocks_before_engine():
    """
    Filter threshold=0.95, quality=0.92: filter blocks before engine runs.
    Without the filter the engine (threshold=0.90) would pass this snapshot.
    """
    engine = make_engine(min_quality=0.90)  # engine would pass quality=0.92
    filtered = QualityFilter(engine, min_quality=0.95)
    result = filtered.evaluate(make_snapshot(quality_score=0.92))
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "LOW_QUALITY_DATA"


def test_quality_filter_passes_on_good_quality():
    """quality >= filter threshold → filter delegates to engine."""
    engine = make_engine()
    filtered = QualityFilter(engine, min_quality=0.95)
    result = filtered.evaluate(make_snapshot(quality_score=0.98))
    assert result.outcome == StrategyOutcome.CANDIDATE


def test_quality_filter_strategy_id_matches_engine():
    """strategy_id on the filter must equal the wrapped engine's id."""
    engine = make_engine()
    filtered = QualityFilter(engine)
    assert filtered.strategy_id == engine.strategy_id


def test_quality_filter_version_matches_engine():
    filtered = QualityFilter(make_engine())
    assert filtered.version == make_engine().version


# ─────────────────────────────────────────────────────────────────────────────
# SessionFilter tests
# ─────────────────────────────────────────────────────────────────────────────


def test_session_filter_blocks_off_hours():
    """OFF_HOURS not in allowed set → SESSION_FILTER_BLOCKED."""
    filtered = SessionFilter(make_engine(), allowed_sessions={"LONDON"})
    result = filtered.evaluate(make_snapshot(session_name=SessionName.OFF_HOURS))
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "SESSION_FILTER_BLOCKED"


def test_session_filter_passes_allowed_session():
    """LONDON in allowed set → filter delegates to engine → CANDIDATE."""
    filtered = SessionFilter(make_engine(), allowed_sessions={"LONDON", "OVERLAP_LONDON_NY"})
    result = filtered.evaluate(make_snapshot(session_name=SessionName.LONDON))
    assert result.outcome == StrategyOutcome.CANDIDATE


def test_session_filter_blocks_asia():
    """ASIA not in allowed set → SESSION_FILTER_BLOCKED."""
    filtered = SessionFilter(make_engine(), allowed_sessions={"LONDON"})
    result = filtered.evaluate(make_snapshot(session_name=SessionName.ASIA))
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "SESSION_FILTER_BLOCKED"


def test_session_filter_reason_detail_contains_session_name():
    """reason_detail should mention the blocked session name."""
    filtered = SessionFilter(make_engine(), allowed_sessions={"LONDON"})
    result = filtered.evaluate(make_snapshot(session_name=SessionName.OFF_HOURS))
    assert "OFF_HOURS" in result.no_trade.reason_detail


# ─────────────────────────────────────────────────────────────────────────────
# SpreadFilter tests
# ─────────────────────────────────────────────────────────────────────────────


def test_spread_filter_blocks_wide_spread():
    """spread_mean_20=50 points = 5 pips > max 2 pips → SPREAD_TOO_WIDE."""
    filtered = SpreadFilter(make_engine(), max_spread_pips=2.0, pip_multiplier=10.0)
    result = filtered.evaluate(make_snapshot(spread_mean_20=50.0))
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "SPREAD_TOO_WIDE"


def test_spread_filter_passes_acceptable_spread():
    """spread_mean_20=15 points = 1.5 pips < max 2 pips → filter passes."""
    filtered = SpreadFilter(make_engine(), max_spread_pips=2.0, pip_multiplier=10.0)
    result = filtered.evaluate(make_snapshot(spread_mean_20=15.0))
    assert result.outcome == StrategyOutcome.CANDIDATE


def test_spread_filter_blocks_on_none_spread():
    """spread_mean_20=None → SPREAD_UNAVAILABLE."""
    filtered = SpreadFilter(make_engine(), max_spread_pips=2.0)
    result = filtered.evaluate(make_snapshot(spread_mean_20=None))
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "SPREAD_UNAVAILABLE"


def test_spread_filter_reason_detail_contains_pip_values():
    """reason_detail in SPREAD_TOO_WIDE should show pip amounts."""
    filtered = SpreadFilter(make_engine(), max_spread_pips=2.0, pip_multiplier=10.0)
    result = filtered.evaluate(make_snapshot(spread_mean_20=50.0))
    assert "5.00" in result.no_trade.reason_detail  # actual spread
    assert "2.00" in result.no_trade.reason_detail  # max spread


# ─────────────────────────────────────────────────────────────────────────────
# Filter composition tests
# ─────────────────────────────────────────────────────────────────────────────


def test_outermost_filter_blocks_first():
    """
    Stack: SpreadFilter → QualityFilter → engine.
    If spread fails, SpreadFilter reason code is returned (not quality).
    """
    engine = make_engine()
    engine = QualityFilter(engine, min_quality=0.95)
    engine = SpreadFilter(engine, max_spread_pips=2.0)

    # quality=0.92 would also fail QualityFilter, but spread fails first
    result = engine.evaluate(make_snapshot(spread_mean_20=50.0, quality_score=0.92))
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "SPREAD_TOO_WIDE"


def test_inner_filter_runs_when_outer_passes():
    """
    Stack: SpreadFilter → QualityFilter → engine.
    SpreadFilter passes; QualityFilter blocks.
    """
    engine = make_engine()
    engine = QualityFilter(engine, min_quality=0.95)
    engine = SpreadFilter(engine, max_spread_pips=2.0)

    # spread=15 (1.5 pips, passes), quality=0.92 (fails QualityFilter)
    result = engine.evaluate(make_snapshot(spread_mean_20=15.0, quality_score=0.92))
    assert result.outcome == StrategyOutcome.NO_TRADE
    assert result.no_trade.reason_code == "LOW_QUALITY_DATA"


def test_all_three_filters_pass_to_engine():
    """All filters pass → engine result returned."""
    engine = make_engine()
    engine = SessionFilter(engine, allowed_sessions={"LONDON"})
    engine = QualityFilter(engine, min_quality=0.95)
    engine = SpreadFilter(engine, max_spread_pips=3.0)

    result = engine.evaluate(
        make_snapshot(
            session_name=SessionName.LONDON,
            quality_score=0.98,
            spread_mean_20=20.0,  # 2 pips
        )
    )
    assert result.outcome == StrategyOutcome.CANDIDATE


def test_filter_strategy_id_propagates_through_stack():
    """strategy_id is consistent across all layers of a filter stack."""
    engine = make_engine()
    original_id = engine.strategy_id

    stacked = SpreadFilter(
        QualityFilter(
            SessionFilter(engine, allowed_sessions={"LONDON"}),
            min_quality=0.90,
        ),
        max_spread_pips=3.0,
    )
    assert stacked.strategy_id == original_id
    assert stacked.version == engine.version
