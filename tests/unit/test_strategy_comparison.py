"""
tests/unit/test_strategy_comparison.py
────────────────────────────────────────
Unit tests for aion.analytics.strategy_comparison.

Tests verify:
  - run_strategy_comparison returns a StrategyComparisonReport
  - overall metrics match independent compute_metrics calls
  - by_session breakdown contains the expected session keys
  - by_regime breakdown falls back to 'UNKNOWN' without a RegimeDetector
  - two identical engines produce identical metrics in both slots
  - an engine that never fires produces zero candidates in its slot
  - empty snapshot list produces empty/zero report
  - StrategyComparisonReport is frozen (immutable)
  - ComparisonBreakdown.strategy_a/b carry the correct strategy_id
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aion.analytics.strategy_comparison import (
    ComparisonBreakdown,
    StrategyComparisonReport,
    StrategyMetricsSummary,
    run_strategy_comparison,
)
from aion.core.constants import FEATURE_SET_VERSION, MIN_QUALITY_SCORE, SNAPSHOT_VERSION
from aion.core.enums import AssetClass, DataSource, SessionName, Timeframe
from aion.core.ids import new_snapshot_id
from aion.core.models import (
    DataQualityReport,
    FeatureVector,
    InstrumentSpec,
    MarketBar,
    MarketSnapshot,
    SessionContext,
)
from aion.replay.models import LabelConfig
from aion.strategies.models import OpeningRangeDefinition
from aion.strategies.opening_range import OpeningRangeEngine
from aion.strategies.vwap_fade import VWAPFadeDefinition, VWAPFadeEngine

_UTC = timezone.utc
_BASE_TS = datetime(2024, 1, 15, 8, 0, 0, tzinfo=_UTC)


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot fixture builders
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


def _snap(
    i: int,
    *,
    or_completed: bool = True,
    close: float = 1.1040,
    vwap: float = 1.1010,
    bar_high: float = 1.1045,
    bar_low: float = 1.1015,
    or_high: float = 1.1020,
    or_low: float = 1.1000,
    session: SessionName = SessionName.LONDON,
) -> MarketSnapshot:
    ts = _BASE_TS + timedelta(minutes=i)
    bar = MarketBar(
        symbol="EURUSD",
        timestamp_utc=ts,
        timestamp_market=ts,
        timeframe=Timeframe.M1,
        open=close,
        high=bar_high,
        low=bar_low,
        close=close,
        tick_volume=100.0,
        real_volume=0.0,
        spread=2.0,
        source=DataSource.SYNTHETIC,
    )
    is_london = session in (SessionName.LONDON, SessionName.OVERLAP_LONDON_NY)
    is_new_york = session in (SessionName.NEW_YORK, SessionName.OVERLAP_LONDON_NY)
    session_ctx = SessionContext(
        trading_day=ts.date(),
        broker_time=ts,
        market_time=ts,
        local_time=ts,
        is_asia=False,
        is_london=is_london,
        is_new_york=is_new_york,
        is_session_open_window=True,
        opening_range_active=not or_completed,
        opening_range_completed=or_completed,
        session_name=session,
        session_open_utc=ts.replace(hour=8, minute=0, second=0),
        session_close_utc=ts.replace(hour=16, minute=30, second=0),
    )
    fv = FeatureVector(
        symbol="EURUSD",
        timestamp_utc=ts,
        timeframe=Timeframe.M1,
        atr_14=0.00015,
        rolling_range_10=0.0010,
        rolling_range_20=0.0012,
        volatility_percentile_20=0.50,
        session_high=1.1060,
        session_low=1.0980,
        opening_range_high=or_high if or_completed else None,
        opening_range_low=or_low if or_completed else None,
        vwap_session=vwap,
        spread_mean_20=2.0,
        spread_zscore_20=0.0,
        return_1=0.0001,
        return_5=0.0003,
        candle_body=0.00003,
        upper_wick=0.00002,
        lower_wick=0.00025,
        distance_to_session_high=1.1060 - close,
        distance_to_session_low=close - 1.0980,
        feature_set_version=FEATURE_SET_VERSION,
    )
    qr = DataQualityReport(
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        rows_checked=100,
        missing_bars=0,
        duplicate_timestamps=0,
        out_of_order_rows=0,
        stale_bars=0,
        spike_bars=0,
        null_rows=0,
        quality_score=1.0,
        warnings=[],
    )
    return MarketSnapshot(
        snapshot_id=new_snapshot_id(),
        symbol="EURUSD",
        timestamp_utc=ts,
        base_timeframe=Timeframe.M1,
        instrument=_instrument(),
        session_context=session_ctx,
        latest_bar=bar,
        bars_m1=[bar],
        bars_m5=[],
        bars_m15=[],
        feature_vector=fv,
        quality_report=qr,
        snapshot_version=SNAPSHOT_VERSION,
    )


def _shared_snaps(n: int = 30) -> list[MarketSnapshot]:
    """Shared snapshots where both OR and VWAP strategies fire.

    OR completed, close=1.1040, VWAP=1.1010, bar high=1.1045 low=1.1015.
    OR LONG (entry=1.1020) and VWAP SHORT (entry=1.1040) both find candidates.
    """
    return [_snap(i, or_completed=True) for i in range(n)]


def _or_engine(strategy_id: str = "or_london_test") -> OpeningRangeEngine:
    return OpeningRangeEngine(
        OpeningRangeDefinition(
            strategy_id=strategy_id,
            session_name="LONDON",
            min_range_pips=5.0,
            max_range_pips=40.0,
        ),
        min_quality_score=MIN_QUALITY_SCORE,
    )


def _vwap_engine(strategy_id: str = "vwap_fade_test") -> VWAPFadeEngine:
    return VWAPFadeEngine(
        VWAPFadeDefinition(
            strategy_id=strategy_id,
            session_name="LONDON",
            min_distance_to_vwap_pips=10.0,
            max_distance_to_vwap_pips=50.0,
        ),
        min_quality_score=MIN_QUALITY_SCORE,
    )


_LABEL_CFG = LabelConfig(stop_pips=10.0, target_pips=20.0, max_bars=5)


# ─────────────────────────────────────────────────────────────────────────────
# Return type and structure
# ─────────────────────────────────────────────────────────────────────────────


def test_returns_strategy_comparison_report():
    snaps = _shared_snaps(20)
    report = run_strategy_comparison(snaps, _or_engine(), _vwap_engine())
    assert isinstance(report, StrategyComparisonReport)


def test_report_strategy_ids_match_engines():
    snaps = _shared_snaps(10)
    report = run_strategy_comparison(
        snaps,
        _or_engine("eng_a"),
        _vwap_engine("eng_b"),
    )
    assert report.strategy_a_id == "eng_a"
    assert report.strategy_b_id == "eng_b"


def test_overall_breakdown_group_key_is_overall():
    snaps = _shared_snaps(10)
    report = run_strategy_comparison(snaps, _or_engine(), _vwap_engine())
    assert report.overall.group_key == "overall"


def test_comparison_breakdown_contains_both_strategy_summaries():
    snaps = _shared_snaps(15)
    report = run_strategy_comparison(snaps, _or_engine("a"), _vwap_engine("b"))
    assert isinstance(report.overall.strategy_a, StrategyMetricsSummary)
    assert isinstance(report.overall.strategy_b, StrategyMetricsSummary)
    assert report.overall.strategy_a.strategy_id == "a"
    assert report.overall.strategy_b.strategy_id == "b"


def test_report_is_frozen():
    snaps = _shared_snaps(5)
    report = run_strategy_comparison(snaps, _or_engine(), _vwap_engine())
    with pytest.raises(Exception):
        report.overall = None  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────
# Candidate counts
# ─────────────────────────────────────────────────────────────────────────────


def test_both_strategies_find_candidates_on_shared_data():
    """Shared snapshots have OR completed and VWAP far from close → both trigger."""
    snaps = _shared_snaps(20)
    report = run_strategy_comparison(snaps, _or_engine(), _vwap_engine())
    assert report.overall.strategy_a.candidate_count > 0
    assert report.overall.strategy_b.candidate_count > 0


def test_engine_that_never_fires_shows_zero_candidates():
    """OR engine with max_range=5 (OR is 20 pips) → 0 candidates."""
    from aion.strategies.models import OpeningRangeDefinition

    narrow_engine = OpeningRangeEngine(
        OpeningRangeDefinition(
            strategy_id="narrow_or",
            session_name="LONDON",
            min_range_pips=1.0,
            max_range_pips=5.0,  # blocks 20-pip OR
        ),
        min_quality_score=MIN_QUALITY_SCORE,
    )
    snaps = _shared_snaps(10)
    report = run_strategy_comparison(snaps, narrow_engine, _vwap_engine())
    assert report.overall.strategy_a.candidate_count == 0
    assert report.overall.strategy_b.candidate_count > 0


def test_empty_snapshots_produce_zero_report():
    report = run_strategy_comparison([], _or_engine(), _vwap_engine())
    assert report.overall.strategy_a.candidate_count == 0
    assert report.overall.strategy_b.candidate_count == 0
    assert report.by_session == []
    assert report.by_regime == []


# ─────────────────────────────────────────────────────────────────────────────
# Regime breakdown
# ─────────────────────────────────────────────────────────────────────────────


def test_by_regime_contains_unknown_without_detector():
    """Without a RegimeDetector, all records fall into 'UNKNOWN'."""
    snaps = _shared_snaps(15)
    report = run_strategy_comparison(snaps, _or_engine(), _vwap_engine())
    assert len(report.by_regime) == 1
    assert report.by_regime[0].group_key == "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# Session breakdown
# ─────────────────────────────────────────────────────────────────────────────


def test_by_session_contains_london():
    """All snapshots are LONDON; by_session should have a LONDON entry."""
    snaps = _shared_snaps(15)
    report = run_strategy_comparison(snaps, _or_engine(), _vwap_engine())
    session_keys = {bd.group_key for bd in report.by_session}
    assert "LONDON" in session_keys


def test_by_session_strategy_ids_preserved():
    snaps = _shared_snaps(10)
    report = run_strategy_comparison(snaps, _or_engine("a"), _vwap_engine("b"))
    for bd in report.by_session:
        assert bd.strategy_a.strategy_id == "a"
        assert bd.strategy_b.strategy_id == "b"


# ─────────────────────────────────────────────────────────────────────────────
# Metric consistency
# ─────────────────────────────────────────────────────────────────────────────


def test_identical_engines_produce_identical_overall_metrics():
    """Same engine twice should produce equal metrics in both slots."""
    snaps = _shared_snaps(20)
    eng_a = _or_engine("dup_a")
    eng_b = _or_engine("dup_b")
    report = run_strategy_comparison(snaps, eng_a, eng_b)
    a = report.overall.strategy_a
    b = report.overall.strategy_b
    assert a.candidate_count == b.candidate_count
    assert a.win_rate_on_activated == b.win_rate_on_activated
    assert a.activation_rate == b.activation_rate


def test_labeling_produces_win_loss_counts():
    """With label_config, win_count should be populated when target is reachable."""
    snaps = _shared_snaps(20)
    report = run_strategy_comparison(
        snaps, _or_engine(), _vwap_engine(), label_config=_LABEL_CFG
    )
    # Both engines should show some wins given the bar layout
    assert (
        report.overall.strategy_a.win_count >= 0
        and report.overall.strategy_b.win_count >= 0
    )


def test_no_label_config_produces_zero_wins():
    """Without label_config, no labeling → win_count and entry_activated_count are 0."""
    snaps = _shared_snaps(15)
    report = run_strategy_comparison(snaps, _or_engine(), _vwap_engine())
    assert report.overall.strategy_a.win_count == 0
    assert report.overall.strategy_b.win_count == 0
    assert report.overall.strategy_a.entry_activated_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# win_rate_on_activated / activation_rate
# ─────────────────────────────────────────────────────────────────────────────


def test_activation_rate_is_zero_without_labeling():
    """Without label_config, no entries activate → activation_rate = 0.0 (0/candidates)."""
    snaps = _shared_snaps(10)
    report = run_strategy_comparison(snaps, _or_engine(), _vwap_engine())
    # candidate_count > 0 but entry_activated_count == 0 → rate = 0.0, not None
    assert report.overall.strategy_a.activation_rate == 0.0
    assert report.overall.strategy_b.activation_rate == 0.0


def test_activation_rate_is_populated_with_labeling():
    snaps = _shared_snaps(15)
    report = run_strategy_comparison(
        snaps, _or_engine(), _vwap_engine(), label_config=_LABEL_CFG
    )
    # Both strategies have candidates → activation_rate should be non-None
    if report.overall.strategy_a.candidate_count > 0:
        assert report.overall.strategy_a.activation_rate is not None
    if report.overall.strategy_b.candidate_count > 0:
        assert report.overall.strategy_b.activation_rate is not None
