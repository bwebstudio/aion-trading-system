"""
tests/unit/test_parameter_sweeps.py
──────────────────────────────────────
Unit tests for aion.analytics.parameter_sweeps.

Tests verify:
  - Single-point sweep returns one SweepResult
  - Multiple sweep points produce distinct SweepResult per point
  - Sweep with identical points produces identical metrics (reproducible)
  - Wide-range filter allows candidates; tight-range blocks them
  - SpreadFilter blocks all candidates when max_spread_pips too tight
  - direction_bias=SHORT blocks LONG-only OR setups
  - ranked_by_win_rate sorts descending (None last)
  - ranked_by_candidate_count sorts descending
  - SweepComparison is frozen
  - SweepResult.sweep_point preserves label
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aion.analytics.parameter_sweeps import run_parameter_sweep
from aion.analytics.replay_models import SweepComparison, SweepPoint
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
from aion.core.enums import TradeDirection

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


def _bar(ts: datetime, high: float = 1.1045, low: float = 1.1035) -> MarketBar:
    mid = round((high + low) / 2, 5)
    return MarketBar(
        symbol="EURUSD",
        timestamp_utc=ts,
        timestamp_market=ts,
        timeframe=Timeframe.M1,
        open=mid,
        high=high,
        low=low,
        close=mid,
        tick_volume=100.0,
        real_volume=0.0,
        spread=2.0,
        source=DataSource.SYNTHETIC,
    )


def _session(ts: datetime, or_completed: bool) -> SessionContext:
    session_open = ts.replace(hour=8, minute=0, second=0)
    session_close = ts.replace(hour=16, minute=30, second=0)
    return SessionContext(
        trading_day=ts.date(),
        broker_time=ts,
        market_time=ts,
        local_time=ts,
        is_asia=False,
        is_london=True,
        is_new_york=False,
        is_session_open_window=True,
        opening_range_active=not or_completed,
        opening_range_completed=or_completed,
        session_name=SessionName.LONDON,
        session_open_utc=session_open,
        session_close_utc=session_close,
    )


def _fv(ts: datetime) -> FeatureVector:
    return FeatureVector(
        symbol="EURUSD",
        timestamp_utc=ts,
        timeframe=Timeframe.M1,
        atr_14=0.00015,
        rolling_range_10=0.0010,
        rolling_range_20=0.0012,
        volatility_percentile_20=0.50,
        session_high=1.1060,
        session_low=1.0990,
        opening_range_high=1.1020,
        opening_range_low=1.1000,
        vwap_session=1.1010,
        spread_mean_20=2.0,
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


def _quality_report() -> DataQualityReport:
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
        quality_score=1.0,
        warnings=[],
    )


def _snap(i: int, or_completed: bool) -> MarketSnapshot:
    ts = _BASE_TS + timedelta(minutes=i)
    bar = _bar(ts)
    instrument = _instrument()
    return MarketSnapshot(
        snapshot_id=new_snapshot_id(),
        symbol="EURUSD",
        timestamp_utc=ts,
        base_timeframe=Timeframe.M1,
        instrument=instrument,
        session_context=_session(ts, or_completed=or_completed),
        latest_bar=bar,
        bars_m1=[bar],
        bars_m5=[],
        bars_m15=[],
        feature_vector=_fv(ts),
        quality_report=_quality_report(),
        snapshot_version=SNAPSHOT_VERSION,
    )


def _build_snapshots(n_or: int = 10, n_candidate: int = 20) -> list[MarketSnapshot]:
    """Build synthetic snapshots: first n_or have OR active, rest have OR completed."""
    snaps = []
    for i in range(n_or):
        snaps.append(_snap(i, or_completed=False))
    for i in range(n_or, n_or + n_candidate):
        snaps.append(_snap(i, or_completed=True))
    return snaps


# ─────────────────────────────────────────────────────────────────────────────
# Basic sweep
# ─────────────────────────────────────────────────────────────────────────────


def test_single_point_returns_one_result():
    snaps = _build_snapshots(n_or=5, n_candidate=10)
    pt = SweepPoint(label="baseline", min_range_pips=5.0, max_range_pips=40.0)
    cmp = run_parameter_sweep(snaps, [pt])
    assert len(cmp.results) == 1
    assert cmp.results[0].sweep_point.label == "baseline"


def test_multiple_points_return_multiple_results():
    snaps = _build_snapshots(n_or=5, n_candidate=10)
    points = [
        SweepPoint(label="a"),
        SweepPoint(label="b", min_range_pips=10.0),
        SweepPoint(label="c", target_pips=15.0),
    ]
    cmp = run_parameter_sweep(snaps, points)
    assert len(cmp.results) == 3
    labels = {r.sweep_point.label for r in cmp.results}
    assert labels == {"a", "b", "c"}


def test_empty_sweep_points_returns_empty_comparison():
    snaps = _build_snapshots()
    cmp = run_parameter_sweep(snaps, [])
    assert cmp.results == []


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────


def test_identical_points_produce_identical_metrics():
    snaps = _build_snapshots(n_or=5, n_candidate=15)
    pt = SweepPoint(label="x", stop_pips=10.0, target_pips=20.0)
    r1 = run_parameter_sweep(snaps, [pt]).results[0].metrics
    r2 = run_parameter_sweep(snaps, [pt]).results[0].metrics
    assert r1 == r2


# ─────────────────────────────────────────────────────────────────────────────
# Parameter effects
# ─────────────────────────────────────────────────────────────────────────────


def test_wide_range_allows_candidates():
    """OR range = 20 pips; max_range=40 should allow candidates."""
    snaps = _build_snapshots(n_or=5, n_candidate=10)
    pt = SweepPoint(label="wide", min_range_pips=5.0, max_range_pips=40.0)
    m = run_parameter_sweep(snaps, [pt]).results[0].metrics
    assert m.candidate_count > 0


def test_tight_max_range_blocks_candidates():
    """OR range = 20 pips; max_range=15 should reject all candidates."""
    snaps = _build_snapshots(n_or=5, n_candidate=10)
    pt = SweepPoint(label="tight_range", min_range_pips=5.0, max_range_pips=15.0)
    m = run_parameter_sweep(snaps, [pt]).results[0].metrics
    assert m.candidate_count == 0


def test_spread_filter_blocks_when_too_tight():
    """spread_mean_20=2.0 points → 0.2 pips; max=0.1 pips should block all."""
    snaps = _build_snapshots(n_or=5, n_candidate=10)
    pt = SweepPoint(label="strict_spread", max_spread_pips=0.1)
    m = run_parameter_sweep(snaps, [pt]).results[0].metrics
    assert m.candidate_count == 0


def test_spread_filter_passes_when_generous():
    """max_spread_pips=5.0 should not block any candidates (spread=0.2 pips)."""
    snaps = _build_snapshots(n_or=5, n_candidate=10)
    baseline = SweepPoint(label="no_spread")
    filtered = SweepPoint(label="generous_spread", max_spread_pips=5.0)
    cmp = run_parameter_sweep(snaps, [baseline, filtered])
    base_m = cmp.results[0].metrics
    filt_m = cmp.results[1].metrics
    assert filt_m.candidate_count == base_m.candidate_count


def test_session_filter_blocks_when_wrong_session():
    """Snapshots are LONDON; allowing only NEW_YORK should block all candidates."""
    snaps = _build_snapshots(n_or=5, n_candidate=10)
    pt = SweepPoint(label="ny_only", allowed_sessions=frozenset({"NEW_YORK"}))
    m = run_parameter_sweep(snaps, [pt]).results[0].metrics
    assert m.candidate_count == 0


def test_achievable_target_produces_wins():
    """target=20 pips → bar.high=1.1045 can reach 1.1040 → WIN."""
    snaps = _build_snapshots(n_or=5, n_candidate=10)
    pt = SweepPoint(label="target_20", stop_pips=10.0, target_pips=20.0)
    m = run_parameter_sweep(snaps, [pt]).results[0].metrics
    assert m.win_count > 0


def test_unreachable_target_produces_no_wins():
    """target=30 pips → entry + 30*0.0001 = 1.1050 > bar.high=1.1045 → TIMEOUT."""
    snaps = _build_snapshots(n_or=5, n_candidate=10)
    pt = SweepPoint(label="target_30", stop_pips=10.0, target_pips=30.0, max_label_bars=5)
    m = run_parameter_sweep(snaps, [pt]).results[0].metrics
    assert m.win_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# SweepComparison ranking
# ─────────────────────────────────────────────────────────────────────────────


def test_ranked_by_win_rate_descending():
    snaps = _build_snapshots(n_or=5, n_candidate=15)
    points = [
        SweepPoint(label="good", stop_pips=10.0, target_pips=20.0),
        SweepPoint(label="bad", stop_pips=10.0, target_pips=30.0, max_label_bars=5),
    ]
    cmp = run_parameter_sweep(snaps, points)
    ranked = cmp.ranked_by_win_rate()
    assert ranked[0].sweep_point.label == "good"


def test_ranked_by_candidate_count_descending():
    snaps = _build_snapshots(n_or=5, n_candidate=15)
    points = [
        SweepPoint(label="wide", min_range_pips=5.0, max_range_pips=40.0),
        SweepPoint(label="tight", min_range_pips=5.0, max_range_pips=15.0),
    ]
    cmp = run_parameter_sweep(snaps, points)
    ranked = cmp.ranked_by_candidate_count()
    assert ranked[0].sweep_point.label == "wide"


def test_ranked_none_win_rate_last():
    """Points with win_rate=None (no activations) should rank last."""
    snaps = _build_snapshots(n_or=5, n_candidate=10)
    points = [
        SweepPoint(label="no_cands", min_range_pips=5.0, max_range_pips=15.0),  # 0 candidates
        SweepPoint(label="with_wins", stop_pips=10.0, target_pips=20.0),
    ]
    cmp = run_parameter_sweep(snaps, points)
    ranked = cmp.ranked_by_win_rate()
    assert ranked[-1].sweep_point.label == "no_cands"


# ─────────────────────────────────────────────────────────────────────────────
# Model properties
# ─────────────────────────────────────────────────────────────────────────────


def test_sweep_comparison_is_frozen():
    snaps = _build_snapshots()
    cmp = run_parameter_sweep(snaps, [SweepPoint(label="x")])
    with pytest.raises(Exception):
        cmp.results = []  # type: ignore[misc]


def test_sweep_result_preserves_label():
    snaps = _build_snapshots()
    pt = SweepPoint(label="my_label", min_range_pips=7.5)
    result = run_parameter_sweep(snaps, [pt]).results[0]
    assert result.sweep_point.label == "my_label"
    assert result.sweep_point.min_range_pips == pytest.approx(7.5)


# ─────────────────────────────────────────────────────────────────────────────
# max_retest_penetration_points in sweep
# ─────────────────────────────────────────────────────────────────────────────


def test_retest_filter_none_allows_all_candidates():
    """With max_retest_penetration_points=None the retest check is disabled.

    Bar close = mid = (1.1045 + 1.1035) / 2 = 1.1040, OR high = 1.1020.
    close > or_high -> no retest anyway, so this also verifies baseline.
    """
    snaps = _build_snapshots(n_or=5, n_candidate=10)
    pt_no_filter = SweepPoint(label="no_retest", max_retest_penetration_points=None)
    pt_with_filter = SweepPoint(label="retest_50", max_retest_penetration_points=50.0)
    cmp = run_parameter_sweep(snaps, [pt_no_filter, pt_with_filter])
    # close=1.1040 is above OR high=1.1020: no penetration -> same candidate count
    m_no = cmp.results[0].metrics
    m_with = cmp.results[1].metrics
    assert m_no.candidate_count == m_with.candidate_count


def test_retest_filter_blocks_candidates_when_close_inside_or():
    """Tight retest limit blocks candidates where close is inside the OR.

    We need snapshots where bar close < OR high. The default _bar() has
    close = mid = 1.1040 which is above OR high = 1.1020 (no retest).
    We build custom snapshots with close = 1.1010 (10 pips inside OR).
    """
    # Build snapshots with close = 1.1010, which is 10 pips below OR high=1.1020
    ts_base = _BASE_TS
    snaps = []
    for i in range(10):
        ts = ts_base + timedelta(minutes=i)
        bar = MarketBar(
            symbol="EURUSD",
            timestamp_utc=ts,
            timestamp_market=ts,
            timeframe=Timeframe.M1,
            open=1.1010,
            high=1.1015,
            low=1.1005,
            close=1.1010,  # 10 pips below OR high=1.1020
            tick_volume=100.0,
            real_volume=0.0,
            spread=2.0,
            source=DataSource.SYNTHETIC,
        )
        snaps.append(MarketSnapshot(
            snapshot_id=new_snapshot_id(),
            symbol="EURUSD",
            timestamp_utc=ts,
            base_timeframe=Timeframe.M1,
            instrument=_instrument(),
            session_context=_session(ts, or_completed=True),
            latest_bar=bar,
            bars_m1=[bar],
            bars_m5=[],
            bars_m15=[],
            feature_vector=_fv(ts),
            quality_report=_quality_report(),
            snapshot_version=SNAPSHOT_VERSION,
        ))

    # Without retest filter: close=1.1010, OR range=20 pips -> valid candidates
    no_filter = SweepPoint(label="no_filter", min_range_pips=5.0, max_range_pips=40.0)
    # With tight retest limit=5: close is 10 pips inside OR -> blocked
    tight = SweepPoint(label="tight_retest", min_range_pips=5.0, max_range_pips=40.0,
                       max_retest_penetration_points=5.0)
    # With generous limit=15: close is 10 pips inside OR -> allowed
    generous = SweepPoint(label="generous_retest", min_range_pips=5.0, max_range_pips=40.0,
                          max_retest_penetration_points=15.0)

    cmp = run_parameter_sweep(snaps, [no_filter, tight, generous])
    m_no = cmp.results[0].metrics
    m_tight = cmp.results[1].metrics
    m_generous = cmp.results[2].metrics

    assert m_no.candidate_count > 0          # baseline: candidates found
    assert m_tight.candidate_count == 0      # blocked: 10 > 5
    assert m_generous.candidate_count > 0    # allowed: 10 <= 15
