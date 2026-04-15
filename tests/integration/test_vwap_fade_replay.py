"""
tests/integration/test_vwap_fade_replay.py
────────────────────────────────────────────
Integration tests: VWAPFadeEngine → run_replay → labeler.

Snapshot design:
  Snapshots 0-29 : close near VWAP (3 pips) → EXTENSION_TOO_SMALL → NO_TRADE
  Snapshots 30-99: close 15 pips above VWAP  → SHORT candidate → WIN on labeling

Labeling setup for SHORT fade (stop=10, target=20 pips):
  entry_reference = bar.close = 1.1025
  stop_price      = 1.1025 + 10*0.0001 = 1.1035
  target_price    = 1.1025 - 20*0.0001 = 1.1005
  Future bars     : high=1.1030, low=1.0995
    → Entry activation : low=1.0995 <= entry=1.1025  ✓
    → Stop check       : high=1.1030 >= stop=1.1035? NO ✓
    → Target check     : low=1.0995 <= target=1.1005? YES → WIN ✓
  Exception: last candidate (snapshot 99) has no future bars → ENTRY_NOT_ACTIVATED

Tests verify:
  - len(records) == len(snapshots)
  - records have sequential bar_index
  - no_trade_count == 30 (first 30 snapshots, EXTENSION_TOO_SMALL)
  - candidate_count == 70 (snapshots 30-99)
  - label_count == candidate_count (all candidates labeled)
  - win_count == 69 (last candidate has no future bars)
  - entry_not_activated == 1 (last candidate)
  - entry direction is SHORT for all candidates
  - reason_code for no-trade is EXTENSION_TOO_SMALL
  - replay result can round-trip through CandidateJournal
  - run_replay with no label_config produces 0 labels
  - basic analytics metrics consistent with replay result
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aion.analytics.replay_metrics import compute_metrics
from aion.analytics.replay_reports import breakdown_by_reason_code, build_report
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
from aion.replay.journal import CandidateJournal
from aion.replay.models import LabelConfig, LabelOutcome
from aion.replay.runner import run_replay
from aion.strategies.models import StrategyOutcome
from aion.strategies.vwap_fade import VWAPFadeDefinition, VWAPFadeEngine

_UTC = timezone.utc
_BASE_TS = datetime(2024, 1, 15, 8, 0, 0, tzinfo=_UTC)

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic snapshot builder
# ─────────────────────────────────────────────────────────────────────────────

_VWAP = 1.1010
_CLOSE_NEAR = 1.1013    # 3 pips above VWAP → NO_TRADE (EXTENSION_TOO_SMALL)
_CLOSE_EXTENDED = 1.1025  # 15 pips above VWAP → SHORT CANDIDATE

# Future bars for SHORT fade:
#   entry=1.1025, stop=1.1035, target=1.1005
#   bar.high=1.1030 < stop=1.1035 → no stop
#   bar.low=1.0995 <= target=1.1005 → WIN
_FUTURE_HIGH = 1.1030
_FUTURE_LOW = 1.0995


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


def _bar(ts: datetime, *, close: float, high: float, low: float) -> MarketBar:
    return MarketBar(
        symbol="EURUSD",
        timestamp_utc=ts,
        timestamp_market=ts,
        timeframe=Timeframe.M1,
        open=close,
        high=high,
        low=low,
        close=close,
        tick_volume=100.0,
        real_volume=0.0,
        spread=2.0,
        source=DataSource.SYNTHETIC,
    )


def _session(ts: datetime) -> SessionContext:
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
        opening_range_active=False,
        opening_range_completed=True,
        session_name=SessionName.LONDON,
        session_open_utc=session_open,
        session_close_utc=session_close,
    )


def _fv(ts: datetime, vwap: float) -> FeatureVector:
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
        vwap_session=vwap,
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


def _quality() -> DataQualityReport:
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


def build_vwap_snapshots(n: int = 100, n_near: int = 30) -> list[MarketSnapshot]:
    """
    Build synthetic snapshots for VWAP Fade testing.

    Snapshots 0..(n_near-1) : close near VWAP (_CLOSE_NEAR) → NO_TRADE
    Snapshots n_near..(n-1) : close extended from VWAP (_CLOSE_EXTENDED) → CANDIDATE
    """
    instrument = _instrument()
    snaps = []
    for i in range(n):
        ts = _BASE_TS + timedelta(minutes=i)
        is_extended = i >= n_near
        close = _CLOSE_EXTENDED if is_extended else _CLOSE_NEAR
        high = _FUTURE_HIGH if is_extended else (_CLOSE_NEAR + 0.0002)
        low = _FUTURE_LOW if is_extended else (_CLOSE_NEAR - 0.0002)
        bar = _bar(ts, close=close, high=high, low=low)
        snap = MarketSnapshot(
            snapshot_id=new_snapshot_id(),
            symbol="EURUSD",
            timestamp_utc=ts,
            base_timeframe=Timeframe.M1,
            instrument=instrument,
            session_context=_session(ts),
            latest_bar=bar,
            bars_m1=[bar],
            bars_m5=[],
            bars_m15=[],
            feature_vector=_fv(ts, vwap=_VWAP),
            quality_report=_quality(),
            snapshot_version=SNAPSHOT_VERSION,
        )
        snaps.append(snap)
    return snaps


# ─────────────────────────────────────────────────────────────────────────────
# Engine and label config
# ─────────────────────────────────────────────────────────────────────────────


def make_engine() -> VWAPFadeEngine:
    defn = VWAPFadeDefinition(
        strategy_id="vwap_fade_london_v1",
        session_name="LONDON",
        min_distance_to_vwap_pips=10.0,
        max_distance_to_vwap_pips=50.0,
    )
    return VWAPFadeEngine(defn, min_quality_score=MIN_QUALITY_SCORE)


_LABEL_CFG = LabelConfig(stop_pips=10.0, target_pips=20.0, max_bars=30)


# ─────────────────────────────────────────────────────────────────────────────
# Core replay invariants
# ─────────────────────────────────────────────────────────────────────────────


def test_record_count_equals_snapshot_count():
    snaps = build_vwap_snapshots()
    result = run_replay(snaps, make_engine(), label_config=_LABEL_CFG)
    assert len(result.records) == len(snaps)


def test_bar_index_is_sequential():
    snaps = build_vwap_snapshots(n=20)
    result = run_replay(snaps, make_engine(), label_config=_LABEL_CFG)
    indices = [r.bar_index for r in result.records]
    assert indices == list(range(20))


def test_no_trade_count():
    """First 30 snapshots → EXTENSION_TOO_SMALL → 30 no-trade records."""
    snaps = build_vwap_snapshots()
    result = run_replay(snaps, make_engine(), label_config=_LABEL_CFG)
    no_trades = [
        r for r in result.records
        if r.evaluation_result.outcome == StrategyOutcome.NO_TRADE
    ]
    assert len(no_trades) == 30


def test_candidate_count():
    snaps = build_vwap_snapshots()
    result = run_replay(snaps, make_engine(), label_config=_LABEL_CFG)
    candidates = [
        r for r in result.records
        if r.evaluation_result.outcome == StrategyOutcome.CANDIDATE
    ]
    assert len(candidates) == 70


def test_no_trade_reason_code_extension_too_small():
    snaps = build_vwap_snapshots()
    result = run_replay(snaps, make_engine(), label_config=_LABEL_CFG)
    no_trade_codes = {
        r.evaluation_result.no_trade.reason_code
        for r in result.records
        if r.evaluation_result.outcome == StrategyOutcome.NO_TRADE
    }
    assert no_trade_codes == {"EXTENSION_TOO_SMALL"}


def test_all_candidates_are_short_direction():
    snaps = build_vwap_snapshots()
    result = run_replay(snaps, make_engine(), label_config=_LABEL_CFG)
    for r in result.records:
        if r.evaluation_result.outcome == StrategyOutcome.CANDIDATE:
            assert r.evaluation_result.candidate.direction == TradeDirection.SHORT


# ─────────────────────────────────────────────────────────────────────────────
# Labeling outcomes
# ─────────────────────────────────────────────────────────────────────────────


def test_label_count_equals_candidate_count():
    snaps = build_vwap_snapshots()
    result = run_replay(snaps, make_engine(), label_config=_LABEL_CFG)
    candidates = sum(
        1 for r in result.records
        if r.evaluation_result.outcome == StrategyOutcome.CANDIDATE
    )
    assert len(result.labeled_outcomes) == candidates


def test_win_count():
    """69 of 70 candidates produce WIN; last has no future bars."""
    snaps = build_vwap_snapshots()
    result = run_replay(snaps, make_engine(), label_config=_LABEL_CFG)
    wins = [lbl for lbl in result.labeled_outcomes if lbl.outcome == LabelOutcome.WIN]
    assert len(wins) == 69


def test_last_candidate_not_activated():
    """Snapshot 99 has no future bars → ENTRY_NOT_ACTIVATED."""
    snaps = build_vwap_snapshots()
    result = run_replay(snaps, make_engine(), label_config=_LABEL_CFG)
    not_activated = [
        lbl for lbl in result.labeled_outcomes
        if lbl.outcome == LabelOutcome.ENTRY_NOT_ACTIVATED
    ]
    assert len(not_activated) == 1


def test_win_labels_have_correct_direction():
    snaps = build_vwap_snapshots()
    result = run_replay(snaps, make_engine(), label_config=_LABEL_CFG)
    for lbl in result.labeled_outcomes:
        if lbl.outcome == LabelOutcome.WIN:
            assert lbl.direction == TradeDirection.SHORT


def test_no_labels_when_no_label_config():
    snaps = build_vwap_snapshots()
    result = run_replay(snaps, make_engine())
    assert result.labeled_outcomes == []


# ─────────────────────────────────────────────────────────────────────────────
# Summary counts
# ─────────────────────────────────────────────────────────────────────────────


def test_summary_candidate_count():
    snaps = build_vwap_snapshots()
    result = run_replay(snaps, make_engine(), label_config=_LABEL_CFG)
    assert result.summary.total_candidates == 70


def test_summary_no_trade_count():
    snaps = build_vwap_snapshots()
    result = run_replay(snaps, make_engine(), label_config=_LABEL_CFG)
    assert result.summary.total_no_trade == 30


def test_summary_label_wins():
    snaps = build_vwap_snapshots()
    result = run_replay(snaps, make_engine(), label_config=_LABEL_CFG)
    assert result.summary.label_wins == 69


# ─────────────────────────────────────────────────────────────────────────────
# Analytics integration
# ─────────────────────────────────────────────────────────────────────────────


def test_metrics_win_rate_on_activated():
    snaps = build_vwap_snapshots()
    result = run_replay(snaps, make_engine(), label_config=_LABEL_CFG)
    m = compute_metrics(result.records, result.labeled_outcomes)
    # 69 wins / 69 activated = 100%
    assert m.win_rate_on_activated == pytest.approx(1.0)


def test_metrics_activation_rate():
    snaps = build_vwap_snapshots()
    result = run_replay(snaps, make_engine(), label_config=_LABEL_CFG)
    m = compute_metrics(result.records, result.labeled_outcomes)
    # 69 activated / 70 candidates
    assert m.activation_rate == pytest.approx(69 / 70, abs=1e-4)


def test_report_session_breakdown_london():
    snaps = build_vwap_snapshots()
    result = run_replay(snaps, make_engine(), label_config=_LABEL_CFG)
    report = build_report(result)
    assert len(report.by_session) == 1
    row = report.by_session[0]
    assert row.group_key == "LONDON"
    assert row.candidate_count == 70


def test_report_top_reason_code_is_extension_too_small():
    snaps = build_vwap_snapshots()
    result = run_replay(snaps, make_engine(), label_config=_LABEL_CFG)
    report = build_report(result)
    assert report.top_reason_codes[0][0] == "EXTENSION_TOO_SMALL"
    assert report.top_reason_codes[0][1] == 30


# ─────────────────────────────────────────────────────────────────────────────
# Journal round-trip
# ─────────────────────────────────────────────────────────────────────────────


def test_journal_round_trip(tmp_path: Path):
    snaps = build_vwap_snapshots(n=20)
    result = run_replay(snaps, make_engine(), label_config=_LABEL_CFG)

    journal = CandidateJournal()
    for rec in result.records:
        journal.add_record(rec)
    for lbl in result.labeled_outcomes:
        journal.add_label(lbl)

    rec_path = tmp_path / "records.jsonl"
    lbl_path = tmp_path / "labels.jsonl"
    journal.save_records_jsonl(rec_path)
    journal.save_labels_jsonl(lbl_path)

    loaded_records = CandidateJournal.load_records_jsonl(rec_path)
    loaded_labels = CandidateJournal.load_labels_jsonl(lbl_path)

    assert len(loaded_records) == len(result.records)
    assert len(loaded_labels) == len(result.labeled_outcomes)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison: VWAP Fade vs OR on VWAP-optimised data
# ─────────────────────────────────────────────────────────────────────────────


def test_or_engine_produces_different_candidates_on_vwap_data():
    """
    OpeningRangeEngine on VWAP-fade data:
      OR completed, range 1.1000-1.1020 → LONG candidate at 1.1020.
      Entry activation: bar.high=1.1030 >= 1.1020 → activated.
      Stop: bar.low=1.0995 <= stop=1.1010 → STOP HIT → LOSS.
    This test documents the expected cross-strategy behaviour.
    """
    from aion.replay.models import LabelOutcome as LO
    from aion.strategies.models import OpeningRangeDefinition
    from aion.strategies.opening_range import OpeningRangeEngine

    # Add OR-required context to snapshots
    snaps = build_vwap_snapshots()
    # Add opening_range_completed=True to session context (already done in builder)
    or_engine = OpeningRangeEngine(
        OpeningRangeDefinition(
            strategy_id="or_london_v1",
            session_name="LONDON",
            min_range_pips=5.0,
            max_range_pips=40.0,
        )
    )
    or_label_cfg = LabelConfig(stop_pips=10.0, target_pips=20.0, max_bars=30)
    or_result = run_replay(snaps, or_engine, label_config=or_label_cfg)

    # OR produces candidates on all 100 snapshots (OR is completed and valid)
    or_candidates = sum(
        1 for r in or_result.records
        if r.evaluation_result.outcome == StrategyOutcome.CANDIDATE
    )
    assert or_candidates == 100  # OR fires on all (OR always completed here)

    # With bar.low=1.0995, OR LONG stop=1.1010 gets hit → LOSS on most
    or_losses = sum(
        1 for lbl in or_result.labeled_outcomes
        if lbl.outcome == LO.LOSS
    )
    assert or_losses > 0  # OR breakout fails on reversion data
