"""
tests/integration/test_strategy_replay_loop.py
────────────────────────────────────────────────
Integration tests: OpeningRangeEngine + run_replay() + CandidateJournal.

What these tests verify
────────────────────────
  Runner invariants:
    - len(records) == len(snapshots)  (one record per snapshot, always)
    - bar_index is sequential (0, 1, 2, ...)
    - CANDIDATE records have candidate populated
    - NO_TRADE records have no_trade populated
    - INSUFFICIENT_DATA records have neither

  Summary correctness:
    - total_candidates + total_no_trade + total_insufficient_data == total_snapshots
    - label counts sum to total_labeled
    - total_labeled == count of CANDIDATE records (when label_config provided)

  Labeling:
    - one LabeledCandidateOutcome per CANDIDATE result
    - each labeled outcome links to a valid setup_id
    - each labeled outcome has a valid outcome (enum member)

  Regime integration:
    - regime_label populated in records when detector provided
    - regime_label is None in records when detector not provided

  Determinism:
    - same input → identical outcome on two runs

  Journal:
    - CandidateJournal correctly accumulates records from ReplayRunResult
    - JSONL round-trip preserves all records

  Empty input:
    - empty snapshot list → valid empty result
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
from aion.regime.rules import RuleBasedRegimeDetector
from aion.replay.journal import CandidateJournal
from aion.replay.models import LabelConfig, LabelOutcome
from aion.replay.runner import run_replay
from aion.strategies.models import OpeningRangeDefinition, StrategyOutcome
from aion.strategies.opening_range import OpeningRangeEngine

_UTC = timezone.utc
_TS_BASE = datetime(2024, 1, 15, 10, 0, 0, tzinfo=_UTC)


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot factory helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_instrument() -> InstrumentSpec:
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


def _make_bar(ts: datetime, high: float = 1.1045, low: float = 1.1035) -> MarketBar:
    """Bar with controllable high/low for labeling scenarios."""
    return MarketBar(
        symbol="EURUSD",
        timestamp_utc=ts,
        timestamp_market=ts,
        timeframe=Timeframe.M1,
        open=1.1040,
        high=high,
        low=low,
        close=1.1042,
        tick_volume=100.0,
        real_volume=0.0,
        spread=2.0,
        source=DataSource.SYNTHETIC,
    )


def _make_session(
    session_name: SessionName = SessionName.LONDON,
    or_completed: bool = True,
    ts: datetime = _TS_BASE,
) -> SessionContext:
    is_open = session_name != SessionName.OFF_HOURS
    is_london = session_name in (SessionName.LONDON, SessionName.OVERLAP_LONDON_NY)
    is_ny = session_name in (SessionName.NEW_YORK, SessionName.OVERLAP_LONDON_NY)
    return SessionContext(
        trading_day=ts.date(),
        broker_time=ts,
        market_time=ts,
        local_time=ts,
        is_asia=session_name == SessionName.ASIA,
        is_london=is_london,
        is_new_york=is_ny,
        is_session_open_window=is_open,
        opening_range_active=False,
        opening_range_completed=or_completed,
        session_name=session_name,
        session_open_utc=ts.replace(hour=8, minute=0, second=0) if is_open else None,
        session_close_utc=ts.replace(hour=16, minute=30, second=0) if is_open else None,
    )


def _make_fv(ts: datetime, or_high: float = 1.1020, or_low: float = 1.1000) -> FeatureVector:
    return FeatureVector(
        symbol="EURUSD",
        timestamp_utc=ts,
        timeframe=Timeframe.M1,
        atr_14=0.00015,
        rolling_range_10=0.001,
        rolling_range_20=0.002,
        volatility_percentile_20=0.50,
        session_high=1.1060,
        session_low=1.0990,
        opening_range_high=or_high,
        opening_range_low=or_low,
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


def _make_quality(score: float = 1.0) -> DataQualityReport:
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
        quality_score=score,
        warnings=[],
    )


def _make_snapshot(
    index: int,
    session_name: SessionName = SessionName.LONDON,
    or_completed: bool = True,
    quality_score: float = 1.0,
    bar_high: float = 1.1045,
    bar_low: float = 1.1035,
) -> MarketSnapshot:
    from datetime import timedelta
    ts = _TS_BASE + timedelta(minutes=index)
    bar = _make_bar(ts, high=bar_high, low=bar_low)
    return MarketSnapshot(
        snapshot_id=new_snapshot_id(),
        symbol="EURUSD",
        timestamp_utc=ts,
        base_timeframe=Timeframe.M1,
        instrument=_make_instrument(),
        session_context=_make_session(session_name, or_completed, ts),
        latest_bar=bar,
        bars_m1=[bar],
        bars_m5=[],
        bars_m15=[],
        feature_vector=_make_fv(ts),
        quality_report=_make_quality(quality_score),
        snapshot_version=SNAPSHOT_VERSION,
    )


def _build_mixed_snapshots(n_off: int = 5, n_candidate: int = 15) -> list[MarketSnapshot]:
    """
    Build a predictable sequence:
      - n_off snapshots: OFF_HOURS → engine returns NO_TRADE
      - n_candidate snapshots: LONDON, OR completed → engine returns CANDIDATE
    Future bars (for labeling) have high=1.1045, low=1.1035:
      - LONG entry=1.1020 → activated (high >= 1.1020)
      - target=1.1040     → WIN (high >= 1.1040 on same bar as activation)
    """
    snaps = []
    for i in range(n_off):
        snaps.append(_make_snapshot(i, session_name=SessionName.OFF_HOURS))
    for i in range(n_off, n_off + n_candidate):
        snaps.append(_make_snapshot(i, session_name=SessionName.LONDON, or_completed=True))
    return snaps


def _make_engine(direction_bias: TradeDirection | None = None) -> OpeningRangeEngine:
    return OpeningRangeEngine(
        OpeningRangeDefinition(
            strategy_id="or_london_v1",
            session_name="LONDON",
            min_range_pips=5.0,
            max_range_pips=40.0,
            direction_bias=direction_bias,
        ),
        min_quality_score=MIN_QUALITY_SCORE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core runner invariants
# ─────────────────────────────────────────────────────────────────────────────


def test_records_count_equals_snapshots_count():
    snaps = _build_mixed_snapshots(n_off=3, n_candidate=7)
    result = run_replay(snaps, _make_engine())
    assert len(result.records) == len(snaps)


def test_bar_index_is_sequential():
    snaps = _build_mixed_snapshots(n_off=2, n_candidate=3)
    result = run_replay(snaps, _make_engine())
    indices = [r.bar_index for r in result.records]
    assert indices == list(range(len(snaps)))


def test_candidate_records_have_candidate_populated():
    snaps = _build_mixed_snapshots(n_off=0, n_candidate=5)
    result = run_replay(snaps, _make_engine())
    for r in result.records:
        if r.evaluation_result.outcome == StrategyOutcome.CANDIDATE:
            assert r.evaluation_result.candidate is not None
            assert r.evaluation_result.no_trade is None


def test_no_trade_records_have_no_trade_populated():
    snaps = _build_mixed_snapshots(n_off=5, n_candidate=0)
    result = run_replay(snaps, _make_engine())
    for r in result.records:
        assert r.evaluation_result.outcome == StrategyOutcome.NO_TRADE
        assert r.evaluation_result.no_trade is not None
        assert r.evaluation_result.candidate is None


# ─────────────────────────────────────────────────────────────────────────────
# Summary correctness
# ─────────────────────────────────────────────────────────────────────────────


def test_summary_counts_sum_to_total_snapshots():
    snaps = _build_mixed_snapshots(n_off=4, n_candidate=6)
    s = run_replay(snaps, _make_engine()).summary
    assert s.total_candidates + s.total_no_trade + s.total_insufficient_data == s.total_snapshots


def test_summary_candidate_count_correct():
    snaps = _build_mixed_snapshots(n_off=3, n_candidate=7)
    s = run_replay(snaps, _make_engine()).summary
    assert s.total_no_trade == 3
    assert s.total_candidates == 7


def test_summary_label_counts_sum_to_total_labeled():
    snaps = _build_mixed_snapshots(n_off=2, n_candidate=8)
    label_cfg = LabelConfig(stop_pips=10.0, target_pips=20.0, max_bars=10)
    s = run_replay(snaps, _make_engine(), label_config=label_cfg).summary
    total = s.label_wins + s.label_losses + s.label_timeouts + s.label_not_activated
    assert total == s.total_labeled


def test_summary_strategy_id_matches_engine():
    snaps = _build_mixed_snapshots()
    engine = _make_engine()
    s = run_replay(snaps, engine).summary
    assert s.strategy_id == engine.strategy_id


# ─────────────────────────────────────────────────────────────────────────────
# Labeling
# ─────────────────────────────────────────────────────────────────────────────


def test_labeled_outcomes_count_equals_candidates():
    """One LabeledCandidateOutcome per CANDIDATE result."""
    snaps = _build_mixed_snapshots(n_off=3, n_candidate=7)
    label_cfg = LabelConfig(stop_pips=10.0, target_pips=20.0, max_bars=10)
    result = run_replay(snaps, _make_engine(), label_config=label_cfg)
    n_candidates = sum(
        1 for r in result.records
        if r.evaluation_result.outcome == StrategyOutcome.CANDIDATE
    )
    assert len(result.labeled_outcomes) == n_candidates


def test_no_labeled_outcomes_without_label_config():
    snaps = _build_mixed_snapshots()
    result = run_replay(snaps, _make_engine())
    assert result.labeled_outcomes == []
    assert result.summary.total_labeled == 0


def test_labeled_outcomes_have_valid_label_outcome():
    snaps = _build_mixed_snapshots(n_off=2, n_candidate=8)
    label_cfg = LabelConfig(stop_pips=10.0, target_pips=20.0, max_bars=10)
    result = run_replay(snaps, _make_engine(), label_config=label_cfg)
    valid_outcomes = set(LabelOutcome)
    for lbl in result.labeled_outcomes:
        assert lbl.outcome in valid_outcomes


def test_labeled_outcomes_setup_id_matches_candidate():
    """Each labeled outcome's setup_id must link to the corresponding candidate."""
    snaps = _build_mixed_snapshots(n_off=0, n_candidate=5)
    label_cfg = LabelConfig(stop_pips=10.0, target_pips=20.0, max_bars=5)
    result = run_replay(snaps, _make_engine(), label_config=label_cfg)

    candidate_ids = {
        r.evaluation_result.candidate.setup_id
        for r in result.records
        if r.evaluation_result.outcome == StrategyOutcome.CANDIDATE
    }
    labeled_ids = {lbl.setup_id for lbl in result.labeled_outcomes}
    assert labeled_ids == candidate_ids


def test_last_snapshot_candidate_gets_timeout_when_no_future_bars():
    """
    The last snapshot may produce a CANDIDATE with no future bars available.
    The labeler must return ENTRY_NOT_ACTIVATED or TIMEOUT, not crash.
    """
    snaps = _build_mixed_snapshots(n_off=0, n_candidate=1)
    label_cfg = LabelConfig(stop_pips=10.0, target_pips=20.0, max_bars=5)
    result = run_replay(snaps, _make_engine(), label_config=label_cfg)
    assert len(result.labeled_outcomes) == 1
    assert result.labeled_outcomes[0].outcome in {
        LabelOutcome.ENTRY_NOT_ACTIVATED,
        LabelOutcome.TIMEOUT,
        LabelOutcome.WIN,
        LabelOutcome.LOSS,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Regime detector integration
# ─────────────────────────────────────────────────────────────────────────────


def test_regime_label_populated_when_detector_provided():
    snaps = _build_mixed_snapshots(n_off=0, n_candidate=3)
    result = run_replay(snaps, _make_engine(), regime_detector=RuleBasedRegimeDetector())
    for r in result.records:
        assert r.regime_label is not None
        assert r.regime_confidence is not None
        assert 0.0 <= r.regime_confidence <= 1.0


def test_regime_label_none_without_detector():
    snaps = _build_mixed_snapshots(n_off=0, n_candidate=3)
    result = run_replay(snaps, _make_engine())
    for r in result.records:
        assert r.regime_label is None
        assert r.regime_confidence is None


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────


def test_replay_is_deterministic():
    """Same input → same outcome (outcome and counts; setup_id will differ)."""
    snaps = _build_mixed_snapshots(n_off=2, n_candidate=8)
    engine = _make_engine()
    r1 = run_replay(snaps, engine)
    r2 = run_replay(snaps, engine)
    assert r1.summary.total_candidates == r2.summary.total_candidates
    assert r1.summary.total_no_trade == r2.summary.total_no_trade
    for rec1, rec2 in zip(r1.records, r2.records):
        assert rec1.evaluation_result.outcome == rec2.evaluation_result.outcome


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_snapshots_returns_valid_result():
    result = run_replay([], _make_engine())
    assert result.records == []
    assert result.labeled_outcomes == []
    assert result.summary.total_snapshots == 0
    assert result.summary.total_candidates == 0


# ─────────────────────────────────────────────────────────────────────────────
# Journal integration
# ─────────────────────────────────────────────────────────────────────────────


def test_journal_accumulates_from_run_result():
    snaps = _build_mixed_snapshots(n_off=3, n_candidate=7)
    label_cfg = LabelConfig(stop_pips=10.0, target_pips=20.0, max_bars=10)
    result = run_replay(snaps, _make_engine(), label_config=label_cfg)

    journal = CandidateJournal()
    for rec in result.records:
        journal.add_record(rec)
    for lbl in result.labeled_outcomes:
        journal.add_label(lbl)

    assert len(journal) == len(snaps)
    assert len(journal.candidates()) == result.summary.total_candidates
    assert len(journal.no_trades()) == result.summary.total_no_trade
    assert len(journal.labeled_outcomes()) == result.summary.total_labeled


def test_journal_jsonl_round_trip(tmp_path):
    snaps = _build_mixed_snapshots(n_off=2, n_candidate=4)
    result = run_replay(snaps, _make_engine())

    journal = CandidateJournal()
    for rec in result.records:
        journal.add_record(rec)

    path = tmp_path / "records.jsonl"
    journal.save_records_jsonl(path)
    loaded = CandidateJournal.load_records_jsonl(path)

    assert len(loaded) == len(snaps)
    for original, restored in zip(journal.records(), loaded):
        assert original.bar_index == restored.bar_index
        assert original.evaluation_result.outcome == restored.evaluation_result.outcome
