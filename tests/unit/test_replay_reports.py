"""
tests/unit/test_replay_reports.py
────────────────────────────────────
Unit tests for aion.analytics.replay_reports.

Tests verify:
  - breakdown_by_regime groups records correctly
  - breakdown_by_regime win_rates computed from linked labels
  - breakdown_by_regime handles regime=None as "UNKNOWN"
  - breakdown_by_session groups only CANDIDATE records
  - breakdown_by_session links labels via setup_id
  - breakdown_by_reason_code sorted by count descending
  - breakdown_by_direction groups labels by direction
  - build_report is consistent with individual breakdowns
  - top_reason_codes sorted descending
  - BreakdownRow is frozen
  - ReplayReport is frozen
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aion.analytics.replay_models import BreakdownRow, ReplayReport
from aion.analytics.replay_reports import (
    breakdown_by_direction,
    breakdown_by_reason_code,
    breakdown_by_regime,
    breakdown_by_session,
    build_report,
)
from aion.core.enums import RegimeLabel, TradeDirection
from aion.replay.models import (
    LabelOutcome,
    LabeledCandidateOutcome,
    ReplayEvaluationRecord,
    ReplayRunResult,
    ReplayRunSummary,
)
from aion.strategies.models import (
    CandidateSetup,
    NoTradeDecision,
    StrategyEvaluationResult,
    StrategyOutcome,
)

_UTC = timezone.utc
_TS = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_UTC)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _candidate_setup(setup_id: str, session: str = "LONDON") -> CandidateSetup:
    return CandidateSetup(
        setup_id=setup_id,
        strategy_id="or_v1",
        strategy_version="1.0.0",
        symbol="EURUSD",
        timestamp_utc=_TS,
        direction=TradeDirection.LONG,
        entry_reference=1.1020,
        range_high=1.1020,
        range_low=1.1000,
        range_size_pips=20.0,
        session_name=session,
        quality_score=1.0,
        atr_14=0.00015,
    )


def _candidate_record(
    bar_index: int,
    setup_id: str,
    session: str = "LONDON",
    regime: RegimeLabel | None = None,
) -> ReplayEvaluationRecord:
    return ReplayEvaluationRecord(
        bar_index=bar_index,
        snapshot_id=f"snap_{bar_index:04d}",
        symbol="EURUSD",
        timestamp_utc=_TS,
        evaluation_result=StrategyEvaluationResult(
            outcome=StrategyOutcome.CANDIDATE,
            strategy_id="or_v1",
            symbol="EURUSD",
            timestamp_utc=_TS,
            candidate=_candidate_setup(setup_id, session),
        ),
        regime_label=regime,
    )


def _no_trade_record(
    bar_index: int,
    reason: str = "OR_NOT_COMPLETED",
    regime: RegimeLabel | None = None,
) -> ReplayEvaluationRecord:
    return ReplayEvaluationRecord(
        bar_index=bar_index,
        snapshot_id=f"snap_{bar_index:04d}",
        symbol="EURUSD",
        timestamp_utc=_TS,
        evaluation_result=StrategyEvaluationResult(
            outcome=StrategyOutcome.NO_TRADE,
            strategy_id="or_v1",
            symbol="EURUSD",
            timestamp_utc=_TS,
            no_trade=NoTradeDecision(
                strategy_id="or_v1",
                symbol="EURUSD",
                timestamp_utc=_TS,
                reason_code=reason,
                reason_detail="detail",
            ),
        ),
        regime_label=regime,
    )


def _win_label(setup_id: str, direction: TradeDirection = TradeDirection.LONG) -> LabeledCandidateOutcome:
    return LabeledCandidateOutcome(
        setup_id=setup_id,
        symbol="EURUSD",
        timestamp_utc=_TS,
        direction=direction,
        entry_reference=1.1020,
        stop_price=1.1010,
        target_price=1.1040,
        outcome=LabelOutcome.WIN,
        entry_activated=True,
        bars_to_entry=0,
        bars_to_resolution=2,
        mfe_pips=25.0,
        mae_pips=0.0,
        pnl_pips=20.0,
    )


def _not_activated_label(setup_id: str) -> LabeledCandidateOutcome:
    return LabeledCandidateOutcome(
        setup_id=setup_id,
        symbol="EURUSD",
        timestamp_utc=_TS,
        direction=TradeDirection.LONG,
        entry_reference=1.1020,
        stop_price=1.1010,
        target_price=1.1040,
        outcome=LabelOutcome.ENTRY_NOT_ACTIVATED,
        entry_activated=False,
        bars_to_entry=None,
        bars_to_resolution=None,
        mfe_pips=None,
        mae_pips=None,
        pnl_pips=None,
    )


def _dummy_summary(n: int = 0) -> ReplayRunSummary:
    return ReplayRunSummary(
        run_id="run_test",
        strategy_id="or_v1",
        symbol="EURUSD",
        total_snapshots=n,
        total_candidates=0,
        total_no_trade=0,
        total_insufficient_data=0,
        total_labeled=0,
        label_wins=0,
        label_losses=0,
        label_timeouts=0,
        label_not_activated=0,
        start_time_utc=_TS,
        end_time_utc=_TS,
        elapsed_seconds=0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# breakdown_by_regime
# ─────────────────────────────────────────────────────────────────────────────


def test_regime_breakdown_groups_all_records():
    records = [
        _candidate_record(0, "s1", regime=RegimeLabel.RANGE),
        _no_trade_record(1, regime=RegimeLabel.RANGE),
        _candidate_record(2, "s2", regime=RegimeLabel.TREND_UP),
    ]
    rows = breakdown_by_regime(records, [])
    keys = {r.group_key for r in rows}
    assert keys == {"RANGE", "TREND_UP"}


def test_regime_breakdown_record_counts():
    records = [
        _candidate_record(0, "s1", regime=RegimeLabel.RANGE),
        _no_trade_record(1, regime=RegimeLabel.RANGE),
        _no_trade_record(2, regime=RegimeLabel.RANGE),
    ]
    rows = breakdown_by_regime(records, [])
    range_row = next(r for r in rows if r.group_key == "RANGE")
    assert range_row.record_count == 3
    assert range_row.candidate_count == 1
    assert range_row.no_trade_count == 2


def test_regime_breakdown_none_regime_grouped_as_unknown():
    records = [_no_trade_record(0, regime=None)]
    rows = breakdown_by_regime(records, [])
    assert rows[0].group_key == "UNKNOWN"


def test_regime_breakdown_win_rate_via_label_index():
    records = [
        _candidate_record(0, "s1", regime=RegimeLabel.RANGE),
        _candidate_record(1, "s2", regime=RegimeLabel.RANGE),
    ]
    labels = [_win_label("s1"), _not_activated_label("s2")]
    rows = breakdown_by_regime(records, labels)
    row = rows[0]
    assert row.entry_activated_count == 1
    assert row.win_count == 1
    assert row.win_rate == pytest.approx(1.0)


def test_regime_breakdown_no_labels_win_rate_none():
    records = [_candidate_record(0, "s1", regime=RegimeLabel.COMPRESSION)]
    rows = breakdown_by_regime(records, [])
    # win_rate is None when no activations (denominator = 0)
    assert rows[0].win_rate is None
    # activation_rate is 0.0 when candidates exist but none activated
    assert rows[0].activation_rate == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# breakdown_by_session
# ─────────────────────────────────────────────────────────────────────────────


def test_session_breakdown_only_candidates():
    records = [
        _candidate_record(0, "s1", session="LONDON"),
        _no_trade_record(1),
        _no_trade_record(2),
    ]
    rows = breakdown_by_session(records, [])
    assert len(rows) == 1
    assert rows[0].group_key == "LONDON"


def test_session_breakdown_no_trade_count_is_zero():
    records = [_candidate_record(0, "s1", session="LONDON")]
    rows = breakdown_by_session(records, [])
    assert rows[0].no_trade_count == 0


def test_session_breakdown_multiple_sessions():
    records = [
        _candidate_record(0, "s1", session="LONDON"),
        _candidate_record(1, "s2", session="NEW_YORK"),
        _candidate_record(2, "s3", session="LONDON"),
    ]
    rows = breakdown_by_session(records, [])
    keys = {r.group_key for r in rows}
    assert keys == {"LONDON", "NEW_YORK"}
    london = next(r for r in rows if r.group_key == "LONDON")
    assert london.candidate_count == 2


def test_session_breakdown_links_labels():
    records = [_candidate_record(0, "s1", session="LONDON")]
    labels = [_win_label("s1")]
    rows = breakdown_by_session(records, labels)
    assert rows[0].win_count == 1
    assert rows[0].win_rate == pytest.approx(1.0)


def test_session_breakdown_empty_when_no_candidates():
    records = [_no_trade_record(0), _no_trade_record(1)]
    rows = breakdown_by_session(records, [])
    assert rows == []


# ─────────────────────────────────────────────────────────────────────────────
# breakdown_by_reason_code
# ─────────────────────────────────────────────────────────────────────────────


def test_reason_code_breakdown_counts():
    records = [
        _no_trade_record(0, reason="OR_NOT_COMPLETED"),
        _no_trade_record(1, reason="OR_NOT_COMPLETED"),
        _no_trade_record(2, reason="RANGE_TOO_TIGHT"),
    ]
    rows = breakdown_by_reason_code(records)
    keys = {r.group_key for r in rows}
    assert keys == {"OR_NOT_COMPLETED", "RANGE_TOO_TIGHT"}
    or_row = next(r for r in rows if r.group_key == "OR_NOT_COMPLETED")
    assert or_row.no_trade_count == 2


def test_reason_code_breakdown_sorted_desc():
    records = [
        _no_trade_record(0, "A"),
        _no_trade_record(1, "B"),
        _no_trade_record(2, "B"),
        _no_trade_record(3, "B"),
        _no_trade_record(4, "A"),
    ]
    rows = breakdown_by_reason_code(records)
    assert rows[0].group_key == "B"
    assert rows[1].group_key == "A"


def test_reason_code_excludes_candidate_records():
    records = [
        _no_trade_record(0, "X"),
        _candidate_record(1, "s1"),
    ]
    rows = breakdown_by_reason_code(records)
    assert len(rows) == 1
    assert rows[0].group_key == "X"


def test_reason_code_empty_when_no_no_trade():
    records = [_candidate_record(0, "s1")]
    rows = breakdown_by_reason_code(records)
    assert rows == []


# ─────────────────────────────────────────────────────────────────────────────
# breakdown_by_direction
# ─────────────────────────────────────────────────────────────────────────────


def test_direction_breakdown_long_only():
    labels = [_win_label("s1"), _win_label("s2"), _win_label("s3")]
    rows = breakdown_by_direction(labels)
    assert len(rows) == 1
    assert rows[0].group_key == "LONG"
    assert rows[0].win_count == 3


def test_direction_breakdown_short_separate():
    labels = [
        _win_label("s1", TradeDirection.LONG),
        _win_label("s2", TradeDirection.SHORT),
    ]
    rows = breakdown_by_direction(labels)
    keys = {r.group_key for r in rows}
    assert keys == {"LONG", "SHORT"}


def test_direction_breakdown_no_trade_zero():
    labels = [_win_label("s1")]
    rows = breakdown_by_direction(labels)
    assert rows[0].no_trade_count == 0


def test_direction_breakdown_empty_when_no_labels():
    rows = breakdown_by_direction([])
    assert rows == []


def test_direction_activation_rate():
    labels = [
        _win_label("s1"),
        _not_activated_label("s2"),
    ]
    rows = breakdown_by_direction(labels)
    row = rows[0]
    assert row.activation_rate == pytest.approx(0.5)
    assert row.win_rate == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# build_report
# ─────────────────────────────────────────────────────────────────────────────


def test_build_report_total_records_consistent():
    records = [
        _candidate_record(0, "s1"),
        _no_trade_record(1),
        _no_trade_record(2),
    ]
    result = ReplayRunResult(
        summary=_dummy_summary(3),
        records=records,
        labeled_outcomes=[_win_label("s1")],
    )
    report = build_report(result)
    assert report.total_records == 3


def test_build_report_overall_metrics_candidate_count():
    records = [_candidate_record(0, "s1"), _no_trade_record(1)]
    result = ReplayRunResult(
        summary=_dummy_summary(2),
        records=records,
        labeled_outcomes=[_win_label("s1")],
    )
    report = build_report(result)
    assert report.overall_metrics.candidate_count == 1
    assert report.overall_metrics.win_count == 1


def test_build_report_top_reason_codes_sorted():
    records = [
        _no_trade_record(0, "A"),
        _no_trade_record(1, "B"),
        _no_trade_record(2, "B"),
        _no_trade_record(3, "B"),
    ]
    result = ReplayRunResult(
        summary=_dummy_summary(4),
        records=records,
        labeled_outcomes=[],
    )
    report = build_report(result)
    codes = [code for code, _ in report.top_reason_codes]
    assert codes[0] == "B"
    assert codes[1] == "A"


def test_build_report_is_frozen():
    result = ReplayRunResult(
        summary=_dummy_summary(),
        records=[],
        labeled_outcomes=[],
    )
    report = build_report(result)
    with pytest.raises(Exception):
        report.total_records = 99  # type: ignore[misc]


def test_breakdown_row_is_frozen():
    rows = breakdown_by_reason_code([_no_trade_record(0, "X")])
    with pytest.raises(Exception):
        rows[0].win_count = 99  # type: ignore[misc]


def test_build_report_serialises():
    result = ReplayRunResult(
        summary=_dummy_summary(),
        records=[_no_trade_record(0)],
        labeled_outcomes=[],
    )
    report = build_report(result)
    js = report.model_dump_json()
    assert "overall_metrics" in js
