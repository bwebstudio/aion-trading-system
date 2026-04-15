"""
tests/unit/test_journal.py
────────────────────────────
Unit tests for aion.replay.journal.CandidateJournal.

Tests verify:
  - Empty journal has zero records and labels
  - add_record appends to records list
  - add_label appends to labels list
  - candidates() returns only CANDIDATE outcome records
  - no_trades() returns only NO_TRADE outcome records
  - insufficient_data() returns only INSUFFICIENT_DATA records
  - labeled_outcomes() returns all labels
  - __len__ counts records (not labels)
  - records() and labeled_outcomes() return copies (no mutation)
  - save_records_jsonl / load_records_jsonl round-trip
  - save_labels_jsonl / load_labels_jsonl round-trip
  - load returns empty list for non-existent file
  - JSONL files have one line per record
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from aion.core.enums import TradeDirection
from aion.replay.journal import CandidateJournal
from aion.replay.models import (
    LabelConfig,
    LabelOutcome,
    LabeledCandidateOutcome,
    ReplayEvaluationRecord,
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


def _candidate_result() -> StrategyEvaluationResult:
    return StrategyEvaluationResult(
        outcome=StrategyOutcome.CANDIDATE,
        strategy_id="or_london_v1",
        symbol="EURUSD",
        timestamp_utc=_TS,
        candidate=CandidateSetup(
            strategy_id="or_london_v1",
            strategy_version="1.0.0",
            symbol="EURUSD",
            timestamp_utc=_TS,
            direction=TradeDirection.LONG,
            entry_reference=1.1020,
            range_high=1.1020,
            range_low=1.1000,
            range_size_pips=20.0,
            session_name="LONDON",
            quality_score=1.0,
            atr_14=0.00015,
        ),
    )


def _no_trade_result(reason_code: str = "NOT_IN_TARGET_SESSION") -> StrategyEvaluationResult:
    return StrategyEvaluationResult(
        outcome=StrategyOutcome.NO_TRADE,
        strategy_id="or_london_v1",
        symbol="EURUSD",
        timestamp_utc=_TS,
        no_trade=NoTradeDecision(
            strategy_id="or_london_v1",
            symbol="EURUSD",
            timestamp_utc=_TS,
            reason_code=reason_code,
            reason_detail="Detail.",
        ),
    )


def _insufficient_result() -> StrategyEvaluationResult:
    return StrategyEvaluationResult(
        outcome=StrategyOutcome.INSUFFICIENT_DATA,
        strategy_id="or_london_v1",
        symbol="EURUSD",
        timestamp_utc=_TS,
        reason_detail="Low quality.",
    )


def _make_record(
    bar_index: int = 0,
    result: StrategyEvaluationResult | None = None,
) -> ReplayEvaluationRecord:
    return ReplayEvaluationRecord(
        bar_index=bar_index,
        snapshot_id=f"snap_{bar_index:04d}",
        symbol="EURUSD",
        timestamp_utc=_TS,
        evaluation_result=result or _no_trade_result(),
    )


def _make_label(setup_id: str = "setup_abc") -> LabeledCandidateOutcome:
    return LabeledCandidateOutcome(
        setup_id=setup_id,
        symbol="EURUSD",
        timestamp_utc=_TS,
        direction=TradeDirection.LONG,
        entry_reference=1.1020,
        stop_price=1.1010,
        target_price=1.1040,
        outcome=LabelOutcome.WIN,
        entry_activated=True,
        bars_to_entry=0,
        bars_to_resolution=2,
        mfe_pips=25.0,
        mae_pips=5.0,
        pnl_pips=20.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Empty journal
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_journal_has_no_records():
    j = CandidateJournal()
    assert j.records() == []


def test_empty_journal_has_no_labels():
    j = CandidateJournal()
    assert j.labeled_outcomes() == []


def test_empty_journal_len_is_zero():
    j = CandidateJournal()
    assert len(j) == 0


# ─────────────────────────────────────────────────────────────────────────────
# add_record / add_label
# ─────────────────────────────────────────────────────────────────────────────


def test_add_record_increases_len():
    j = CandidateJournal()
    j.add_record(_make_record())
    assert len(j) == 1


def test_add_multiple_records():
    j = CandidateJournal()
    for i in range(5):
        j.add_record(_make_record(bar_index=i))
    assert len(j) == 5


def test_add_label_appears_in_labeled_outcomes():
    j = CandidateJournal()
    j.add_label(_make_label("setup_x"))
    assert len(j.labeled_outcomes()) == 1
    assert j.labeled_outcomes()[0].setup_id == "setup_x"


def test_len_counts_records_not_labels():
    j = CandidateJournal()
    j.add_record(_make_record())
    j.add_label(_make_label())
    assert len(j) == 1  # only records


# ─────────────────────────────────────────────────────────────────────────────
# Accessors / filters
# ─────────────────────────────────────────────────────────────────────────────


def test_candidates_filter():
    j = CandidateJournal()
    j.add_record(_make_record(0, _candidate_result()))
    j.add_record(_make_record(1, _no_trade_result()))
    j.add_record(_make_record(2, _candidate_result()))
    assert len(j.candidates()) == 2


def test_no_trades_filter():
    j = CandidateJournal()
    j.add_record(_make_record(0, _candidate_result()))
    j.add_record(_make_record(1, _no_trade_result()))
    j.add_record(_make_record(2, _no_trade_result()))
    assert len(j.no_trades()) == 2


def test_insufficient_data_filter():
    j = CandidateJournal()
    j.add_record(_make_record(0, _insufficient_result()))
    j.add_record(_make_record(1, _candidate_result()))
    assert len(j.insufficient_data()) == 1


def test_records_returns_copy():
    """Mutating the returned list does not affect the journal."""
    j = CandidateJournal()
    j.add_record(_make_record())
    lst = j.records()
    lst.clear()
    assert len(j) == 1


def test_labeled_outcomes_returns_copy():
    j = CandidateJournal()
    j.add_label(_make_label())
    lst = j.labeled_outcomes()
    lst.clear()
    assert len(j.labeled_outcomes()) == 1


# ─────────────────────────────────────────────────────────────────────────────
# JSONL persistence — records
# ─────────────────────────────────────────────────────────────────────────────


def test_save_load_records_jsonl_round_trip(tmp_path: Path):
    j = CandidateJournal()
    j.add_record(_make_record(0, _candidate_result()))
    j.add_record(_make_record(1, _no_trade_result("RANGE_TOO_TIGHT")))

    path = tmp_path / "records.jsonl"
    j.save_records_jsonl(path)

    loaded = CandidateJournal.load_records_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0].bar_index == 0
    assert loaded[0].evaluation_result.outcome == StrategyOutcome.CANDIDATE
    assert loaded[1].bar_index == 1
    assert loaded[1].evaluation_result.no_trade.reason_code == "RANGE_TOO_TIGHT"


def test_records_jsonl_has_one_line_per_record(tmp_path: Path):
    j = CandidateJournal()
    for i in range(3):
        j.add_record(_make_record(i))

    path = tmp_path / "records.jsonl"
    j.save_records_jsonl(path)

    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 3


def test_load_records_returns_empty_for_missing_file(tmp_path: Path):
    result = CandidateJournal.load_records_jsonl(tmp_path / "nonexistent.jsonl")
    assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# JSONL persistence — labels
# ─────────────────────────────────────────────────────────────────────────────


def test_save_load_labels_jsonl_round_trip(tmp_path: Path):
    j = CandidateJournal()
    j.add_label(_make_label("s1"))
    j.add_label(_make_label("s2"))

    path = tmp_path / "labels.jsonl"
    j.save_labels_jsonl(path)

    loaded = CandidateJournal.load_labels_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0].setup_id == "s1"
    assert loaded[1].setup_id == "s2"
    assert loaded[0].outcome == LabelOutcome.WIN


def test_labels_jsonl_has_one_line_per_label(tmp_path: Path):
    j = CandidateJournal()
    for i in range(4):
        j.add_label(_make_label(f"s{i}"))

    path = tmp_path / "labels.jsonl"
    j.save_labels_jsonl(path)

    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 4


def test_load_labels_returns_empty_for_missing_file(tmp_path: Path):
    result = CandidateJournal.load_labels_jsonl(tmp_path / "nonexistent.jsonl")
    assert result == []


def test_save_creates_parent_directories(tmp_path: Path):
    j = CandidateJournal()
    j.add_record(_make_record())
    nested = tmp_path / "a" / "b" / "c" / "records.jsonl"
    j.save_records_jsonl(nested)
    assert nested.exists()
