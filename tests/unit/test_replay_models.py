"""
tests/unit/test_replay_models.py
──────────────────────────────────
Unit tests for aion.replay.models.

Tests verify:
  - LabelOutcome enum values are strings
  - LabelConfig construction and defaults
  - LabelConfig is frozen
  - ReplayEvaluationRecord construction and frozen
  - LabeledCandidateOutcome construction and frozen
  - ReplayRunSummary construction and frozen
  - ReplayRunResult construction and frozen
  - All models serialise to JSON without error
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aion.core.enums import RegimeLabel, TradeDirection
from aion.replay.models import (
    LabelConfig,
    LabelOutcome,
    LabeledCandidateOutcome,
    ReplayEvaluationRecord,
    ReplayRunResult,
    ReplayRunSummary,
)
from aion.strategies.models import (
    NoTradeDecision,
    StrategyEvaluationResult,
    StrategyOutcome,
)

_UTC = timezone.utc
_TS = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_UTC)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_no_trade_result() -> StrategyEvaluationResult:
    return StrategyEvaluationResult(
        outcome=StrategyOutcome.NO_TRADE,
        strategy_id="or_london_v1",
        symbol="EURUSD",
        timestamp_utc=_TS,
        no_trade=NoTradeDecision(
            strategy_id="or_london_v1",
            symbol="EURUSD",
            timestamp_utc=_TS,
            reason_code="NOT_IN_TARGET_SESSION",
            reason_detail="Off hours.",
        ),
    )


def _make_record(bar_index: int = 0) -> ReplayEvaluationRecord:
    return ReplayEvaluationRecord(
        bar_index=bar_index,
        snapshot_id="snap_abc123",
        symbol="EURUSD",
        timestamp_utc=_TS,
        evaluation_result=_make_no_trade_result(),
    )


def _make_label(outcome: LabelOutcome = LabelOutcome.WIN) -> LabeledCandidateOutcome:
    return LabeledCandidateOutcome(
        setup_id="setup_xyz",
        symbol="EURUSD",
        timestamp_utc=_TS,
        direction=TradeDirection.LONG,
        entry_reference=1.1020,
        stop_price=1.1010,
        target_price=1.1040,
        outcome=outcome,
        entry_activated=True,
        bars_to_entry=0,
        bars_to_resolution=2,
        mfe_pips=25.0,
        mae_pips=5.0,
        pnl_pips=20.0,
    )


def _make_summary() -> ReplayRunSummary:
    return ReplayRunSummary(
        run_id="run_abc123",
        strategy_id="or_london_v1",
        symbol="EURUSD",
        total_snapshots=100,
        total_candidates=30,
        total_no_trade=60,
        total_insufficient_data=10,
        total_labeled=30,
        label_wins=20,
        label_losses=5,
        label_timeouts=3,
        label_not_activated=2,
        start_time_utc=_TS,
        end_time_utc=_TS,
        elapsed_seconds=1.23,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LabelOutcome enum
# ─────────────────────────────────────────────────────────────────────────────


def test_label_outcome_values_are_strings():
    for outcome in LabelOutcome:
        assert isinstance(outcome.value, str)


def test_label_outcome_has_four_variants():
    values = {o.value for o in LabelOutcome}
    assert values == {"WIN", "LOSS", "TIMEOUT", "ENTRY_NOT_ACTIVATED"}


# ─────────────────────────────────────────────────────────────────────────────
# LabelConfig
# ─────────────────────────────────────────────────────────────────────────────


def test_label_config_builds():
    cfg = LabelConfig(stop_pips=10.0, target_pips=20.0)
    assert cfg.stop_pips == 10.0
    assert cfg.target_pips == 20.0


def test_label_config_defaults():
    cfg = LabelConfig(stop_pips=10.0, target_pips=20.0)
    assert cfg.pip_multiplier == 10.0
    assert cfg.tick_size == 0.00001
    assert cfg.max_bars == 50


def test_label_config_is_frozen():
    cfg = LabelConfig(stop_pips=10.0, target_pips=20.0)
    with pytest.raises(Exception):
        cfg.stop_pips = 5.0  # type: ignore[misc]


def test_label_config_custom_values():
    cfg = LabelConfig(
        stop_pips=5.0,
        target_pips=15.0,
        pip_multiplier=100.0,
        tick_size=0.01,
        max_bars=30,
    )
    assert cfg.pip_multiplier == 100.0
    assert cfg.max_bars == 30


# ─────────────────────────────────────────────────────────────────────────────
# ReplayEvaluationRecord
# ─────────────────────────────────────────────────────────────────────────────


def test_record_builds():
    r = _make_record(bar_index=5)
    assert r.bar_index == 5
    assert r.symbol == "EURUSD"


def test_record_is_frozen():
    r = _make_record()
    with pytest.raises(Exception):
        r.bar_index = 99  # type: ignore[misc]


def test_record_regime_fields_default_none():
    r = _make_record()
    assert r.regime_label is None
    assert r.regime_confidence is None


def test_record_with_regime():
    r = ReplayEvaluationRecord(
        bar_index=0,
        snapshot_id="snap_x",
        symbol="EURUSD",
        timestamp_utc=_TS,
        evaluation_result=_make_no_trade_result(),
        regime_label=RegimeLabel.RANGE,
        regime_confidence=0.75,
    )
    assert r.regime_label == RegimeLabel.RANGE
    assert r.regime_confidence == pytest.approx(0.75)


def test_record_serialises_to_json():
    r = _make_record()
    json_str = r.model_dump_json()
    assert "EURUSD" in json_str
    assert "NOT_IN_TARGET_SESSION" in json_str


# ─────────────────────────────────────────────────────────────────────────────
# LabeledCandidateOutcome
# ─────────────────────────────────────────────────────────────────────────────


def test_labeled_outcome_builds():
    lbl = _make_label()
    assert lbl.outcome == LabelOutcome.WIN
    assert lbl.entry_activated is True


def test_labeled_outcome_is_frozen():
    lbl = _make_label()
    with pytest.raises(Exception):
        lbl.outcome = LabelOutcome.LOSS  # type: ignore[misc]


def test_labeled_outcome_not_activated_has_none_fields():
    lbl = LabeledCandidateOutcome(
        setup_id="s",
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
    assert lbl.bars_to_entry is None
    assert lbl.mfe_pips is None
    assert lbl.pnl_pips is None


def test_labeled_outcome_serialises_to_json():
    lbl = _make_label(LabelOutcome.LOSS)
    json_str = lbl.model_dump_json()
    assert "LOSS" in json_str
    assert "EURUSD" in json_str


# ─────────────────────────────────────────────────────────────────────────────
# ReplayRunSummary
# ─────────────────────────────────────────────────────────────────────────────


def test_summary_builds():
    s = _make_summary()
    assert s.total_snapshots == 100
    assert s.total_candidates == 30


def test_summary_is_frozen():
    s = _make_summary()
    with pytest.raises(Exception):
        s.total_candidates = 0  # type: ignore[misc]


def test_summary_label_counts_accessible():
    s = _make_summary()
    assert s.label_wins + s.label_losses + s.label_timeouts + s.label_not_activated == 30


# ─────────────────────────────────────────────────────────────────────────────
# ReplayRunResult
# ─────────────────────────────────────────────────────────────────────────────


def test_run_result_builds():
    r = ReplayRunResult(
        summary=_make_summary(),
        records=[_make_record()],
        labeled_outcomes=[_make_label()],
    )
    assert len(r.records) == 1
    assert len(r.labeled_outcomes) == 1


def test_run_result_is_frozen():
    r = ReplayRunResult(
        summary=_make_summary(),
        records=[],
        labeled_outcomes=[],
    )
    with pytest.raises(Exception):
        r.records = []  # type: ignore[misc]


def test_run_result_serialises_to_json():
    r = ReplayRunResult(
        summary=_make_summary(),
        records=[_make_record()],
        labeled_outcomes=[_make_label()],
    )
    json_str = r.model_dump_json()
    assert "or_london_v1" in json_str
    assert "WIN" in json_str
