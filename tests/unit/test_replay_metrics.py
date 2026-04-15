"""
tests/unit/test_replay_metrics.py
────────────────────────────────────
Unit tests for aion.analytics.replay_metrics.compute_metrics().

Tests verify:
  - Empty inputs produce zero counts and None rates
  - Counts are correct for mixed WIN/LOSS/TIMEOUT/ENTRY_NOT_ACTIVATED
  - activation_rate = None when candidate_count == 0
  - win_rate_on_activated = None when no entries activated
  - ENTRY_NOT_ACTIVATED excluded from win_rate denominator
  - avg_mfe/mae computed only over activated labels
  - avg_bars_to_entry/resolution correct
  - All wins → win_rate == 1.0; all losses → win_rate == 0.0
  - ReplayMetrics is frozen
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aion.analytics.replay_metrics import compute_metrics
from aion.analytics.replay_models import ReplayMetrics
from aion.core.enums import TradeDirection
from aion.replay.models import (
    LabelOutcome,
    LabeledCandidateOutcome,
    ReplayEvaluationRecord,
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


def _no_trade_record(bar_index: int = 0, reason: str = "OR_NOT_COMPLETED") -> ReplayEvaluationRecord:
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
    )


def _candidate_record(bar_index: int = 0) -> ReplayEvaluationRecord:
    from aion.strategies.models import CandidateSetup

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
            candidate=CandidateSetup(
                strategy_id="or_v1",
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
        ),
    )


def _label(
    outcome: LabelOutcome,
    entry_activated: bool,
    mfe: float | None = None,
    mae: float | None = None,
    bars_entry: int | None = None,
    bars_res: int | None = None,
    pnl: float | None = None,
    setup_id: str = "setup_abc",
) -> LabeledCandidateOutcome:
    return LabeledCandidateOutcome(
        setup_id=setup_id,
        symbol="EURUSD",
        timestamp_utc=_TS,
        direction=TradeDirection.LONG,
        entry_reference=1.1020,
        stop_price=1.1010,
        target_price=1.1040,
        outcome=outcome,
        entry_activated=entry_activated,
        bars_to_entry=bars_entry,
        bars_to_resolution=bars_res,
        mfe_pips=mfe,
        mae_pips=mae,
        pnl_pips=pnl,
    )


def _win(mfe: float = 25.0, mae: float = 0.0, bars_e: int = 0, bars_r: int = 2, sid: str = "s") -> LabeledCandidateOutcome:
    return _label(LabelOutcome.WIN, True, mfe=mfe, mae=mae, bars_entry=bars_e, bars_res=bars_r, pnl=20.0, setup_id=sid)


def _loss(mfe: float = 5.0, mae: float = 10.0, bars_e: int = 1, bars_r: int = 3, sid: str = "s") -> LabeledCandidateOutcome:
    return _label(LabelOutcome.LOSS, True, mfe=mfe, mae=mae, bars_entry=bars_e, bars_res=bars_r, pnl=-10.0, setup_id=sid)


def _timeout(sid: str = "s") -> LabeledCandidateOutcome:
    return _label(LabelOutcome.TIMEOUT, True, mfe=15.0, mae=3.0, bars_entry=0, bars_res=29, pnl=None, setup_id=sid)


def _not_activated(sid: str = "s") -> LabeledCandidateOutcome:
    return _label(LabelOutcome.ENTRY_NOT_ACTIVATED, False, setup_id=sid)


# ─────────────────────────────────────────────────────────────────────────────
# Empty inputs
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_inputs_zero_counts():
    m = compute_metrics([], [])
    assert m.total_records == 0
    assert m.candidate_count == 0
    assert m.no_trade_count == 0
    assert m.total_labeled == 0
    assert m.win_count == 0


def test_empty_inputs_none_rates():
    m = compute_metrics([], [])
    assert m.activation_rate is None
    assert m.win_rate_on_activated is None
    assert m.avg_mfe is None
    assert m.avg_mae is None
    assert m.avg_bars_to_entry is None
    assert m.avg_bars_to_resolution is None


# ─────────────────────────────────────────────────────────────────────────────
# Record counts
# ─────────────────────────────────────────────────────────────────────────────


def test_no_trade_count():
    recs = [_no_trade_record(i) for i in range(5)]
    m = compute_metrics(recs, [])
    assert m.no_trade_count == 5
    assert m.candidate_count == 0


def test_candidate_count():
    recs = [_candidate_record(i) for i in range(3)]
    m = compute_metrics(recs, [])
    assert m.candidate_count == 3


def test_mixed_record_counts():
    recs = [_no_trade_record(0), _candidate_record(1), _no_trade_record(2)]
    m = compute_metrics(recs, [])
    assert m.total_records == 3
    assert m.no_trade_count == 2
    assert m.candidate_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# Label outcome counts
# ─────────────────────────────────────────────────────────────────────────────


def test_label_counts_correct():
    labels = [_win(), _win(sid="s2"), _loss(sid="s3"), _timeout(sid="s4"), _not_activated(sid="s5")]
    m = compute_metrics([], labels)
    assert m.total_labeled == 5
    assert m.win_count == 2
    assert m.loss_count == 1
    assert m.timeout_count == 1
    assert m.entry_not_activated_count == 1


def test_entry_activated_count():
    labels = [_win(), _loss(sid="s2"), _timeout(sid="s3"), _not_activated(sid="s4")]
    m = compute_metrics([], labels)
    assert m.entry_activated_count == 3  # win + loss + timeout


# ─────────────────────────────────────────────────────────────────────────────
# Rates
# ─────────────────────────────────────────────────────────────────────────────


def test_activation_rate_none_when_no_candidates():
    labels = [_win()]
    m = compute_metrics([], labels)
    assert m.activation_rate is None


def test_activation_rate_correct():
    recs = [_candidate_record(0), _candidate_record(1), _candidate_record(2), _candidate_record(3)]
    labels = [_win(sid="a"), _win(sid="b")]  # 2 activated out of 4 candidates
    m = compute_metrics(recs, labels)
    assert m.activation_rate == pytest.approx(2 / 4)


def test_win_rate_none_when_no_activations():
    recs = [_candidate_record(0)]
    labels = [_not_activated()]
    m = compute_metrics(recs, labels)
    assert m.win_rate_on_activated is None


def test_win_rate_not_activated_excluded():
    """ENTRY_NOT_ACTIVATED does not count in win_rate denominator."""
    labels = [_win(), _not_activated(sid="s2")]
    m = compute_metrics([], labels)
    # activated = 1 (only the WIN), win_rate = 1/1 = 1.0
    assert m.win_rate_on_activated == pytest.approx(1.0)


def test_win_rate_all_wins():
    labels = [_win(sid=f"s{i}") for i in range(5)]
    m = compute_metrics([], labels)
    assert m.win_rate_on_activated == pytest.approx(1.0)


def test_win_rate_all_losses():
    labels = [_loss(sid=f"s{i}") for i in range(4)]
    m = compute_metrics([], labels)
    assert m.win_rate_on_activated == pytest.approx(0.0)


def test_win_rate_mixed():
    labels = [_win(), _win(sid="s2"), _loss(sid="s3")]
    m = compute_metrics([], labels)
    assert m.win_rate_on_activated == pytest.approx(2 / 3)


def test_win_rate_includes_timeouts_in_denominator():
    """TIMEOUT activations count in denominator; only wins in numerator."""
    labels = [_win(), _timeout(sid="s2"), _timeout(sid="s3")]
    m = compute_metrics([], labels)
    # 1 win / 3 activated
    assert m.win_rate_on_activated == pytest.approx(1 / 3)


# ─────────────────────────────────────────────────────────────────────────────
# Averages
# ─────────────────────────────────────────────────────────────────────────────


def test_avg_mfe_only_over_activated():
    """Not-activated labels (mfe=None) are excluded from avg_mfe."""
    labels = [
        _win(mfe=20.0, sid="s1"),
        _win(mfe=30.0, sid="s2"),
        _not_activated(sid="s3"),
    ]
    m = compute_metrics([], labels)
    assert m.avg_mfe == pytest.approx(25.0)


def test_avg_mae_correct():
    labels = [_win(mae=0.0, sid="s1"), _loss(mae=10.0, sid="s2")]
    m = compute_metrics([], labels)
    assert m.avg_mae == pytest.approx(5.0)


def test_avg_bars_to_entry_correct():
    labels = [
        _win(bars_e=0, sid="s1"),
        _win(bars_e=2, sid="s2"),
        _win(bars_e=4, sid="s3"),
    ]
    m = compute_metrics([], labels)
    assert m.avg_bars_to_entry == pytest.approx(2.0)


def test_avg_bars_to_resolution_correct():
    labels = [
        _win(bars_r=0, sid="s1"),
        _loss(bars_r=10, sid="s2"),
    ]
    m = compute_metrics([], labels)
    assert m.avg_bars_to_resolution == pytest.approx(5.0)


def test_avg_none_when_no_activated_labels():
    labels = [_not_activated()]
    m = compute_metrics([], labels)
    assert m.avg_mfe is None
    assert m.avg_mae is None
    assert m.avg_bars_to_entry is None
    assert m.avg_bars_to_resolution is None


# ─────────────────────────────────────────────────────────────────────────────
# Model properties
# ─────────────────────────────────────────────────────────────────────────────


def test_metrics_is_frozen():
    m = compute_metrics([], [])
    with pytest.raises(Exception):
        m.win_count = 99  # type: ignore[misc]


def test_metrics_serialises_to_json():
    m = compute_metrics([_no_trade_record()], [])
    json_str = m.model_dump_json()
    assert "total_records" in json_str
