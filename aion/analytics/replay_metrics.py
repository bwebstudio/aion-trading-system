"""
aion/analytics/replay_metrics.py
──────────────────────────────────
Pure functions for computing aggregate statistics from a replay run.

Public API:
  compute_metrics(records, labeled_outcomes) -> ReplayMetrics

Aggregation decisions:
  - activation_rate       = entry_activated_count / candidate_count
                            (None when candidate_count == 0)
  - win_rate_on_activated = win_count / entry_activated_count
                            ENTRY_NOT_ACTIVATED labels are in the total
                            but excluded from the win-rate denominator.
                            (None when entry_activated_count == 0)
  - avg_mfe / avg_mae     = mean over activated labels (entry_activated=True);
                            None when no activations.
  - avg_bars_to_entry /
    avg_bars_to_resolution  same scope as mfe/mae.
"""

from __future__ import annotations

from aion.analytics.replay_models import ReplayMetrics
from aion.replay.models import LabelOutcome, LabeledCandidateOutcome, ReplayEvaluationRecord
from aion.strategies.models import StrategyOutcome


def compute_metrics(
    records: list[ReplayEvaluationRecord],
    labeled_outcomes: list[LabeledCandidateOutcome],
) -> ReplayMetrics:
    """Compute aggregate statistics from replay records and their labels.

    Parameters
    ----------
    records:
        All evaluation records produced by ``run_replay()``.
    labeled_outcomes:
        All labeled outcomes produced by ``run_replay()`` (may be empty
        if ``label_config`` was not provided).

    Returns
    -------
    ReplayMetrics
        Frozen model with counts, rates, and averages.
    """
    total_records = len(records)
    no_trade_count = sum(
        1 for r in records if r.evaluation_result.outcome == StrategyOutcome.NO_TRADE
    )
    candidate_count = sum(
        1 for r in records if r.evaluation_result.outcome == StrategyOutcome.CANDIDATE
    )
    insufficient_data_count = sum(
        1 for r in records if r.evaluation_result.outcome == StrategyOutcome.INSUFFICIENT_DATA
    )

    total_labeled = len(labeled_outcomes)
    entry_activated_count = sum(1 for lbl in labeled_outcomes if lbl.entry_activated)
    win_count = sum(1 for lbl in labeled_outcomes if lbl.outcome == LabelOutcome.WIN)
    loss_count = sum(1 for lbl in labeled_outcomes if lbl.outcome == LabelOutcome.LOSS)
    timeout_count = sum(1 for lbl in labeled_outcomes if lbl.outcome == LabelOutcome.TIMEOUT)
    entry_not_activated_count = sum(
        1 for lbl in labeled_outcomes if lbl.outcome == LabelOutcome.ENTRY_NOT_ACTIVATED
    )

    activation_rate = (
        entry_activated_count / candidate_count if candidate_count > 0 else None
    )
    win_rate_on_activated = (
        win_count / entry_activated_count if entry_activated_count > 0 else None
    )

    activated = [lbl for lbl in labeled_outcomes if lbl.entry_activated]

    avg_mfe = _avg([lbl.mfe_pips for lbl in activated])
    avg_mae = _avg([lbl.mae_pips for lbl in activated])
    avg_bars_to_entry = _avg([
        float(lbl.bars_to_entry) for lbl in activated if lbl.bars_to_entry is not None
    ])
    avg_bars_to_resolution = _avg([
        float(lbl.bars_to_resolution) for lbl in activated if lbl.bars_to_resolution is not None
    ])

    return ReplayMetrics(
        total_records=total_records,
        no_trade_count=no_trade_count,
        candidate_count=candidate_count,
        insufficient_data_count=insufficient_data_count,
        total_labeled=total_labeled,
        entry_activated_count=entry_activated_count,
        win_count=win_count,
        loss_count=loss_count,
        timeout_count=timeout_count,
        entry_not_activated_count=entry_not_activated_count,
        activation_rate=activation_rate,
        win_rate_on_activated=win_rate_on_activated,
        avg_mfe=avg_mfe,
        avg_mae=avg_mae,
        avg_bars_to_entry=avg_bars_to_entry,
        avg_bars_to_resolution=avg_bars_to_resolution,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _avg(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
