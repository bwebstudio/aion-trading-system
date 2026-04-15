"""
aion/analytics/strategy_comparison.py
───────────────────────────────────────
Cross-strategy comparison: run two engines over the same snapshot set and
produce a side-by-side StrategyComparisonReport.

Public API:
  run_strategy_comparison(snapshots, engine_a, engine_b, *, ...) -> StrategyComparisonReport

Models:
  StrategyMetricsSummary   — compact metrics for one strategy in a comparison group
  ComparisonBreakdown      — side-by-side summaries for two strategies within a group
  StrategyComparisonReport — overall + by_session + by_regime breakdowns

Both engines receive the exact same snapshot list.  Each runs an independent
replay so their records and labels do not interfere.  The label_config and
regime_detector (if provided) are shared across both replays.

Session breakdown note:
  Only CANDIDATE records carry session information.  Snapshots that produce
  NO_TRADE or INSUFFICIENT_DATA for one strategy but CANDIDATE for the other
  will appear only in the active strategy's session breakdown.

Regime breakdown note:
  All records carry a regime_label when a RegimeDetector is provided.
  Without one, all records fall into the "UNKNOWN" bucket.
"""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel

from aion.analytics.replay_metrics import compute_metrics
from aion.core.models import MarketSnapshot
from aion.regime.base import RegimeDetector
from aion.replay.models import (
    LabelConfig,
    LabeledCandidateOutcome,
    ReplayEvaluationRecord,
    ReplayRunResult,
)
from aion.replay.runner import run_replay
from aion.strategies.base import StrategyEngine
from aion.strategies.models import StrategyOutcome


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────


class StrategyMetricsSummary(BaseModel, frozen=True):
    """Compact metrics for one strategy within a named comparison group.

    Used as a leaf node inside ComparisonBreakdown.  Contains the minimum
    set of fields needed for side-by-side evaluation of two strategies.
    """

    strategy_id: str
    candidate_count: int
    entry_activated_count: int
    win_count: int
    activation_rate: float | None
    """entry_activated_count / candidate_count; None when candidate_count == 0."""

    win_rate_on_activated: float | None
    """win_count / entry_activated_count; None when entry_activated_count == 0."""

    avg_mfe: float | None
    """Mean MFE in pips over activated entries; None when no activations."""

    avg_mae: float | None
    """Mean MAE in pips over activated entries; None when no activations."""

    avg_bars_to_resolution: float | None
    """Mean bars to WIN/LOSS over activated entries; None when no activations."""


class ComparisonBreakdown(BaseModel, frozen=True):
    """Side-by-side metrics for two strategies within a named group.

    group_key identifies the breakdown dimension:
      'overall'            — all snapshots combined
      session name         — e.g. 'LONDON', 'OVERLAP_LONDON_NY'
      regime label         — e.g. 'TRENDING', 'UNKNOWN'
    """

    group_key: str
    strategy_a: StrategyMetricsSummary
    strategy_b: StrategyMetricsSummary


class StrategyComparisonReport(BaseModel, frozen=True):
    """Full cross-strategy comparison report.

    overall    — both strategies across all snapshots
    by_session — one ComparisonBreakdown per session (union of sessions
                 observed in either strategy's CANDIDATE records)
    by_regime  — one ComparisonBreakdown per regime label (union of labels
                 from both replays; 'UNKNOWN' when no RegimeDetector used)
    """

    strategy_a_id: str
    strategy_b_id: str
    overall: ComparisonBreakdown
    by_session: list[ComparisonBreakdown]
    by_regime: list[ComparisonBreakdown]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def run_strategy_comparison(
    snapshots: list[MarketSnapshot],
    engine_a: StrategyEngine,
    engine_b: StrategyEngine,
    *,
    label_config: LabelConfig | None = None,
    regime_detector: RegimeDetector | None = None,
) -> StrategyComparisonReport:
    """Run both engines over the same snapshots and compare their metrics.

    Parameters
    ----------
    snapshots:
        Historical snapshots in chronological order.  Both engines receive
        the exact same list; no data is shared between the two replays.
    engine_a, engine_b:
        Two strategy engines to compare.  May be the same engine type with
        different configurations.
    label_config:
        Optional labeling config applied to both engines identically.
        If None, no forward labels are computed and win/loss counts are 0.
    regime_detector:
        Optional regime detector applied to both engines identically.

    Returns
    -------
    StrategyComparisonReport
        Side-by-side metrics for overall, by_session, and by_regime groups.
    """
    result_a = run_replay(
        snapshots, engine_a,
        regime_detector=regime_detector,
        label_config=label_config,
    )
    result_b = run_replay(
        snapshots, engine_b,
        regime_detector=regime_detector,
        label_config=label_config,
    )

    a_id = engine_a.strategy_id
    b_id = engine_b.strategy_id

    # ── Overall ───────────────────────────────────────────────────────────────
    overall_a = _metrics_summary(a_id, result_a.records, result_a.labeled_outcomes)
    overall_b = _metrics_summary(b_id, result_b.records, result_b.labeled_outcomes)
    overall = ComparisonBreakdown(
        group_key="overall",
        strategy_a=overall_a,
        strategy_b=overall_b,
    )

    # ── By session ────────────────────────────────────────────────────────────
    sessions_a = _split_by_session(a_id, result_a)
    sessions_b = _split_by_session(b_id, result_b)
    all_sessions = sorted(set(sessions_a.keys()) | set(sessions_b.keys()))
    by_session = [
        ComparisonBreakdown(
            group_key=s,
            strategy_a=sessions_a.get(s, _empty_summary(a_id)),
            strategy_b=sessions_b.get(s, _empty_summary(b_id)),
        )
        for s in all_sessions
    ]

    # ── By regime ─────────────────────────────────────────────────────────────
    regimes_a = _split_by_regime(a_id, result_a)
    regimes_b = _split_by_regime(b_id, result_b)
    all_regimes = sorted(set(regimes_a.keys()) | set(regimes_b.keys()))
    by_regime = [
        ComparisonBreakdown(
            group_key=r,
            strategy_a=regimes_a.get(r, _empty_summary(a_id)),
            strategy_b=regimes_b.get(r, _empty_summary(b_id)),
        )
        for r in all_regimes
    ]

    return StrategyComparisonReport(
        strategy_a_id=a_id,
        strategy_b_id=b_id,
        overall=overall,
        by_session=by_session,
        by_regime=by_regime,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _metrics_summary(
    strategy_id: str,
    records: list[ReplayEvaluationRecord],
    labeled_outcomes: list[LabeledCandidateOutcome],
) -> StrategyMetricsSummary:
    """Build a StrategyMetricsSummary from a set of records and labels."""
    m = compute_metrics(records, labeled_outcomes)
    return StrategyMetricsSummary(
        strategy_id=strategy_id,
        candidate_count=m.candidate_count,
        entry_activated_count=m.entry_activated_count,
        win_count=m.win_count,
        activation_rate=m.activation_rate,
        win_rate_on_activated=m.win_rate_on_activated,
        avg_mfe=m.avg_mfe,
        avg_mae=m.avg_mae,
        avg_bars_to_resolution=m.avg_bars_to_resolution,
    )


def _empty_summary(strategy_id: str) -> StrategyMetricsSummary:
    """Return a zero-filled summary for groups where the strategy produced nothing."""
    return StrategyMetricsSummary(
        strategy_id=strategy_id,
        candidate_count=0,
        entry_activated_count=0,
        win_count=0,
        activation_rate=None,
        win_rate_on_activated=None,
        avg_mfe=None,
        avg_mae=None,
        avg_bars_to_resolution=None,
    )


def _split_by_session(
    strategy_id: str,
    result: ReplayRunResult,
) -> dict[str, StrategyMetricsSummary]:
    """Group CANDIDATE records by session_name and compute per-session summaries."""
    label_index = {lbl.setup_id: lbl for lbl in result.labeled_outcomes}
    groups: dict[str, list[ReplayEvaluationRecord]] = defaultdict(list)

    for rec in result.records:
        if rec.evaluation_result.outcome == StrategyOutcome.CANDIDATE:
            session = rec.evaluation_result.candidate.session_name  # type: ignore[union-attr]
            groups[session].append(rec)

    summaries: dict[str, StrategyMetricsSummary] = {}
    for session, recs in groups.items():
        labels = [
            label_index[r.evaluation_result.candidate.setup_id]  # type: ignore[union-attr]
            for r in recs
            if (
                r.evaluation_result.candidate is not None
                and r.evaluation_result.candidate.setup_id in label_index
            )
        ]
        summaries[session] = _metrics_summary(strategy_id, recs, labels)

    return summaries


def _split_by_regime(
    strategy_id: str,
    result: ReplayRunResult,
) -> dict[str, StrategyMetricsSummary]:
    """Group all records by regime_label and compute per-regime summaries.

    Records with regime_label=None are grouped under 'UNKNOWN'.  Only
    CANDIDATE records within each group contribute to labeled outcome stats.
    """
    label_index = {lbl.setup_id: lbl for lbl in result.labeled_outcomes}
    groups: dict[str, list[ReplayEvaluationRecord]] = defaultdict(list)

    for rec in result.records:
        key = rec.regime_label.value if rec.regime_label is not None else "UNKNOWN"
        groups[key].append(rec)

    summaries: dict[str, StrategyMetricsSummary] = {}
    for regime, recs in groups.items():
        candidate_recs = [
            r for r in recs
            if r.evaluation_result.outcome == StrategyOutcome.CANDIDATE
        ]
        labels = [
            label_index[r.evaluation_result.candidate.setup_id]  # type: ignore[union-attr]
            for r in candidate_recs
            if (
                r.evaluation_result.candidate is not None
                and r.evaluation_result.candidate.setup_id in label_index
            )
        ]
        summaries[regime] = _metrics_summary(strategy_id, recs, labels)

    return summaries
