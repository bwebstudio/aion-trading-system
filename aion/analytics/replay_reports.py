"""
aion/analytics/replay_reports.py
──────────────────────────────────
Functions that build structured breakdown reports from replay results.

Public API:
  build_report(result)                      -> ReplayReport
  breakdown_by_session(records, labels)     -> list[BreakdownRow]
  breakdown_by_regime(records, labels)      -> list[BreakdownRow]
  breakdown_by_reason_code(records)         -> list[BreakdownRow]
  breakdown_by_direction(labeled_outcomes)  -> list[BreakdownRow]

Data availability notes:
  - Session breakdown: only CANDIDATE records carry session_name
    (via evaluation_result.candidate.session_name).  NO_TRADE records
    have no session information in the replay result and are excluded.
  - Regime breakdown: all records carry regime_label (may be None →
    grouped as "UNKNOWN").
  - Reason-code breakdown: only NO_TRADE records with a NoTradeDecision.
  - Direction breakdown: derived directly from labeled outcomes.
"""

from __future__ import annotations

from collections import defaultdict

from aion.analytics.replay_metrics import compute_metrics
from aion.analytics.replay_models import BreakdownRow, ReplayReport
from aion.replay.models import LabelOutcome, LabeledCandidateOutcome, ReplayEvaluationRecord, ReplayRunResult
from aion.strategies.models import StrategyOutcome


# ─────────────────────────────────────────────────────────────────────────────
# Public breakdowns
# ─────────────────────────────────────────────────────────────────────────────


def breakdown_by_session(
    records: list[ReplayEvaluationRecord],
    labeled_outcomes: list[LabeledCandidateOutcome],
) -> list[BreakdownRow]:
    """Group CANDIDATE records and their labels by session_name.

    NO_TRADE records lack session information in the replay result and
    are excluded.  Rows are sorted alphabetically by session name.
    """
    label_index = _build_label_index(labeled_outcomes)

    groups: dict[str, list[ReplayEvaluationRecord]] = defaultdict(list)
    for rec in records:
        if rec.evaluation_result.outcome == StrategyOutcome.CANDIDATE:
            session = rec.evaluation_result.candidate.session_name  # type: ignore[union-attr]
            groups[session].append(rec)

    rows = []
    for key in sorted(groups.keys()):
        recs = groups[key]
        labels = _gather_labels(recs, label_index)
        rows.append(_make_row(key, candidate_records=recs, no_trade_records=[], labels=labels))

    return rows


def breakdown_by_regime(
    records: list[ReplayEvaluationRecord],
    labeled_outcomes: list[LabeledCandidateOutcome],
) -> list[BreakdownRow]:
    """Group all records by regime_label.

    Records with regime_label=None are grouped under "UNKNOWN".
    Rows are sorted alphabetically by regime key.
    """
    label_index = _build_label_index(labeled_outcomes)

    groups: dict[str, list[ReplayEvaluationRecord]] = defaultdict(list)
    for rec in records:
        key = rec.regime_label.value if rec.regime_label is not None else "UNKNOWN"
        groups[key].append(rec)

    rows = []
    for key in sorted(groups.keys()):
        recs = groups[key]
        candidates = [r for r in recs if r.evaluation_result.outcome == StrategyOutcome.CANDIDATE]
        no_trades = [r for r in recs if r.evaluation_result.outcome == StrategyOutcome.NO_TRADE]
        labels = _gather_labels(candidates, label_index)
        rows.append(_make_row(key, candidate_records=candidates, no_trade_records=no_trades, labels=labels))

    return rows


def breakdown_by_reason_code(
    records: list[ReplayEvaluationRecord],
) -> list[BreakdownRow]:
    """Group NO_TRADE records by reason_code.

    Rows are sorted by count descending (highest frequency first).
    INSUFFICIENT_DATA records are excluded (they carry no reason_code).
    """
    groups: dict[str, list[ReplayEvaluationRecord]] = defaultdict(list)
    for rec in records:
        if (
            rec.evaluation_result.outcome == StrategyOutcome.NO_TRADE
            and rec.evaluation_result.no_trade is not None
        ):
            code = rec.evaluation_result.no_trade.reason_code
            groups[code].append(rec)

    rows = []
    for key, recs in sorted(groups.items(), key=lambda x: -len(x[1])):
        rows.append(_make_row(key, candidate_records=[], no_trade_records=recs, labels=[]))

    return rows


def breakdown_by_direction(
    labeled_outcomes: list[LabeledCandidateOutcome],
) -> list[BreakdownRow]:
    """Group labels by trade direction.

    Since labels come from candidates, candidate_count equals record_count
    and no_trade_count is always 0.  Rows are sorted alphabetically.
    """
    groups: dict[str, list[LabeledCandidateOutcome]] = defaultdict(list)
    for lbl in labeled_outcomes:
        groups[lbl.direction.value].append(lbl)

    rows = []
    for key in sorted(groups.keys()):
        labels = groups[key]
        entry_activated = sum(1 for lbl in labels if lbl.entry_activated)
        wins = sum(1 for lbl in labels if lbl.outcome == LabelOutcome.WIN)
        losses = sum(1 for lbl in labels if lbl.outcome == LabelOutcome.LOSS)
        timeouts = sum(1 for lbl in labels if lbl.outcome == LabelOutcome.TIMEOUT)
        not_activated = sum(
            1 for lbl in labels if lbl.outcome == LabelOutcome.ENTRY_NOT_ACTIVATED
        )
        win_rate = wins / entry_activated if entry_activated > 0 else None
        activation_rate = entry_activated / len(labels) if labels else None

        rows.append(
            BreakdownRow(
                group_key=key,
                record_count=len(labels),
                candidate_count=len(labels),
                no_trade_count=0,
                entry_activated_count=entry_activated,
                win_count=wins,
                loss_count=losses,
                timeout_count=timeouts,
                not_activated_count=not_activated,
                win_rate=win_rate,
                activation_rate=activation_rate,
            )
        )

    return rows


def build_report(result: ReplayRunResult) -> ReplayReport:
    """Build a full analytics report from a replay run result.

    Computes overall metrics and all four breakdowns in one call.
    """
    metrics = compute_metrics(result.records, result.labeled_outcomes)

    by_session = breakdown_by_session(result.records, result.labeled_outcomes)
    by_regime = breakdown_by_regime(result.records, result.labeled_outcomes)
    by_reason_code = breakdown_by_reason_code(result.records)
    by_direction = breakdown_by_direction(result.labeled_outcomes)

    # top_reason_codes: by_reason_code is already sorted desc by count
    top_reason_codes: list[tuple[str, int]] = [
        (row.group_key, row.no_trade_count) for row in by_reason_code
    ]

    return ReplayReport(
        total_records=len(result.records),
        overall_metrics=metrics,
        by_session=by_session,
        by_regime=by_regime,
        by_reason_code=by_reason_code,
        by_direction=by_direction,
        top_reason_codes=top_reason_codes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_label_index(
    labeled_outcomes: list[LabeledCandidateOutcome],
) -> dict[str, LabeledCandidateOutcome]:
    """Map setup_id → LabeledCandidateOutcome for O(1) lookup."""
    return {lbl.setup_id: lbl for lbl in labeled_outcomes}


def _gather_labels(
    candidate_records: list[ReplayEvaluationRecord],
    label_index: dict[str, LabeledCandidateOutcome],
) -> list[LabeledCandidateOutcome]:
    """Collect labels for a list of CANDIDATE records via setup_id."""
    labels = []
    for rec in candidate_records:
        if rec.evaluation_result.candidate is not None:
            setup_id = rec.evaluation_result.candidate.setup_id
            if setup_id in label_index:
                labels.append(label_index[setup_id])
    return labels


def _make_row(
    group_key: str,
    *,
    candidate_records: list[ReplayEvaluationRecord],
    no_trade_records: list[ReplayEvaluationRecord],
    labels: list[LabeledCandidateOutcome],
) -> BreakdownRow:
    """Build a BreakdownRow from pre-partitioned record lists and labels."""
    record_count = len(candidate_records) + len(no_trade_records)
    candidate_count = len(candidate_records)
    no_trade_count = len(no_trade_records)

    entry_activated = sum(1 for lbl in labels if lbl.entry_activated)
    wins = sum(1 for lbl in labels if lbl.outcome == LabelOutcome.WIN)
    losses = sum(1 for lbl in labels if lbl.outcome == LabelOutcome.LOSS)
    timeouts = sum(1 for lbl in labels if lbl.outcome == LabelOutcome.TIMEOUT)
    not_activated = sum(
        1 for lbl in labels if lbl.outcome == LabelOutcome.ENTRY_NOT_ACTIVATED
    )

    win_rate = wins / entry_activated if entry_activated > 0 else None
    activation_rate = entry_activated / candidate_count if candidate_count > 0 else None

    return BreakdownRow(
        group_key=group_key,
        record_count=record_count,
        candidate_count=candidate_count,
        no_trade_count=no_trade_count,
        entry_activated_count=entry_activated,
        win_count=wins,
        loss_count=losses,
        timeout_count=timeouts,
        not_activated_count=not_activated,
        win_rate=win_rate,
        activation_rate=activation_rate,
    )
