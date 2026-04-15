"""
aion.replay.runner
───────────────────
run_replay() — sequential historical replay loop.

Takes a list of MarketSnapshot objects (oldest first) and evaluates a
StrategyEngine on each one in order, producing a ReplayRunResult.

No-lookahead guarantee
───────────────────────
The engine receives only snapshot[i] when evaluating bar i.
Every MarketSnapshot already contains only bars[0..i] (enforced upstream
by the pipeline), so no future price data leaks into the evaluation step.

The labeler is the only component that uses future data, and it does so
explicitly and only for post-hoc annotation — not for live decisions.

Labeling
─────────
If label_config is provided, every CANDIDATE result triggers a call to
label_candidate() with the latest_bar from subsequent snapshots as future
bars.  The look-ahead window is capped at label_config.max_bars.

When there are fewer future snapshots than max_bars, only the available
bars are used.  The labeler returns TIMEOUT or ENTRY_NOT_ACTIVATED in those
cases — this is expected and correct.

Empty input
────────────
An empty snapshots list is valid.  Returns a ReplayRunResult with zero
records and a zero-filled summary.
"""

from __future__ import annotations

from datetime import datetime, timezone

from aion.core.ids import new_pipeline_run_id
from aion.core.models import MarketSnapshot
from aion.regime.base import RegimeDetector
from aion.replay.labeler import label_candidate
from aion.replay.models import (
    LabelConfig,
    LabeledCandidateOutcome,
    LabelOutcome,
    ReplayEvaluationRecord,
    ReplayRunResult,
    ReplayRunSummary,
)
from aion.strategies.base import StrategyEngine
from aion.strategies.models import StrategyOutcome


def run_replay(
    snapshots: list[MarketSnapshot],
    engine: StrategyEngine,
    *,
    regime_detector: RegimeDetector | None = None,
    label_config: LabelConfig | None = None,
) -> ReplayRunResult:
    """
    Evaluate engine sequentially over a list of snapshots.

    Parameters
    ----------
    snapshots:
        Historical snapshots in chronological order, oldest first.
        Each snapshot must contain only bars up to its own timestamp.
    engine:
        Any StrategyEngine (or filter-wrapped engine).  Called once per snapshot.
    regime_detector:
        Optional.  If provided, detect() is called on every snapshot and the
        result is stored in the record's regime_label / regime_confidence.
    label_config:
        Optional.  If provided, CANDIDATE results are labelled using the
        latest_bar from subsequent snapshots as forward bars.

    Returns
    -------
    ReplayRunResult
        Contains summary counts, all evaluation records, and all labeled outcomes.
    """
    run_id = new_pipeline_run_id()
    start_time = datetime.now(timezone.utc)

    records: list[ReplayEvaluationRecord] = []
    labeled: list[LabeledCandidateOutcome] = []

    for i, snapshot in enumerate(snapshots):

        # ── Engine evaluation (no lookahead) ──────────────────────────────────
        result = engine.evaluate(snapshot)

        # ── Optional regime classification ────────────────────────────────────
        regime_label = None
        regime_confidence = None
        if regime_detector is not None:
            reg = regime_detector.detect(snapshot)
            regime_label = reg.label
            regime_confidence = reg.confidence

        record = ReplayEvaluationRecord(
            bar_index=i,
            snapshot_id=snapshot.snapshot_id,
            symbol=snapshot.symbol,
            timestamp_utc=snapshot.timestamp_utc,
            evaluation_result=result,
            regime_label=regime_label,
            regime_confidence=regime_confidence,
        )
        records.append(record)

        # ── Optional labeling (CANDIDATE results only) ────────────────────────
        if label_config is not None and result.outcome == StrategyOutcome.CANDIDATE:
            future_end = min(i + 1 + label_config.max_bars, len(snapshots))
            future_bars = [snapshots[j].latest_bar for j in range(i + 1, future_end)]
            lbl = label_candidate(result.candidate, future_bars, label_config)
            labeled.append(lbl)

    end_time = datetime.now(timezone.utc)
    elapsed = (end_time - start_time).total_seconds()

    # ── Aggregate summary counts ──────────────────────────────────────────────
    n_candidates = sum(
        1 for r in records
        if r.evaluation_result.outcome == StrategyOutcome.CANDIDATE
    )
    n_no_trade = sum(
        1 for r in records
        if r.evaluation_result.outcome == StrategyOutcome.NO_TRADE
    )
    n_insufficient = sum(
        1 for r in records
        if r.evaluation_result.outcome == StrategyOutcome.INSUFFICIENT_DATA
    )

    n_wins = sum(1 for lbl in labeled if lbl.outcome == LabelOutcome.WIN)
    n_losses = sum(1 for lbl in labeled if lbl.outcome == LabelOutcome.LOSS)
    n_timeouts = sum(1 for lbl in labeled if lbl.outcome == LabelOutcome.TIMEOUT)
    n_not_activated = sum(
        1 for lbl in labeled if lbl.outcome == LabelOutcome.ENTRY_NOT_ACTIVATED
    )

    summary = ReplayRunSummary(
        run_id=run_id,
        strategy_id=engine.strategy_id,
        symbol=snapshots[0].symbol if snapshots else "",
        total_snapshots=len(snapshots),
        total_candidates=n_candidates,
        total_no_trade=n_no_trade,
        total_insufficient_data=n_insufficient,
        total_labeled=len(labeled),
        label_wins=n_wins,
        label_losses=n_losses,
        label_timeouts=n_timeouts,
        label_not_activated=n_not_activated,
        start_time_utc=start_time,
        end_time_utc=end_time,
        elapsed_seconds=elapsed,
    )

    return ReplayRunResult(
        summary=summary,
        records=records,
        labeled_outcomes=labeled,
    )
