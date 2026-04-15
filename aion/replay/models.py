"""
aion.replay.models
───────────────────
Domain models for the Replay module.

These models carry the output of evaluating a StrategyEngine over a sequence
of historical snapshots and the forward-looking labels attached to each
candidate that was found.

Model hierarchy
────────────────
  LabelConfig             — parameters for the labeler (stop/target in pips,
                            max look-ahead bars)
  ReplayEvaluationRecord  — one record per snapshot evaluated; embeds the full
                            StrategyEvaluationResult plus optional regime context
  LabeledCandidateOutcome — forward-looking label for a CandidateSetup, produced
                            by label_candidate() from future bars
  ReplayRunSummary        — aggregate counts and timing for one complete run
  ReplayRunResult         — top-level container returned by run_replay()

All models are frozen (immutable) and serialise to JSON via model_dump_json().
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict

from aion.core.enums import RegimeLabel, TradeDirection
from aion.strategies.models import StrategyEvaluationResult


# ─────────────────────────────────────────────────────────────────────────────
# LabelOutcome
# ─────────────────────────────────────────────────────────────────────────────


class LabelOutcome(str, Enum):
    """
    Forward-looking outcome of a CandidateSetup, determined by the labeler.

    WIN                  — target price reached before stop price.
    LOSS                 — stop price reached before target price.
    TIMEOUT              — neither stop nor target reached within max_bars.
    ENTRY_NOT_ACTIVATED  — entry reference level never crossed; trade not taken.
    """

    WIN = "WIN"
    LOSS = "LOSS"
    TIMEOUT = "TIMEOUT"
    ENTRY_NOT_ACTIVATED = "ENTRY_NOT_ACTIVATED"


# ─────────────────────────────────────────────────────────────────────────────
# LabelConfig
# ─────────────────────────────────────────────────────────────────────────────


class LabelConfig(BaseModel):
    """
    Parameters for the forward labeler.

    stop_pips and target_pips are applied symmetrically from entry_reference.
    The labeler converts them to absolute price levels using tick_size and
    pip_multiplier; those levels are stored in LabeledCandidateOutcome.

    pip_multiplier and tick_size must match the instrument being labelled.
    Defaults match a standard 5-decimal forex instrument (EURUSD).
    """

    model_config = ConfigDict(frozen=True)

    stop_pips: float
    """Stop distance from entry in pips (positive value)."""

    target_pips: float
    """Target distance from entry in pips (positive value)."""

    pip_multiplier: float = 10.0
    """Ticks per pip.  Default 10.0 (standard 5-decimal forex)."""

    tick_size: float = 0.00001
    """Minimum price increment.  Default matches EURUSD."""

    max_bars: int = 50
    """Maximum number of future bars to inspect before declaring TIMEOUT."""


# ─────────────────────────────────────────────────────────────────────────────
# ReplayEvaluationRecord
# ─────────────────────────────────────────────────────────────────────────────


class ReplayEvaluationRecord(BaseModel):
    """
    One evaluation record, produced for every snapshot in the replay sequence.

    Embeds the full StrategyEvaluationResult so downstream analysis can
    access candidate details, no-trade reasons, etc. without a separate join.

    regime_label and regime_confidence are populated only if a RegimeDetector
    was passed to run_replay().
    """

    model_config = ConfigDict(frozen=True)

    bar_index: int
    """Zero-based position of this snapshot in the replay sequence."""

    snapshot_id: str
    symbol: str
    timestamp_utc: datetime

    evaluation_result: StrategyEvaluationResult

    regime_label: RegimeLabel | None = None
    regime_confidence: float | None = None


# ─────────────────────────────────────────────────────────────────────────────
# LabeledCandidateOutcome
# ─────────────────────────────────────────────────────────────────────────────


class LabeledCandidateOutcome(BaseModel):
    """
    Forward-looking outcome attached to a CandidateSetup.

    Computed by label_candidate() using bars *after* the setup was generated.

    Assumptions
    ────────────
    - Entry is filled at entry_reference exactly (no slippage).
    - Same-bar stop + target touch → LOSS (conservative tie-break).
    - MFE / MAE are measured from entry_reference after activation until
      resolution (WIN/LOSS) or exhaustion of future bars (TIMEOUT).
    - If entry was never activated, mfe_pips, mae_pips, and pnl_pips are None.
    """

    model_config = ConfigDict(frozen=True)

    setup_id: str
    """Links this outcome back to CandidateSetup.setup_id."""

    symbol: str
    timestamp_utc: datetime
    """Timestamp of the snapshot that generated the candidate."""

    direction: TradeDirection
    entry_reference: float
    stop_price: float
    target_price: float

    outcome: LabelOutcome

    entry_activated: bool
    bars_to_entry: int | None
    """
    Index (0-based) in the future_bars list where entry was activated.
    None if entry was never activated.
    """

    bars_to_resolution: int | None
    """
    Offset from the activation bar to the bar where outcome resolved (WIN/LOSS).
    0 = resolved on the activation bar itself.  None for TIMEOUT or no entry.
    """

    mfe_pips: float | None
    """Maximum Favorable Excursion in pips from entry_reference. None if not activated."""

    mae_pips: float | None
    """Maximum Adverse Excursion in pips from entry_reference. None if not activated."""

    pnl_pips: float | None
    """
    Simplified P&L in pips:
      WIN     → +target_pips
      LOSS    → -stop_pips
      TIMEOUT / ENTRY_NOT_ACTIVATED → None
    """


# ─────────────────────────────────────────────────────────────────────────────
# ReplayRunSummary
# ─────────────────────────────────────────────────────────────────────────────


class ReplayRunSummary(BaseModel):
    """Aggregate statistics for one complete replay run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    strategy_id: str
    symbol: str

    total_snapshots: int
    total_candidates: int
    total_no_trade: int
    total_insufficient_data: int

    total_labeled: int
    """Number of CANDIDATE setups sent to the labeler (0 if label_config=None)."""

    label_wins: int
    label_losses: int
    label_timeouts: int
    label_not_activated: int

    start_time_utc: datetime
    end_time_utc: datetime
    elapsed_seconds: float


# ─────────────────────────────────────────────────────────────────────────────
# ReplayRunResult
# ─────────────────────────────────────────────────────────────────────────────


class ReplayRunResult(BaseModel):
    """
    Full output of run_replay().

    summary          — aggregate counts and timing
    records          — one ReplayEvaluationRecord per input snapshot
    labeled_outcomes — one LabeledCandidateOutcome per CANDIDATE
                       (empty list if label_config was not provided)
    """

    model_config = ConfigDict(frozen=True)

    summary: ReplayRunSummary
    records: list[ReplayEvaluationRecord]
    labeled_outcomes: list[LabeledCandidateOutcome]
