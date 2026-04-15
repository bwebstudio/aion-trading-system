"""
aion.strategies.models
───────────────────────
Domain models for the Strategy Engine layer.

These models describe what a strategy *found* (a candidate setup) and
why it *didn't fire* (a no-trade decision).  They are NOT execution
orders — sizing, SL/TP, and submission live in aion.execution.

Design principles:
  - All models are immutable (frozen=True).
  - CandidateSetup carries only the information that exists at signal
    generation time: direction, price levels, and context metadata.
    Risk sizing is deliberately absent — that belongs to the Risk module.
  - NoTradeDecision is explicit about the reason so downstream modules
    (logging, ML training, research) can analyse why trades were skipped.
  - StrategyEvaluationResult is the single return type of every strategy
    engine's `evaluate()` method, making the interface uniform regardless
    of strategy type.
  - OpeningRangeDefinition holds the strategy's configuration.  It is
    separate from the engine so parameters can be varied per instrument,
    session, or research experiment without subclassing.

Rules:
  - No business logic here — only data + simple derived properties.
  - All float price levels are in the instrument's native price units.
  - All pip calculations are the caller's responsibility.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aion.core.enums import TradeDirection
from aion.core.ids import new_snapshot_id


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class StrategyOutcome(str, Enum):
    """
    Top-level result of a strategy evaluation.

    CANDIDATE        — A valid setup was found.  `candidate` is populated.
    NO_TRADE         — Conditions checked but no setup triggered.  `reason` explains.
    INSUFFICIENT_DATA — Not enough data to evaluate (early bars, low quality).
                        Distinct from NO_TRADE: the strategy didn't *choose* to skip,
                        it simply couldn't run.
    """
    CANDIDATE = "CANDIDATE"
    NO_TRADE = "NO_TRADE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class OpeningRangeState(str, Enum):
    """
    Describes the state of the opening range window at evaluation time.

    ACTIVE     — The opening range period is still accumulating.
    COMPLETED  — The opening range period has closed.  OR high/low are final.
    NOT_IN_SESSION — No active session at this time.
    UNAVAILABLE — Session is open but OR data is missing (e.g. first bar).
    """
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    NOT_IN_SESSION = "NOT_IN_SESSION"
    UNAVAILABLE = "UNAVAILABLE"


# ─────────────────────────────────────────────────────────────────────────────
# Opening Range configuration
# ─────────────────────────────────────────────────────────────────────────────


class OpeningRangeDefinition(BaseModel):
    """
    Configuration for an Opening Range strategy instance.

    All range sizes are in pips (instrument-agnostic units).
    The engine converts pips → price using `tick_size * pip_multiplier`.

    Typical EURUSD configuration:
        min_range_pips  = 5    (range too tight → choppy, skip)
        max_range_pips  = 50   (range too wide → news event, skip)
        session_name    = "LONDON" or "NEW_YORK"

    Pip multiplier note:
        For 5-decimal instruments (EURUSD): 1 pip = 10 ticks = 0.0001
        For 3-decimal instruments (USDJPY): 1 pip = 10 ticks = 0.01
        The multiplier is always 10 for standard forex instruments.
    """

    model_config = ConfigDict(frozen=True)

    strategy_id: str
    """Stable identifier for this strategy variant.  E.g. 'or_london_v1'."""

    version: str = "1.0.0"

    session_name: str
    """
    The session whose opening range to track.
    Must match a SessionName value: 'LONDON', 'NEW_YORK', 'ASIA'.
    """

    min_range_pips: float
    """
    Minimum opening range size (pips).
    Ranges smaller than this are considered too tight and skipped.
    """

    max_range_pips: float
    """
    Maximum opening range size (pips).
    Ranges larger than this indicate an abnormal event (news, gap) and are skipped.
    """

    pip_multiplier: float = 10.0
    """
    Number of ticks per pip.  Default 10 (standard 5-decimal forex instruments).
    Used together with InstrumentSpec.tick_size to convert pips → price units.
    """

    direction_bias: TradeDirection | None = None
    """
    If set, only setups in this direction are generated.
    None means both long (breakout above OR high) and short (below OR low) are valid.
    """

    require_completed_range: bool = True
    """
    If True (default), the engine skips evaluation while the OR window is still
    accumulating.  Set to False only for real-time monitoring where you want
    intermediate OR levels.
    """

    max_retest_penetration_points: float | None = None
    """
    Maximum allowed retest penetration into the OR range, expressed in pips
    (or instrument-equivalent points).

    After the OR is completed, the current bar's close may have pulled back
    inside the OR range — this is called a retest.  If the retest depth
    exceeds this threshold, the engine returns NO_TRADE with reason_code
    'RETEST_TOO_DEEP'.

    Penetration is computed as:
      LONG  → max(0, or_high - close) converted to pips via price_to_pips()
      SHORT → max(0, close - or_low)  converted to pips via price_to_pips()

    When None (default), the retest depth is not checked at all.

    Unit note for non-forex instruments (e.g. US100.cash):
      With pip_multiplier=1 and tick_size=1.0, one "pip" equals one index
      point.  Setting max_retest_penetration_points=10 then means "reject
      setups where price has re-entered the OR by more than 10 index points".
      Configure InstrumentSpec accordingly when adapting to equity indices.

    Typical values for OR on US100.cash:
      None  — no retest filter (baseline, most candidates)
      5     — tight filter (price must be at or very near the OR boundary)
      10    — recommended starting value based on historical analysis
      15    — relaxed filter (allows deeper retests)
    """

    @field_validator("min_range_pips", "max_range_pips", "pip_multiplier")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Range sizes and pip_multiplier must be positive.")
        return v

    @field_validator("max_range_pips")
    @classmethod
    def max_must_exceed_min(cls, v: float, info: Any) -> float:
        min_r = info.data.get("min_range_pips")
        if min_r is not None and v <= min_r:
            raise ValueError(
                f"max_range_pips ({v}) must be greater than min_range_pips ({min_r})."
            )
        return v

    def pips_to_price(self, pips: float, tick_size: float) -> float:
        """Convert a pip distance to a price-unit distance."""
        return pips * tick_size * self.pip_multiplier

    def price_to_pips(self, price_distance: float, tick_size: float) -> float:
        """Convert a price-unit distance to pips."""
        return price_distance / (tick_size * self.pip_multiplier)


# ─────────────────────────────────────────────────────────────────────────────
# Candidate Setup
# ─────────────────────────────────────────────────────────────────────────────


def _new_setup_id() -> str:
    return f"setup_{new_snapshot_id()[5:]}"


class CandidateSetup(BaseModel):
    """
    A candidate trade setup produced by a strategy engine.

    This is NOT an order.  It describes where a setup *exists* given the
    current market structure.  Sizing, SL/TP, and order submission are handled
    by the Risk module and Execution module respectively.

    Price levels are in the instrument's native price units.
    """

    model_config = ConfigDict(frozen=True)

    setup_id: str = Field(default_factory=_new_setup_id)
    """Unique identifier for this specific setup instance."""

    strategy_id: str
    """Which strategy produced this setup.  Matches OpeningRangeDefinition.strategy_id."""

    strategy_version: str
    """Strategy version at time of generation."""

    symbol: str
    timestamp_utc: datetime
    """Timestamp of the bar that triggered this setup."""

    direction: TradeDirection
    """LONG or SHORT."""

    # ── Key price levels ──────────────────────────────────────────────────────

    entry_reference: float
    """
    The reference price for entry.
    For OR breakout: the OR high (LONG) or OR low (SHORT).
    Execution module decides the exact entry method (market, limit, stop).
    """

    range_high: float
    """Opening range high at time of setup generation."""

    range_low: float
    """Opening range low at time of setup generation."""

    range_size_pips: float
    """Range size expressed in pips at time of setup generation."""

    # ── Context metadata ──────────────────────────────────────────────────────

    session_name: str
    """Session during which this setup was identified."""

    quality_score: float
    """Data quality score from the snapshot's DataQualityReport."""

    atr_14: float | None
    """ATR-14 from the snapshot's FeatureVector.  None if unavailable."""

    strategy_detail: dict[str, Any] = Field(default_factory=dict)
    """
    Strategy-specific data opaque to risk allocation and execution.

    Each engine populates this with whatever extra context it needs to
    communicate downstream (e.g. break candle timestamp, FVG gap size,
    retest phase).  Risk and execution modules MUST NOT depend on the
    contents of this field — it exists solely for logging, research,
    and strategy-aware post-processing.
    """

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def range_size(self) -> float:
        """Range size in price units."""
        return self.range_high - self.range_low

    @property
    def is_long(self) -> bool:
        return self.direction == TradeDirection.LONG

    @property
    def is_short(self) -> bool:
        return self.direction == TradeDirection.SHORT


# ─────────────────────────────────────────────────────────────────────────────
# No-Trade Decision
# ─────────────────────────────────────────────────────────────────────────────


class NoTradeDecision(BaseModel):
    """
    Explains why a strategy chose not to generate a setup.

    Distinct from INSUFFICIENT_DATA: here the strategy ran its full logic
    but decided conditions were not met.

    Used for:
      - Live logging (human review of skipped setups)
      - Research (dataset of non-events for ML training)
      - Debugging (confirm the filter is working correctly)
    """

    model_config = ConfigDict(frozen=True)

    strategy_id: str
    symbol: str
    timestamp_utc: datetime
    reason_code: str
    """
    Short, stable code for the rejection reason.  Examples:
      'RANGE_TOO_TIGHT'
      'RANGE_TOO_WIDE'
      'OR_NOT_COMPLETED'
      'OR_UNAVAILABLE'
      'NOT_IN_TARGET_SESSION'
      'LOW_QUALITY_DATA'
      'DIRECTION_FILTERED'
    """

    reason_detail: str
    """Human-readable explanation.  May include numeric context (e.g. actual range)."""

    or_high: float | None = None
    or_low: float | None = None
    or_state: OpeningRangeState | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Evaluation Result
# ─────────────────────────────────────────────────────────────────────────────


class StrategyEvaluationResult(BaseModel):
    """
    The uniform return type of every strategy engine's `evaluate()` call.

    Exactly one of `candidate` or `no_trade` is populated:
      - outcome=CANDIDATE        → candidate is set, no_trade is None
      - outcome=NO_TRADE         → no_trade is set, candidate is None
      - outcome=INSUFFICIENT_DATA → both are None, reason_detail explains

    This ensures callers always check `outcome` before accessing either field.
    """

    model_config = ConfigDict(frozen=True)

    outcome: StrategyOutcome
    strategy_id: str
    symbol: str
    timestamp_utc: datetime

    candidate: CandidateSetup | None = None
    no_trade: NoTradeDecision | None = None
    reason_detail: str | None = None
    """Populated when outcome=INSUFFICIENT_DATA."""

    @property
    def has_setup(self) -> bool:
        return self.outcome == StrategyOutcome.CANDIDATE

    @property
    def is_no_trade(self) -> bool:
        return self.outcome == StrategyOutcome.NO_TRADE

    @property
    def is_insufficient_data(self) -> bool:
        return self.outcome == StrategyOutcome.INSUFFICIENT_DATA
