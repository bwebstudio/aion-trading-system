"""
aion.strategies.vwap_fade
──────────────────────────
VWAP Fade v1 — intraday mean-reversion strategy.

What it does
────────────
Identifies when the latest bar's close is significantly extended from the
session VWAP, indicating a potential reversion opportunity.

Signal logic (V1):
  1. Verify data quality is above minimum threshold.
  2. Verify session VWAP is available in the feature vector.
  3. Verify the snapshot is in the target session (or skip if session_name="ALL").
  4. Compute the distance from bar.close to vwap_session in pips.
  5. Accept only if distance is within [min_distance, max_distance].
  6. Optionally require a minimal reversal bar signal (require_rejection).
  7. Apply direction_bias if set.
  8. Return a CandidateSetup or a NoTradeDecision with a specific reason code.

Direction convention:
  close > VWAP → SHORT fade (price overextended above, expect reversion down)
  close < VWAP → LONG  fade (price overextended below, expect reversion up)

Entry reference:
  The close price of the triggering bar.  The execution module decides the
  actual order type; here we record the reference price at signal time.

Range interpretation:
  range_high = max(close, vwap)
  range_low  = min(close, vwap)
  range_size_pips = distance from close to VWAP in pips

No-lookahead guarantee:
  All consumed data comes from the MarketSnapshot assembled from bars[0..T].
  The VWAP in the feature vector is computed with the same guarantee.

What this engine does NOT do:
  - No stop-loss or take-profit calculation.
  - No position sizing or risk allocation.
  - No portfolio awareness.
  - No regime filter (add via filters.py composable wrappers).

Reason codes:
  INSUFFICIENT_DATA   — quality score or VWAP unavailable
  NOT_IN_TARGET_SESSION — snapshot session doesn't match definition
  EXTENSION_TOO_SMALL — close is too close to VWAP (noise zone)
  EXTENSION_TOO_LARGE — close is too far from VWAP (extreme event, skip)
  DIRECTION_BIAS_MISMATCH — signal direction conflicts with direction_bias
  NO_REJECTION_SIGNAL — require_rejection=True but bar shows no reversal
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

from aion.core.constants import MIN_QUALITY_SCORE
from aion.core.enums import SessionName, TradeDirection
from aion.core.models import MarketBar, MarketSnapshot
from aion.strategies.base import StrategyEngine
from aion.strategies.models import (
    CandidateSetup,
    NoTradeDecision,
    StrategyEvaluationResult,
    StrategyOutcome,
)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy configuration
# ─────────────────────────────────────────────────────────────────────────────


class VWAPFadeDefinition(BaseModel):
    """
    Configuration for a VWAP Fade strategy instance.

    All distance thresholds are in pips (instrument-agnostic).
    Pip size = tick_size × pip_multiplier.

    session_name:
      String matching a SessionName value: 'LONDON', 'NEW_YORK', 'ASIA'.
      Use "ALL" to evaluate on every snapshot regardless of session.
      OVERLAP_LONDON_NY is treated as both LONDON and NEW_YORK active.

    Typical EURUSD configuration:
      min_distance_to_vwap_pips = 10   (below this → noise, skip)
      max_distance_to_vwap_pips = 50   (above this → extreme event, skip)
      require_rejection = False        (no bar confirmation required for V1)
    """

    model_config = ConfigDict(frozen=True)

    strategy_id: str
    """Stable identifier for this strategy variant.  E.g. 'vwap_fade_london_v1'."""

    version: str = "1.0.0"

    session_name: str = "LONDON"
    """Target session name.  'ALL' disables the session guard."""

    min_distance_to_vwap_pips: float = 10.0
    """Minimum distance from close to VWAP.  Below this → EXTENSION_TOO_SMALL."""

    max_distance_to_vwap_pips: float = 50.0
    """Maximum distance from close to VWAP.  Above this → EXTENSION_TOO_LARGE."""

    require_rejection: bool = False
    """
    If True, the triggering bar must show a minimal reversal signal:
      LONG fade  → bullish bar (close > open) OR lower wick > body
      SHORT fade → bearish bar (close < open) OR upper wick > body
    """

    direction_bias: TradeDirection | None = None
    """If set, only setups in this direction are generated."""

    pip_multiplier: float = 10.0
    """Ticks per pip.  Default 10 for standard 5-decimal forex."""

    tick_size: float = 0.00001
    """Minimum price increment.  Default 0.00001 for EURUSD."""

    @field_validator("min_distance_to_vwap_pips", "pip_multiplier", "tick_size")
    @classmethod
    def _must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Distance thresholds, pip_multiplier, and tick_size must be positive.")
        return v

    @field_validator("max_distance_to_vwap_pips")
    @classmethod
    def _max_must_exceed_min(cls, v: float, info) -> float:
        min_d = info.data.get("min_distance_to_vwap_pips")
        if min_d is not None and v <= min_d:
            raise ValueError(
                f"max_distance_to_vwap_pips ({v}) must exceed "
                f"min_distance_to_vwap_pips ({min_d})."
            )
        return v

    @property
    def pip_size(self) -> float:
        """Price distance of one pip."""
        return self.tick_size * self.pip_multiplier

    def pips_to_price(self, pips: float) -> float:
        """Convert pip distance to price-unit distance."""
        return pips * self.pip_size

    def price_to_pips(self, price_distance: float) -> float:
        """Convert absolute price distance to pips."""
        return abs(price_distance) / self.pip_size


# ─────────────────────────────────────────────────────────────────────────────
# Strategy engine
# ─────────────────────────────────────────────────────────────────────────────


class VWAPFadeEngine(StrategyEngine):
    """
    VWAP Fade v1 — mean-reversion strategy engine.

    Instantiate with a VWAPFadeDefinition that specifies the session,
    distance thresholds, and optional direction bias.

    Parameters
    ----------
    definition:
        Strategy configuration.  Immutable after construction.
    min_quality_score:
        Data quality threshold below which the engine returns INSUFFICIENT_DATA.
        Defaults to the platform constant MIN_QUALITY_SCORE.

    Example
    -------
    defn = VWAPFadeDefinition(
        strategy_id="vwap_fade_london_v1",
        session_name="LONDON",
        min_distance_to_vwap_pips=10.0,
        max_distance_to_vwap_pips=50.0,
    )
    engine = VWAPFadeEngine(defn)
    result = engine.evaluate(snapshot)
    """

    def __init__(
        self,
        definition: VWAPFadeDefinition,
        min_quality_score: float = MIN_QUALITY_SCORE,
    ) -> None:
        self._defn = definition
        self._min_quality = min_quality_score

    # ── StrategyEngine interface ──────────────────────────────────────────────

    @property
    def strategy_id(self) -> str:
        return self._defn.strategy_id

    @property
    def version(self) -> str:
        return self._defn.version

    # ── Main evaluation entry point ───────────────────────────────────────────

    def evaluate(self, snapshot: MarketSnapshot) -> StrategyEvaluationResult:
        """
        Evaluate the snapshot against the VWAP Fade strategy.

        Returns a StrategyEvaluationResult with:
          outcome=CANDIDATE         if a valid fade setup exists
          outcome=NO_TRADE          if conditions were checked but not met
          outcome=INSUFFICIENT_DATA if quality or VWAP data is insufficient
        """
        defn = self._defn
        fv = snapshot.feature_vector
        bar = snapshot.latest_bar
        ctx = snapshot.session_context
        ts = snapshot.timestamp_utc

        def _no_trade(reason_code: str, reason_detail: str) -> StrategyEvaluationResult:
            return StrategyEvaluationResult(
                outcome=StrategyOutcome.NO_TRADE,
                strategy_id=defn.strategy_id,
                symbol=snapshot.symbol,
                timestamp_utc=ts,
                no_trade=NoTradeDecision(
                    strategy_id=defn.strategy_id,
                    symbol=snapshot.symbol,
                    timestamp_utc=ts,
                    reason_code=reason_code,
                    reason_detail=reason_detail,
                ),
            )

        def _insufficient(reason: str) -> StrategyEvaluationResult:
            return StrategyEvaluationResult(
                outcome=StrategyOutcome.INSUFFICIENT_DATA,
                strategy_id=defn.strategy_id,
                symbol=snapshot.symbol,
                timestamp_utc=ts,
                reason_detail=reason,
            )

        # ── Guard 1: data quality ─────────────────────────────────────────────
        quality_score = snapshot.quality_report.quality_score
        if quality_score < self._min_quality:
            return _insufficient(
                f"Quality score {quality_score:.4f} below threshold {self._min_quality:.4f}."
            )

        # ── Guard 2: VWAP availability ────────────────────────────────────────
        if fv is None or fv.vwap_session is None:
            return _insufficient(
                "vwap_session is None; insufficient bar history to compute VWAP."
            )

        # ── Guard 3: target session ───────────────────────────────────────────
        if defn.session_name != "ALL":
            current_session = ctx.session_name.value
            target_session = defn.session_name
            in_target = current_session == target_session or (
                current_session == SessionName.OVERLAP_LONDON_NY.value
                and target_session in ("LONDON", "NEW_YORK")
            )
            if not in_target:
                return _no_trade(
                    reason_code="NOT_IN_TARGET_SESSION",
                    reason_detail=(
                        f"Current session is {current_session!r}; "
                        f"strategy targets {target_session!r}."
                    ),
                )

        # ── Compute distance from close to VWAP ───────────────────────────────
        close = bar.close
        vwap = fv.vwap_session
        distance_price = close - vwap   # positive → above VWAP (SHORT)
                                        # negative → below VWAP (LONG)
        distance_pips = defn.price_to_pips(distance_price)
        direction = TradeDirection.SHORT if distance_price > 0 else TradeDirection.LONG

        # ── Guard 4: extension too small ──────────────────────────────────────
        if distance_pips < defn.min_distance_to_vwap_pips:
            return _no_trade(
                reason_code="EXTENSION_TOO_SMALL",
                reason_detail=(
                    f"Distance to VWAP is {distance_pips:.1f} pips; "
                    f"minimum is {defn.min_distance_to_vwap_pips:.1f} pips."
                ),
            )

        # ── Guard 5: extension too large ──────────────────────────────────────
        if distance_pips > defn.max_distance_to_vwap_pips:
            return _no_trade(
                reason_code="EXTENSION_TOO_LARGE",
                reason_detail=(
                    f"Distance to VWAP is {distance_pips:.1f} pips; "
                    f"maximum is {defn.max_distance_to_vwap_pips:.1f} pips."
                ),
            )

        # ── Guard 6: direction bias ───────────────────────────────────────────
        if defn.direction_bias is not None and direction != defn.direction_bias:
            return _no_trade(
                reason_code="DIRECTION_BIAS_MISMATCH",
                reason_detail=(
                    f"Signal direction is {direction.value!r}; "
                    f"direction_bias requires {defn.direction_bias.value!r}."
                ),
            )

        # ── Guard 7: rejection signal (optional) ─────────────────────────────
        if defn.require_rejection and not _has_rejection(bar, direction):
            return _no_trade(
                reason_code="NO_REJECTION_SIGNAL",
                reason_detail=(
                    f"Bar shows no reversal signal for {direction.value} fade."
                ),
            )

        # ── Build CandidateSetup ──────────────────────────────────────────────
        range_high = round(max(close, vwap), 5)
        range_low = round(min(close, vwap), 5)

        candidate = CandidateSetup(
            strategy_id=defn.strategy_id,
            strategy_version=defn.version,
            symbol=snapshot.symbol,
            timestamp_utc=ts,
            direction=direction,
            entry_reference=close,
            range_high=range_high,
            range_low=range_low,
            range_size_pips=round(distance_pips, 2),
            session_name=ctx.session_name.value,
            quality_score=quality_score,
            atr_14=fv.atr_14,
        )

        return StrategyEvaluationResult(
            outcome=StrategyOutcome.CANDIDATE,
            strategy_id=defn.strategy_id,
            symbol=snapshot.symbol,
            timestamp_utc=ts,
            candidate=candidate,
        )

    # ── Public helpers ────────────────────────────────────────────────────────

    @property
    def definition(self) -> VWAPFadeDefinition:
        """The strategy configuration."""
        return self._defn


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _has_rejection(bar: MarketBar, direction: TradeDirection) -> bool:
    """Return True if the bar shows a minimal reversal signal.

    LONG fade (price below VWAP — expect move up):
      Signal = bullish bar (close > open)  OR  lower wick > body
    SHORT fade (price above VWAP — expect move down):
      Signal = bearish bar (close < open)  OR  upper wick > body

    A doji (body=0) with any wick passes the wick check, which is
    intentional: a doji at an extreme represents indecision/rejection.
    """
    if direction == TradeDirection.LONG:
        return bar.is_bullish or bar.lower_wick > bar.body
    else:  # SHORT
        return bar.is_bearish or bar.upper_wick > bar.body
