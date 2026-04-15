"""
aion.strategies.opening_range
──────────────────────────────
Opening Range strategy engine.

What it does
────────────
Identifies whether a valid Opening Range (OR) breakout setup exists at the
time of the given MarketSnapshot.

Strategy logic (V1 — breakout variant):
  1. Verify data quality is above minimum threshold.
  2. Verify the snapshot is in the target session.
  3. Check the opening range window state (active / completed / unavailable).
  4. Retrieve OR high and OR low from the feature_vector.
  5. Validate the range size is within [min_range_pips, max_range_pips].
  6. For each permitted direction (LONG, SHORT, or both):
       - LONG  → entry reference = OR high (breakout above)
       - SHORT → entry reference = OR low  (breakout below)
  7. Return a StrategyEvaluationResult with a CandidateSetup, or a
     NoTradeDecision with the specific rejection reason.

No-lookahead guarantee
──────────────────────
All data consumed here comes from MarketSnapshot, which is assembled from
bars[0..T] only.  The OR high/low in the feature vector are computed with
the same guarantee (see features.py).

What this engine does NOT do
─────────────────────────────
- No entry timing (market / limit / stop order selection).
- No stop-loss or take-profit calculation.
- No position sizing.
- No portfolio awareness.
- No regime filter (will be added in a later batch via filters.py).

These responsibilities belong to downstream modules.

Design note — single vs. dual candidate
─────────────────────────────────────────
With direction_bias=None, the engine can theoretically return both a LONG
and a SHORT setup.  V1 returns only one result (the first valid direction).
Returning multiple candidates will be added when the strategy layer gains
a ranking/scoring pass.  For now, LONG is checked before SHORT.
"""

from __future__ import annotations

from aion.core.constants import MIN_QUALITY_SCORE
from aion.core.enums import SessionName, TradeDirection
from aion.core.models import MarketSnapshot
from aion.strategies.base import StrategyEngine
from aion.strategies.models import (
    CandidateSetup,
    NoTradeDecision,
    OpeningRangeDefinition,
    OpeningRangeState,
    StrategyEvaluationResult,
    StrategyOutcome,
)


class OpeningRangeEngine(StrategyEngine):
    """
    Opening Range breakout strategy engine.

    Instantiate with an OpeningRangeDefinition that specifies the session,
    range size limits, and optional direction bias.

    Parameters
    ----------
    definition:
        Strategy configuration.  Immutable after construction.
    min_quality_score:
        Data quality threshold below which the engine returns INSUFFICIENT_DATA.
        Defaults to the platform constant MIN_QUALITY_SCORE.

    Example
    -------
    defn = OpeningRangeDefinition(
        strategy_id="or_london_v1",
        session_name="LONDON",
        min_range_pips=5.0,
        max_range_pips=40.0,
    )
    engine = OpeningRangeEngine(defn)
    result = engine.evaluate(snapshot)
    """

    def __init__(
        self,
        definition: OpeningRangeDefinition,
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
        Evaluate the snapshot against the opening range strategy.

        Returns a StrategyEvaluationResult with:
          outcome=CANDIDATE        if a valid OR breakout setup exists
          outcome=NO_TRADE         if conditions were checked but not met
          outcome=INSUFFICIENT_DATA if quality or data is too poor to evaluate
        """
        ctx = snapshot.session_context
        fv = snapshot.feature_vector
        instrument = snapshot.instrument
        ts = snapshot.timestamp_utc

        def _no_trade(
            reason_code: str,
            reason_detail: str,
            or_high: float | None = None,
            or_low: float | None = None,
            or_state: OpeningRangeState | None = None,
        ) -> StrategyEvaluationResult:
            return StrategyEvaluationResult(
                outcome=StrategyOutcome.NO_TRADE,
                strategy_id=self.strategy_id,
                symbol=snapshot.symbol,
                timestamp_utc=ts,
                no_trade=NoTradeDecision(
                    strategy_id=self.strategy_id,
                    symbol=snapshot.symbol,
                    timestamp_utc=ts,
                    reason_code=reason_code,
                    reason_detail=reason_detail,
                    or_high=or_high,
                    or_low=or_low,
                    or_state=or_state,
                ),
            )

        def _insufficient(reason: str) -> StrategyEvaluationResult:
            return StrategyEvaluationResult(
                outcome=StrategyOutcome.INSUFFICIENT_DATA,
                strategy_id=self.strategy_id,
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

        # ── Guard 2: target session ───────────────────────────────────────────
        current_session = ctx.session_name.value  # e.g. "LONDON"
        target_session = self._defn.session_name

        # OVERLAP_LONDON_NY counts as both LONDON and NEW_YORK active
        in_target = (
            current_session == target_session
            or (
                current_session == SessionName.OVERLAP_LONDON_NY
                and target_session in ("LONDON", "NEW_YORK")
            )
        )
        if not in_target:
            return _no_trade(
                reason_code="NOT_IN_TARGET_SESSION",
                reason_detail=(
                    f"Current session is {current_session!r}; "
                    f"strategy targets {target_session!r}."
                ),
            )

        # ── Guard 3: opening range state ──────────────────────────────────────
        or_state = _resolve_or_state(ctx)

        if or_state == OpeningRangeState.UNAVAILABLE:
            return _no_trade(
                reason_code="OR_UNAVAILABLE",
                reason_detail="Opening range data is not available for this bar.",
                or_state=or_state,
            )

        if self._defn.require_completed_range and or_state != OpeningRangeState.COMPLETED:
            return _no_trade(
                reason_code="OR_NOT_COMPLETED",
                reason_detail=(
                    f"Opening range state is {or_state.value}; "
                    "require_completed_range=True."
                ),
                or_state=or_state,
            )

        # ── Guard 4: OR levels available ─────────────────────────────────────
        or_high = fv.opening_range_high
        or_low = fv.opening_range_low

        if or_high is None or or_low is None:
            return _no_trade(
                reason_code="OR_UNAVAILABLE",
                reason_detail="opening_range_high or opening_range_low is None in feature vector.",
                or_state=or_state,
            )

        # ── Guard 5: range size ───────────────────────────────────────────────
        range_price = or_high - or_low
        if range_price <= 0:
            return _no_trade(
                reason_code="OR_UNAVAILABLE",
                reason_detail=f"OR high ({or_high}) <= OR low ({or_low}).",
                or_high=or_high,
                or_low=or_low,
                or_state=or_state,
            )

        tick_size = instrument.tick_size
        range_pips = self._defn.price_to_pips(range_price, tick_size)

        if range_pips < self._defn.min_range_pips:
            return _no_trade(
                reason_code="RANGE_TOO_TIGHT",
                reason_detail=(
                    f"Range is {range_pips:.1f} pips; "
                    f"minimum is {self._defn.min_range_pips:.1f} pips."
                ),
                or_high=or_high,
                or_low=or_low,
                or_state=or_state,
            )

        if range_pips > self._defn.max_range_pips:
            return _no_trade(
                reason_code="RANGE_TOO_WIDE",
                reason_detail=(
                    f"Range is {range_pips:.1f} pips; "
                    f"maximum is {self._defn.max_range_pips:.1f} pips."
                ),
                or_high=or_high,
                or_low=or_low,
                or_state=or_state,
            )

        # ── Determine direction ───────────────────────────────────────────────
        # V1: check LONG first, then SHORT.  direction_bias limits to one.
        direction = _select_direction(self._defn.direction_bias)

        # ── Guard 6: retest penetration ───────────────────────────────────────
        if self._defn.max_retest_penetration_points is not None:
            close = snapshot.latest_bar.close
            if direction == TradeDirection.LONG:
                penetration_price = or_high - close
            else:
                penetration_price = close - or_low

            if penetration_price > 0:
                penetration_pips = self._defn.price_to_pips(penetration_price, tick_size)
                if penetration_pips > self._defn.max_retest_penetration_points:
                    return _no_trade(
                        reason_code="RETEST_TOO_DEEP",
                        reason_detail=(
                            f"Retest penetration {penetration_pips:.1f} points "
                            f"exceeds max {self._defn.max_retest_penetration_points:.1f} points "
                            f"({'LONG' if direction == TradeDirection.LONG else 'SHORT'})."
                        ),
                        or_high=or_high,
                        or_low=or_low,
                        or_state=or_state,
                    )

        # ── Build CandidateSetup ──────────────────────────────────────────────
        entry_ref = or_high if direction == TradeDirection.LONG else or_low

        candidate = CandidateSetup(
            strategy_id=self.strategy_id,
            strategy_version=self.version,
            symbol=snapshot.symbol,
            timestamp_utc=ts,
            direction=direction,
            entry_reference=entry_ref,
            range_high=or_high,
            range_low=or_low,
            range_size_pips=range_pips,
            session_name=current_session,
            quality_score=quality_score,
            atr_14=fv.atr_14,
        )

        return StrategyEvaluationResult(
            outcome=StrategyOutcome.CANDIDATE,
            strategy_id=self.strategy_id,
            symbol=snapshot.symbol,
            timestamp_utc=ts,
            candidate=candidate,
        )

    # ── Public helpers ────────────────────────────────────────────────────────

    @property
    def definition(self) -> OpeningRangeDefinition:
        """The strategy configuration."""
        return self._defn


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_or_state(ctx) -> OpeningRangeState:
    """
    Translate SessionContext flags into an OpeningRangeState.

    Rules:
      NOT_IN_SESSION    → is_session_open_window is False
      UNAVAILABLE       → session open but no opening range data (session_open_utc is None)
      ACTIVE            → opening_range_active is True
      COMPLETED         → opening_range_completed is True
      UNAVAILABLE       → fallback (open session, but neither active nor completed)
    """
    if not ctx.is_session_open_window:
        return OpeningRangeState.NOT_IN_SESSION

    if ctx.session_open_utc is None:
        return OpeningRangeState.UNAVAILABLE

    if ctx.opening_range_active:
        return OpeningRangeState.ACTIVE

    if ctx.opening_range_completed:
        return OpeningRangeState.COMPLETED

    return OpeningRangeState.UNAVAILABLE


def _select_direction(bias: TradeDirection | None) -> TradeDirection:
    """
    Return the direction to use for the candidate setup.

    With no bias, defaults to LONG (V1 simplification — both directions
    are architecturally supported; multi-candidate output is a future feature).
    """
    if bias is not None:
        return bias
    return TradeDirection.LONG
