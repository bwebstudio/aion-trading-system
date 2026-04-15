"""
aion.strategies.or_retest
──────────────────────────
Opening Range + Break + Retest strategy engine.

Faithful reproduction of the "Stupid Simple 2R" bot logic validated on
live US100.cash trading:

  1. Compute OR from bars using OpeningRangeConfig (single candle or block).
  2. Detect a break candle — body opens inside OR and closes outside.
  3. Wait for a retest — bar touches the broken level and confirms (closes
     outside the range in the breakout direction).
  4. Fake out handling — if the retest bar touches but closes inside:
       - Reset breakout.
       - If allow_fake_out_reversal=True AND the same bar is a valid break
         in the opposite direction → switch to that direction.
  5. Emit one CANDIDATE per session, then stop.

State machine (per session):

    WAITING_OR ──[OR computed]──▶ WAITING_BREAK
                                        │
                         [break candle] ▼
                                  WAITING_RETEST
                                    │       │
                    [retest OK] ◀───┘       └───▶ [fake out]
                        │                           │
                    CANDIDATE                  reset / reverse
                        │
                      DONE (traded=True)

This engine is **stateful**: it tracks the current phase within a session
and resets when session_open_utc changes.  The StrategyEngine.evaluate()
contract is preserved — callers still call evaluate(snapshot) per bar.

Entry / SL / TP (from the live bots):
  LONG:  entry = or_high,  SL = midpoint,  TP = entry + (entry - midpoint) * RR
  SHORT: entry = or_low,   SL = midpoint,  TP = entry - (midpoint - entry) * RR
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

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
from aion.strategies.or_range import (
    OpeningRangeConfig,
    OpeningRangeLevel,
    compute_opening_range,
)


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class RetestPhase(str, Enum):
    """Phase within a single session."""

    WAITING_OR = "WAITING_OR"
    WAITING_BREAK = "WAITING_BREAK"
    WAITING_RETEST = "WAITING_RETEST"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


class RetestDefinition(BaseModel):
    """
    Configuration for an Opening Range + Break + Retest strategy instance.

    Typical US100.cash configuration:
        RetestDefinition(
            strategy_id="or_retest_ny_us100",
            session_name="NEW_YORK",
            or_config=OpeningRangeConfig(
                method=ORMethod.SINGLE_CANDLE,
                reference_time=time(13, 30),
                candle_timeframe=Timeframe.M5,
                min_range_points=5.0,
            ),
            rr_ratio=2.0,
            allow_fake_out_reversal=True,
        )
    """

    model_config = ConfigDict(frozen=True)

    strategy_id: str
    version: str = "1.0.0"

    session_name: str
    """Target session: 'LONDON', 'NEW_YORK', 'ASIA', 'OVERLAP_LONDON_NY'."""

    or_config: OpeningRangeConfig
    """How to compute the Opening Range from bar data."""

    rr_ratio: float = 2.0
    """Reward-to-risk ratio for TP calculation."""

    allow_fake_out_reversal: bool = True
    """
    When a retest fails (fake out), and the same bar is a valid break
    candle in the opposite direction, immediately switch to that direction
    instead of waiting for a fresh break.
    """

    direction_bias: TradeDirection | None = None
    """If set, only setups in this direction are emitted."""

    @field_validator("rr_ratio")
    @classmethod
    def _rr_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("rr_ratio must be positive.")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────


class OpeningRangeRetestEngine(StrategyEngine):
    """
    Opening Range + Break + Retest — stateful per session.

    Call evaluate(snapshot) once per bar in chronological order.
    The engine tracks its phase internally and resets on session change.
    """

    def __init__(
        self,
        definition: RetestDefinition,
        min_quality_score: float = MIN_QUALITY_SCORE,
    ) -> None:
        self._defn = definition
        self._min_quality = min_quality_score
        self._reset_state()

    # ── StrategyEngine interface ────────────────────────────────────────────

    @property
    def strategy_id(self) -> str:
        return self._defn.strategy_id

    @property
    def version(self) -> str:
        return self._defn.version

    def evaluate(self, snapshot: MarketSnapshot) -> StrategyEvaluationResult:
        ts = snapshot.timestamp_utc
        ctx = snapshot.session_context
        bar = snapshot.latest_bar

        # ── Session reset ───────────────────────────────────────────────────
        session_key = ctx.session_open_utc
        if session_key != self._session_key:
            self._reset_state()
            self._session_key = session_key

        # ── Already traded this session ─────────────────────────────────────
        if self._traded:
            return self._no_trade(snapshot, "SESSION_DONE",
                                  "Already emitted a signal this session.")

        # ── Guard: quality ──────────────────────────────────────────────────
        quality = snapshot.quality_report.quality_score
        if quality < self._min_quality:
            return self._insufficient(snapshot,
                                      f"Quality {quality:.4f} < {self._min_quality:.4f}.")

        # ── Guard: session ──────────────────────────────────────────────────
        if not self._in_target_session(ctx):
            return self._no_trade(snapshot, "NOT_IN_TARGET_SESSION",
                                  f"Session is {ctx.session_name.value!r}, "
                                  f"target is {self._defn.session_name!r}.")

        # ── Phase: WAITING_OR ───────────────────────────────────────────────
        if self._phase == RetestPhase.WAITING_OR:
            return self._handle_waiting_or(snapshot)

        # ── Phase: WAITING_BREAK ────────────────────────────────────────────
        if self._phase == RetestPhase.WAITING_BREAK:
            return self._handle_waiting_break(snapshot, bar)

        # ── Phase: WAITING_RETEST ───────────────────────────────────────────
        if self._phase == RetestPhase.WAITING_RETEST:
            return self._handle_waiting_retest(snapshot, bar)

        return self._no_trade(snapshot, "UNKNOWN_PHASE",
                              f"Unexpected phase {self._phase}.")

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def definition(self) -> RetestDefinition:
        return self._defn

    @property
    def phase(self) -> RetestPhase:
        return self._phase

    @property
    def or_level(self) -> OpeningRangeLevel | None:
        return self._or_level

    # ── Internal state management ───────────────────────────────────────────

    def _reset_state(self) -> None:
        self._session_key: datetime | None = None
        self._phase = RetestPhase.WAITING_OR
        self._or_level: OpeningRangeLevel | None = None
        self._break_dir: TradeDirection | None = None
        self._break_bar_ts: datetime | None = None
        self._or_bar_ts: datetime | None = None
        self._traded = False

    # ── Phase handlers ──────────────────────────────────────────────────────

    def _handle_waiting_or(
        self, snapshot: MarketSnapshot
    ) -> StrategyEvaluationResult:
        """Try to compute OR from the snapshot's bars."""
        bars = self._select_bars(snapshot)
        level = compute_opening_range(bars, self._defn.or_config)

        if level is None:
            return self._no_trade(snapshot, "OR_NOT_READY",
                                  "Opening range not yet available from bars.")

        self._or_level = level
        self._or_bar_ts = level.computed_at
        self._phase = RetestPhase.WAITING_BREAK

        # Do NOT check break on the same bar that defined the OR.
        # The break must come from a subsequent bar.
        return self._no_trade(snapshot, "OR_SET",
                              f"OR set: [{level.or_low:.2f} – {level.or_high:.2f}]. "
                              f"Waiting for break on next bar.")

    def _handle_waiting_break(
        self, snapshot: MarketSnapshot, bar: MarketBar
    ) -> StrategyEvaluationResult:
        """Check if the current bar is a valid break candle."""
        lvl = self._or_level

        # Block break on the same bar that defined the OR
        if self._or_bar_ts is not None and bar.timestamp_utc <= self._or_bar_ts:
            return self._no_trade(
                snapshot, "OR_BAR_SKIP",
                "Break not allowed on the same bar that defined the OR.",
            )

        direction = self._check_break(bar, lvl)

        if direction is None:
            return self._no_trade(
                snapshot, "NO_BREAK",
                f"No break candle. Range [{lvl.or_low:.2f} – {lvl.or_high:.2f}].",
            )

        # Direction bias filter
        if self._defn.direction_bias is not None and direction != self._defn.direction_bias:
            return self._no_trade(
                snapshot, "DIRECTION_BIAS",
                f"Break is {direction.value} but bias is {self._defn.direction_bias.value}.",
            )

        self._break_dir = direction
        self._break_bar_ts = bar.timestamp_utc
        self._phase = RetestPhase.WAITING_RETEST

        return self._no_trade(
            snapshot, "BREAK_DETECTED",
            f"{direction.value} break at {bar.close:.2f}. Waiting for retest.",
        )

    def _handle_waiting_retest(
        self, snapshot: MarketSnapshot, bar: MarketBar
    ) -> StrategyEvaluationResult:
        """Check retest confirmation, fake out, or nothing."""
        lvl = self._or_level
        direction = self._break_dir

        if direction == TradeDirection.LONG:
            touches = bar.low <= lvl.or_high
            confirms = bar.close > lvl.or_high
        else:
            touches = bar.high >= lvl.or_low
            confirms = bar.close < lvl.or_low

        # ── Retest confirmed ────────────────────────────────────────────────
        if touches and confirms:
            return self._emit_candidate(snapshot, bar, direction, lvl)

        # ── Fake out ────────────────────────────────────────────────────────
        if touches and not confirms:
            return self._handle_fake_out(snapshot, bar, lvl)

        # ── No touch — still waiting ────────────────────────────────────────
        return self._no_trade(
            snapshot, "WAITING_RETEST",
            f"Waiting for retest of {'or_high' if direction == TradeDirection.LONG else 'or_low'}.",
        )

    def _handle_fake_out(
        self,
        snapshot: MarketSnapshot,
        bar: MarketBar,
        lvl: OpeningRangeLevel,
    ) -> StrategyEvaluationResult:
        """Retest touched but closed inside. Check for reversal or reset."""
        old_dir = self._break_dir

        if self._defn.allow_fake_out_reversal:
            inverse = self._check_break(bar, lvl)
            if inverse is not None and inverse != old_dir:
                # Direction bias check on the inverse
                if (self._defn.direction_bias is None
                        or inverse == self._defn.direction_bias):
                    self._break_dir = inverse
                    self._break_bar_ts = bar.timestamp_utc
                    self._phase = RetestPhase.WAITING_RETEST
                    return self._no_trade(
                        snapshot, "FAKE_OUT_REVERSAL",
                        f"Fake out {old_dir.value}. Same bar is {inverse.value} break.",
                    )

        # Plain reset — back to waiting for a new break
        self._break_dir = None
        self._break_bar_ts = None
        self._phase = RetestPhase.WAITING_BREAK

        return self._no_trade(
            snapshot, "FAKE_OUT_RESET",
            f"Fake out {old_dir.value}. Reset to WAITING_BREAK.",
        )

    # ── Candidate emission ──────────────────────────────────────────────────

    def _emit_candidate(
        self,
        snapshot: MarketSnapshot,
        retest_bar: MarketBar,
        direction: TradeDirection,
        lvl: OpeningRangeLevel,
    ) -> StrategyEvaluationResult:
        """Build and return the CANDIDATE result."""
        rr = self._defn.rr_ratio
        risk = lvl.or_high - lvl.midpoint  # same for both sides (half range)

        if direction == TradeDirection.LONG:
            entry = lvl.or_high
            sl = lvl.midpoint
            tp = entry + risk * rr
        else:
            entry = lvl.or_low
            sl = lvl.midpoint
            tp = entry - risk * rr

        self._traded = True

        candidate = CandidateSetup(
            strategy_id=self._defn.strategy_id,
            strategy_version=self._defn.version,
            symbol=snapshot.symbol,
            timestamp_utc=snapshot.timestamp_utc,
            direction=direction,
            entry_reference=entry,
            range_high=lvl.or_high,
            range_low=lvl.or_low,
            range_size_pips=lvl.range_points,
            session_name=snapshot.session_context.session_name.value,
            quality_score=snapshot.quality_report.quality_score,
            atr_14=snapshot.feature_vector.atr_14 if snapshot.feature_vector else None,
            strategy_detail={
                "engine": "or_retest",
                "phase": RetestPhase.WAITING_RETEST.value,
                "break_direction": direction.value,
                "break_bar_timestamp": (
                    self._break_bar_ts.isoformat() if self._break_bar_ts else None
                ),
                "retest_bar_timestamp": retest_bar.timestamp_utc.isoformat(),
                "entry_price": entry,
                "stop_price": sl,
                "target_price": tp,
                "rr_ratio": rr,
                "or_high": lvl.or_high,
                "or_low": lvl.or_low,
                "midpoint": lvl.midpoint,
                "or_method": lvl.method.value,
                "or_source_bars": lvl.source_bars,
                "was_fake_out_reversal": False,
            },
        )

        return StrategyEvaluationResult(
            outcome=StrategyOutcome.CANDIDATE,
            strategy_id=self._defn.strategy_id,
            symbol=snapshot.symbol,
            timestamp_utc=snapshot.timestamp_utc,
            candidate=candidate,
        )

    # ── Break detection ─────────────────────────────────────────────────────

    @staticmethod
    def _check_break(
        bar: MarketBar, lvl: OpeningRangeLevel
    ) -> TradeDirection | None:
        """
        Check if a bar is a valid break candle.

        Bull break: open <= or_high AND close > or_high
        Bear break: open >= or_low  AND close < or_low
        """
        if bar.open <= lvl.or_high and bar.close > lvl.or_high:
            return TradeDirection.LONG
        if bar.open >= lvl.or_low and bar.close < lvl.or_low:
            return TradeDirection.SHORT
        return None

    # ── Bar selection ───────────────────────────────────────────────────────

    def _select_bars(self, snapshot: MarketSnapshot) -> list[MarketBar]:
        """Pick bars from the snapshot that match the OR config's timeframe."""
        cfg = self._defn.or_config

        # For SINGLE_CANDLE: match by candle_timeframe
        if cfg.candle_timeframe is not None:
            return snapshot.bars_for(cfg.candle_timeframe)

        # For CANDLE_BLOCK: match by block_timeframe
        return snapshot.bars_for(cfg.block_timeframe)

    # ── Session matching ────────────────────────────────────────────────────

    def _in_target_session(self, ctx) -> bool:
        current = ctx.session_name.value
        target = self._defn.session_name
        return (
            current == target
            or (current == SessionName.OVERLAP_LONDON_NY.value
                and target in ("LONDON", "NEW_YORK"))
        )

    # ── Result builders ─────────────────────────────────────────────────────

    def _no_trade(
        self,
        snapshot: MarketSnapshot,
        reason_code: str,
        reason_detail: str,
    ) -> StrategyEvaluationResult:
        return StrategyEvaluationResult(
            outcome=StrategyOutcome.NO_TRADE,
            strategy_id=self._defn.strategy_id,
            symbol=snapshot.symbol,
            timestamp_utc=snapshot.timestamp_utc,
            no_trade=NoTradeDecision(
                strategy_id=self._defn.strategy_id,
                symbol=snapshot.symbol,
                timestamp_utc=snapshot.timestamp_utc,
                reason_code=reason_code,
                reason_detail=reason_detail,
                or_high=self._or_level.or_high if self._or_level else None,
                or_low=self._or_level.or_low if self._or_level else None,
            ),
        )

    def _insufficient(
        self, snapshot: MarketSnapshot, reason: str
    ) -> StrategyEvaluationResult:
        return StrategyEvaluationResult(
            outcome=StrategyOutcome.INSUFFICIENT_DATA,
            strategy_id=self._defn.strategy_id,
            symbol=snapshot.symbol,
            timestamp_utc=snapshot.timestamp_utc,
            reason_detail=reason,
        )
