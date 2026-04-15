"""
aion.strategies.filters
────────────────────────
Composable pre-filters for StrategyEngine.

Each filter wraps any StrategyEngine and intercepts evaluate() before the
engine sees the snapshot.  If the filter condition fails, it returns a
StrategyEvaluationResult with outcome=NO_TRADE immediately.  If the filter
passes, it delegates to the wrapped engine.

Available filters
──────────────────
  QualityFilter   — blocks if data quality score is below threshold
  SessionFilter   — blocks if current session is not in the allowed set
  SpreadFilter    — blocks if the rolling mean spread exceeds a pip limit

Composing filters
──────────────────
Filters can be stacked:

    engine = OpeningRangeEngine(definition)
    engine = QualityFilter(engine, min_quality=0.95)
    engine = SpreadFilter(engine, max_spread_pips=2.0)
    result = engine.evaluate(snapshot)

The outermost filter runs first.  Each filter delegates inward on pass.

strategy_id and version always reflect the innermost wrapped engine so
result metadata matches the strategy that generated it.
"""

from __future__ import annotations

from aion.core.constants import MIN_QUALITY_SCORE
from aion.core.models import MarketSnapshot
from aion.strategies.base import StrategyEngine
from aion.strategies.models import (
    NoTradeDecision,
    StrategyEvaluationResult,
    StrategyOutcome,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────


def _block(
    engine: StrategyEngine,
    snapshot: MarketSnapshot,
    reason_code: str,
    reason_detail: str,
) -> StrategyEvaluationResult:
    """Return a NO_TRADE result produced by a filter (not the wrapped engine)."""
    return StrategyEvaluationResult(
        outcome=StrategyOutcome.NO_TRADE,
        strategy_id=engine.strategy_id,
        symbol=snapshot.symbol,
        timestamp_utc=snapshot.timestamp_utc,
        no_trade=NoTradeDecision(
            strategy_id=engine.strategy_id,
            symbol=snapshot.symbol,
            timestamp_utc=snapshot.timestamp_utc,
            reason_code=reason_code,
            reason_detail=reason_detail,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# QualityFilter
# ─────────────────────────────────────────────────────────────────────────────


class QualityFilter(StrategyEngine):
    """
    Blocks evaluation if the snapshot's quality score is below the threshold.

    Reason code: LOW_QUALITY_DATA

    Parameters
    ----------
    engine:
        The strategy engine to delegate to on pass.
    min_quality:
        Quality threshold [0.0, 1.0].  Defaults to MIN_QUALITY_SCORE (0.90).
    """

    def __init__(
        self,
        engine: StrategyEngine,
        min_quality: float = MIN_QUALITY_SCORE,
    ) -> None:
        self._engine = engine
        self._min_quality = min_quality

    @property
    def strategy_id(self) -> str:
        return self._engine.strategy_id

    @property
    def version(self) -> str:
        return self._engine.version

    def evaluate(self, snapshot: MarketSnapshot) -> StrategyEvaluationResult:
        score = snapshot.quality_report.quality_score
        if score < self._min_quality:
            return _block(
                self._engine,
                snapshot,
                reason_code="LOW_QUALITY_DATA",
                reason_detail=(
                    f"Quality score {score:.4f} is below filter threshold "
                    f"{self._min_quality:.4f}."
                ),
            )
        return self._engine.evaluate(snapshot)


# ─────────────────────────────────────────────────────────────────────────────
# SessionFilter
# ─────────────────────────────────────────────────────────────────────────────


class SessionFilter(StrategyEngine):
    """
    Blocks evaluation if the current session is not in the allowed set.

    Reason code: SESSION_FILTER_BLOCKED

    Parameters
    ----------
    engine:
        The strategy engine to delegate to on pass.
    allowed_sessions:
        Set of SessionName string values (e.g. {"LONDON", "OVERLAP_LONDON_NY"}).
        Comparison is against SessionContext.session_name.value.
    """

    def __init__(
        self,
        engine: StrategyEngine,
        allowed_sessions: set[str],
    ) -> None:
        self._engine = engine
        self._allowed = frozenset(allowed_sessions)

    @property
    def strategy_id(self) -> str:
        return self._engine.strategy_id

    @property
    def version(self) -> str:
        return self._engine.version

    def evaluate(self, snapshot: MarketSnapshot) -> StrategyEvaluationResult:
        current = snapshot.session_context.session_name.value
        if current not in self._allowed:
            return _block(
                self._engine,
                snapshot,
                reason_code="SESSION_FILTER_BLOCKED",
                reason_detail=(
                    f"Session {current!r} is not in allowed set "
                    f"{sorted(self._allowed)}."
                ),
            )
        return self._engine.evaluate(snapshot)


# ─────────────────────────────────────────────────────────────────────────────
# SpreadFilter
# ─────────────────────────────────────────────────────────────────────────────


class SpreadFilter(StrategyEngine):
    """
    Blocks evaluation if the rolling mean spread exceeds a pip threshold.

    Spread is read from FeatureVector.spread_mean_20, which is in broker
    points (minimum price increments).  Conversion to pips:
        pips = spread_points / pip_multiplier

    If spread_mean_20 is None (insufficient history), the filter blocks with
    reason code SPREAD_UNAVAILABLE.

    Reason codes: SPREAD_TOO_WIDE, SPREAD_UNAVAILABLE

    Parameters
    ----------
    engine:
        The strategy engine to delegate to on pass.
    max_spread_pips:
        Maximum allowed spread in pips.  E.g. 2.0 for a 2-pip threshold.
    pip_multiplier:
        Ticks per pip.  Default 10.0 (standard 5-decimal forex instruments).
    """

    def __init__(
        self,
        engine: StrategyEngine,
        max_spread_pips: float,
        pip_multiplier: float = 10.0,
    ) -> None:
        self._engine = engine
        self._max_spread_pips = max_spread_pips
        self._pip_multiplier = pip_multiplier

    @property
    def strategy_id(self) -> str:
        return self._engine.strategy_id

    @property
    def version(self) -> str:
        return self._engine.version

    def evaluate(self, snapshot: MarketSnapshot) -> StrategyEvaluationResult:
        spread_points = snapshot.feature_vector.spread_mean_20

        if spread_points is None:
            return _block(
                self._engine,
                snapshot,
                reason_code="SPREAD_UNAVAILABLE",
                reason_detail=(
                    "spread_mean_20 is None; insufficient history to evaluate spread."
                ),
            )

        spread_pips = spread_points / self._pip_multiplier
        if spread_pips > self._max_spread_pips:
            return _block(
                self._engine,
                snapshot,
                reason_code="SPREAD_TOO_WIDE",
                reason_detail=(
                    f"Mean spread {spread_pips:.2f} pips exceeds "
                    f"maximum {self._max_spread_pips:.2f} pips."
                ),
            )

        return self._engine.evaluate(snapshot)
