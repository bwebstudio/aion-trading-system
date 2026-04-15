"""
aion.strategies.base
─────────────────────
Abstract base class for all strategy engines.

Contract:
  Every strategy engine:
    1. Receives a MarketSnapshot.
    2. Applies its own signal logic (stateless, pure).
    3. Returns a StrategyEvaluationResult.

Rules:
  - Engines are stateless and deterministic given the same snapshot.
  - No file I/O, no network calls, no side effects inside evaluate().
  - The engine may reject a snapshot (INSUFFICIENT_DATA or NO_TRADE) — this is
    expected behaviour, not an error.
  - Engines never raise on valid input.  They return structured results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from aion.core.models import MarketSnapshot
from aion.strategies.models import StrategyEvaluationResult


class StrategyEngine(ABC):
    """
    Abstract base for all strategy engines in AION.

    Subclass and implement `strategy_id`, `version`, and `evaluate()`.

    Example
    -------
    class MyEngine(StrategyEngine):
        @property
        def strategy_id(self) -> str:
            return "my_strategy_v1"

        @property
        def version(self) -> str:
            return "1.0.0"

        def evaluate(self, snapshot: MarketSnapshot) -> StrategyEvaluationResult:
            ...
    """

    @property
    @abstractmethod
    def strategy_id(self) -> str:
        """Unique, stable identifier for this strategy.  E.g. 'or_london_v1'."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Semantic version string.  E.g. '1.0.0'."""

    @abstractmethod
    def evaluate(self, snapshot: MarketSnapshot) -> StrategyEvaluationResult:
        """
        Evaluate the snapshot and return a result.

        Parameters
        ----------
        snapshot:
            The complete market view produced by the Market Engine.
            Contains bars, feature vector, quality report, and session context.

        Returns
        -------
        StrategyEvaluationResult
            outcome=CANDIDATE       if a valid setup was identified
            outcome=NO_TRADE        if conditions were checked but not met
            outcome=INSUFFICIENT_DATA if the engine could not evaluate

        Never raises.  All errors are captured in the result.
        """
