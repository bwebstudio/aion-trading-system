"""
research.meta_strategy.strategy_selector
─────────────────────────────────────────
Maps each StrategyCandidate to the regimes in which it is allowed to
fire, then picks the best candidate per regime.

Regime mapping rules (v1)
─────────────────────────
For each candidate we look at the conditions in its pattern_key and its
inferred direction, then derive a set of allowed regimes:

  * momentum_X_bin == POS  AND  direction == LONG   →  TREND_UP
    momentum_X_bin == NEG  AND  direction == SHORT  →  TREND_DOWN
        (momentum-aligned: classical trend continuation)

  * momentum_X_bin == POS  AND  direction == SHORT  →  RANGE
    momentum_X_bin == NEG  AND  direction == LONG   →  RANGE
        (counter-trend / mean-reversion)

  * range_compression_bin == TRUE                   →  COMPRESSION

  * distance_to_vwap_bin in {LT_NEG_2SIG, LT_NEG_1P5SIG,
                              GT_POS_1P5SIG, GT_POS_2SIG}
                                                    →  RANGE
        (extreme distance → mean reversion context)

  * Nothing specific                                →  {all regimes}

A candidate ends up with the UNION of every regime implied by any of its
conditions.  A candidate may be allowed in several regimes.

Selector ranking (v1)
─────────────────────
Within a regime, candidates are ranked by |mean_test_return| (pulled
from expected_edge).  The top-1 per regime is the "active" candidate
for that regime.  The selector caches this active set so per-row lookup
is O(1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from research.meta_strategy.regime_classifier import (
    ALL_REGIMES,
    COMPRESSION,
    RANGE,
    TREND_DOWN,
    TREND_UP,
)
from research.pattern_strategies.strategy_candidate import StrategyCandidate

if TYPE_CHECKING:
    from research.pattern_strategies.backtest_pattern_strategy import (
        BacktestReport,
    )


# ─────────────────────────────────────────────────────────────────────────────
# NO_TRADE sentinel
# ─────────────────────────────────────────────────────────────────────────────


class _NoTradeSentinel:
    """Returned by StrategySelector.active_for when the top candidate for
    a regime fails the quality bar.  Distinguished from `None` (which
    means: no candidate was ever mapped to this regime)."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover
        return "NO_TRADE"

    def __bool__(self) -> bool:
        return False


NO_TRADE = _NoTradeSentinel()
"""Singleton — `selector.active_for(regime) is NO_TRADE` checks for skip."""


EXTREME_VWAP_BINS: frozenset[str] = frozenset(
    {"LT_NEG_2SIG", "LT_NEG_1P5SIG", "GT_POS_1P5SIG", "GT_POS_2SIG"}
)


def _step_regimes(col: str, val: str, direction: str) -> set[str]:
    """
    Regime(s) implied by a single (column, bin_value) step given the
    trade direction.  Shared by pattern- and sequence-based regime mapping.
    """
    col_l = col.lower()
    val_u = str(val).upper()
    dir_u = direction.upper()
    out: set[str] = set()

    if col_l in ("momentum_3_bin", "momentum_5_bin"):
        if val_u == "POS":
            out.add(TREND_UP if dir_u == "LONG" else RANGE)
        elif val_u == "NEG":
            out.add(TREND_DOWN if dir_u == "SHORT" else RANGE)
    elif col_l == "range_compression_bin":
        if val_u == "TRUE":
            out.add(COMPRESSION)
    elif col_l == "distance_to_vwap_bin" and val_u in EXTREME_VWAP_BINS:
        out.add(RANGE)

    return out


def allowed_regimes_for(candidate: StrategyCandidate) -> set[str]:
    """Derive the set of regimes in which a pattern candidate may fire."""
    direction = candidate.direction.upper()
    allowed: set[str] = set()
    for col, val in candidate.pattern_key:
        allowed |= _step_regimes(col, val, direction)
    if not allowed:
        allowed = set(ALL_REGIMES)
    return allowed


def allowed_regimes_for_sequence(sequence_key, direction: str) -> set[str]:
    """
    Derive the set of regimes for a sequential candidate.

    Apply the per-step rule to every step in the sequence and UNION
    the results.  The intuition: a sequence touching TREND features is
    trend-aware; a sequence touching COMPRESSION is compression-aware.
    Un-opinionated sequences are allowed in every regime (the selector's
    quality gate will still filter weak ones).
    """
    allowed: set[str] = set()
    for col, val in sequence_key:
        allowed |= _step_regimes(col, val, direction)
    if not allowed:
        allowed = set(ALL_REGIMES)
    return allowed


def _edge_magnitude(candidate: StrategyCandidate) -> float:
    edge = candidate.expected_edge or {}
    mean = edge.get("mean_test_return")
    if mean is None:
        return 0.0
    return abs(float(mean))


# ─────────────────────────────────────────────────────────────────────────────
# Selector
# ─────────────────────────────────────────────────────────────────────────────


class StrategySelector:
    """
    Regime-aware strategy selector.

    Usage:
        sel = StrategySelector(candidates, candidate_metrics=reports)
        active = sel.active_for(regime)
        # → StrategyCandidate (trade)
        # → NO_TRADE (top candidate exists but fails quality bar)
        # → None (no candidate mapped to this regime)

    v2 logic:
      * `allowed_regimes_for(candidate)` produces the candidate-regime map.
      * Within a regime, candidates are sorted by |mean_test_return| desc.
      * If `candidate_metrics` is supplied, the top-1 candidate must also
        satisfy `profit_factor >= min_profit_factor` and
        `expectancy > min_expectancy`.  Otherwise the selector returns
        NO_TRADE for that regime — the meta backtest will simply skip
        every bar in that regime instead of forcing a weak entry.
      * Without `candidate_metrics`, the quality gate is bypassed and the
        selector behaves exactly like v1.
    """

    def __init__(
        self,
        candidates: Iterable[StrategyCandidate],
        *,
        top_k_per_regime: int = 1,
        candidate_metrics: "dict[str, BacktestReport] | None" = None,
        min_profit_factor: float = 1.3,
        min_expectancy: float = 0.0,
    ) -> None:
        self._top_k = top_k_per_regime
        self._candidates: list[StrategyCandidate] = list(candidates)
        self._by_regime: dict[str, list[StrategyCandidate]] = {
            r: [] for r in ALL_REGIMES
        }
        self._metrics: dict[str, BacktestReport] = candidate_metrics or {}
        self._min_pf = min_profit_factor
        self._min_exp = min_expectancy
        self._build()

    def _build(self) -> None:
        # Bucket candidates by allowed regime.
        buckets: dict[str, list[StrategyCandidate]] = {
            r: [] for r in ALL_REGIMES
        }
        for cand in self._candidates:
            # UnifiedCandidate exposes .allowed_regimes directly;
            # a plain pattern StrategyCandidate falls back to the helper.
            regimes = getattr(cand, "allowed_regimes", None)
            if regimes is None:
                regimes = allowed_regimes_for(cand)
            for regime in regimes:
                buckets[regime].append(cand)
        # Sort each bucket by |edge| descending.
        for regime, lst in buckets.items():
            lst.sort(key=_edge_magnitude, reverse=True)
            self._by_regime[regime] = lst[: self._top_k]

    # ── Quality gate ──────────────────────────────────────────────────────────

    def _passes_quality(self, candidate: StrategyCandidate) -> bool:
        """
        True iff the candidate's standalone metrics meet the quality bar.

        Returns True unconditionally when no metrics were supplied —
        i.e. the v1 ranking-only behaviour is preserved for callers that
        omit `candidate_metrics`.
        """
        if not self._metrics:
            return True
        rep = self._metrics.get(candidate.name)
        if rep is None:
            # No metrics for this candidate → conservative: block.
            return False
        pf = rep.profit_factor
        if pf is None or pf < self._min_pf:
            return False
        if rep.expectancy <= self._min_exp:
            return False
        return True

    # ── Public API ────────────────────────────────────────────────────────────

    def active_for(
        self, regime: str
    ) -> "StrategyCandidate | _NoTradeSentinel | None":
        """
        Return the active candidate for a regime, or a sentinel.

        Returns
        -------
        StrategyCandidate
            Top-ranked candidate that also passed the quality gate.
        NO_TRADE
            A candidate is mapped to this regime, but the top one fails
            the quality bar (`profit_factor < min_profit_factor` or
            `expectancy <= min_expectancy`).  Caller MUST skip trading.
        None
            No candidate is mapped to this regime at all.
        """
        lst = self._by_regime.get(regime, [])
        if not lst:
            return None
        top = lst[0]
        if not self._passes_quality(top):
            return NO_TRADE
        return top

    def candidates_for(self, regime: str) -> list[StrategyCandidate]:
        """Return the full ranked shortlist for a regime (no quality gating)."""
        return list(self._by_regime.get(regime, []))

    def plan(self) -> dict[str, str | None]:
        """
        Regime → active candidate name, or "NO_TRADE", or None.

        A string-valued plan keeps the report human-readable while still
        distinguishing the two reasons a regime might not trade.
        """
        out: dict[str, str | None] = {}
        for regime in ALL_REGIMES:
            active = self.active_for(regime)
            if active is None:
                out[regime] = None
            elif active is NO_TRADE:
                out[regime] = "NO_TRADE"
            else:
                out[regime] = active.name
        return out


__all__ = [
    "StrategySelector",
    "allowed_regimes_for",
    "allowed_regimes_for_sequence",
    "EXTREME_VWAP_BINS",
    "NO_TRADE",
]
