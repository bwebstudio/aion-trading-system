"""
aion/analytics/baseline_selection.py
──────────────────────────────────────
Select the best parameter configuration from a sweep and build a
StrategyBaselineProfile for downstream use by Risk Allocation v1.

Public API:
  select_best_opening_range_config(comparison, *, min_candidates=1)
      -> tuple[SweepPoint, StrategyBaselineProfile] | None
  select_best_vwap_fade_config(comparison, *, min_candidates=1)
      -> tuple[VWAPSweepPoint, StrategyBaselineProfile] | None
  rank_sweep_configs(results, *, min_candidates=1)
      -> list[tuple[float, str, ReplayMetrics]]

Composite scoring (transparent, no hidden weights)
────────────────────────────────────────────────────
  base_score = win_rate_on_activated * activation_rate

  This rewards both signal quality (win_rate) and signal frequency
  (activation_rate).  A strategy that fires rarely or always loses scores 0.

  When avg_mfe and avg_mae are available, an expectancy bonus is added:
    expectancy = win_rate * avg_mfe - (1 - win_rate) * avg_mae
    bonus = max(0, expectancy / 100)   # normalised: 10 pips exp ≈ +0.10

  The bonus is additive and capped to 0 for negative expectancy, so
  win_rate × activation_rate remains the dominant term.

  When win_rate or activation_rate is None (no candidates / no activations),
  the point is excluded from ranked output entirely.
"""

from __future__ import annotations

from pydantic import BaseModel

from aion.analytics.replay_models import (
    ReplayMetrics,
    SweepComparison,
    SweepPoint,
    VWAPSweepComparison,
    VWAPSweepPoint,
)


# ─────────────────────────────────────────────────────────────────────────────
# Output model
# ─────────────────────────────────────────────────────────────────────────────


class StrategyBaselineProfile(BaseModel, frozen=True):
    """
    Summary profile for one strategy configuration.

    Serves as input for Risk Allocation v1.  Contains the minimum set of
    metrics needed to size positions, set risk budgets, and evaluate expected
    value under different market conditions.

    Fields
    ------
    strategy_id : str
        Stable identifier matching StrategyEngine.strategy_id for this config.
    session : str
        Session name the configuration targets ('LONDON', 'ALL', etc.).
    activation_rate : float | None
        Fraction of signal candidates where entry was activated.
        None when no candidates were generated.
    win_rate : float | None
        Win rate over activated entries only.
        None when no entries were activated.
    avg_mfe : float | None
        Average Maximum Favorable Excursion in pips over activated entries.
        None when no entries were activated.
    avg_mae : float | None
        Average Maximum Adverse Excursion in pips over activated entries.
        None when no entries were activated.
    expected_resolution_bars : float | None
        Average bars to trade resolution (WIN or LOSS) over activated entries.
        Useful for estimating capital tie-up duration.
        None when no entries were activated.
    """

    strategy_id: str
    session: str
    activation_rate: float | None
    win_rate: float | None
    avg_mfe: float | None
    avg_mae: float | None
    expected_resolution_bars: float | None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def rank_sweep_configs(
    results: list[tuple[str, ReplayMetrics]],
    *,
    min_candidates: int = 1,
) -> list[tuple[float, str, ReplayMetrics]]:
    """Rank (label, metrics) pairs by composite score, descending.

    Parameters
    ----------
    results:
        List of (label, metrics) pairs to evaluate.  Typically extracted
        from a SweepComparison or VWAPSweepComparison.
    min_candidates:
        Minimum candidate_count required for a point to be included.
        Points below this threshold are excluded from the output.

    Returns
    -------
    list[tuple[float, str, ReplayMetrics]]
        (score, label, metrics) sorted descending by score.
        Only entries with candidate_count >= min_candidates and a non-None
        composite score are included.
    """
    ranked = []
    for label, metrics in results:
        if metrics.candidate_count < min_candidates:
            continue
        score = _composite_score(metrics)
        if score is None:
            continue
        ranked.append((score, label, metrics))

    return sorted(ranked, key=lambda x: x[0], reverse=True)


def select_best_opening_range_config(
    comparison: SweepComparison,
    *,
    min_candidates: int = 1,
) -> tuple[SweepPoint, StrategyBaselineProfile] | None:
    """Select the highest-scoring SweepPoint from an OR parameter sweep.

    Parameters
    ----------
    comparison:
        Result of run_parameter_sweep().
    min_candidates:
        Minimum candidate_count for a point to be eligible.

    Returns
    -------
    tuple[SweepPoint, StrategyBaselineProfile] | None
        The best point and its baseline profile, or None when no point
        meets the minimum candidate threshold.
    """
    pairs = [(r.sweep_point.label, r.metrics) for r in comparison.results]
    ranked = rank_sweep_configs(pairs, min_candidates=min_candidates)

    if not ranked:
        return None

    _score, best_label, best_metrics = ranked[0]
    best_point = next(
        r.sweep_point
        for r in comparison.results
        if r.sweep_point.label == best_label
    )

    profile = _build_profile(
        strategy_id=f"opening_range_{best_label}",
        session=_or_session_label(best_point),
        metrics=best_metrics,
    )
    return best_point, profile


def select_best_vwap_fade_config(
    comparison: VWAPSweepComparison,
    *,
    min_candidates: int = 1,
) -> tuple[VWAPSweepPoint, StrategyBaselineProfile] | None:
    """Select the highest-scoring VWAPSweepPoint from a VWAP Fade sweep.

    Parameters
    ----------
    comparison:
        Result of run_vwap_parameter_sweep().
    min_candidates:
        Minimum candidate_count for a point to be eligible.

    Returns
    -------
    tuple[VWAPSweepPoint, StrategyBaselineProfile] | None
        The best point and its baseline profile, or None when no point
        meets the minimum candidate threshold.
    """
    pairs = [(r.sweep_point.label, r.metrics) for r in comparison.results]
    ranked = rank_sweep_configs(pairs, min_candidates=min_candidates)

    if not ranked:
        return None

    _score, best_label, best_metrics = ranked[0]
    best_point = next(
        r.sweep_point
        for r in comparison.results
        if r.sweep_point.label == best_label
    )

    profile = _build_profile(
        strategy_id=f"vwap_fade_{best_label}",
        session=best_point.session_name,
        metrics=best_metrics,
    )
    return best_point, profile


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _composite_score(metrics: ReplayMetrics) -> float | None:
    """Compute composite score for a single ReplayMetrics instance.

    Returns None when win_rate or activation_rate is unavailable (which
    happens when candidate_count == 0 or entry_activated_count == 0).
    """
    wr = metrics.win_rate_on_activated
    ar = metrics.activation_rate
    if wr is None or ar is None:
        return None

    base = wr * ar

    # Expectancy bonus: positive expectancy rewards quality, capped at 0 below.
    if metrics.avg_mfe is not None and metrics.avg_mae is not None:
        loss_rate = 1.0 - wr
        expectancy = wr * metrics.avg_mfe - loss_rate * metrics.avg_mae
        base += max(0.0, expectancy / 100.0)

    return base


def _build_profile(
    strategy_id: str,
    session: str,
    metrics: ReplayMetrics,
) -> StrategyBaselineProfile:
    return StrategyBaselineProfile(
        strategy_id=strategy_id,
        session=session,
        activation_rate=metrics.activation_rate,
        win_rate=metrics.win_rate_on_activated,
        avg_mfe=metrics.avg_mfe,
        avg_mae=metrics.avg_mae,
        expected_resolution_bars=metrics.avg_bars_to_resolution,
    )


def _or_session_label(point: SweepPoint) -> str:
    """Derive a readable session label from a SweepPoint.

    When allowed_sessions is None (no SessionFilter), the strategy runs on
    all sessions → label is 'ALL'.  Otherwise join the session names.
    """
    if point.allowed_sessions is None:
        return "ALL"
    return "+".join(sorted(point.allowed_sessions))
