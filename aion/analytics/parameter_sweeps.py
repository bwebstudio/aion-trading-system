"""
aion/analytics/parameter_sweeps.py
────────────────────────────────────
Grid-search over OpeningRangeEngine and VWAPFadeEngine parameters.

Public API:
  run_parameter_sweep(snapshots, sweep_points, *, regime_detector=None)
      -> SweepComparison
  run_vwap_parameter_sweep(snapshots, sweep_points, *, regime_detector=None)
      -> VWAPSweepComparison

Each SweepPoint / VWAPSweepPoint defines one parameter combination.  The
function builds the appropriate engine (optionally wrapped by filters),
runs a full replay, and computes metrics.

Engine construction order for OR (outermost first, checked first):
  SpreadFilter → SessionFilter → OpeningRangeEngine (inner)

Engine construction order for VWAP (outermost first, checked first):
  SpreadFilter → VWAPFadeEngine (session handled internally by the engine)
"""

from __future__ import annotations

from aion.analytics.replay_metrics import compute_metrics
from aion.analytics.replay_models import (
    SweepComparison,
    SweepPoint,
    SweepResult,
    VWAPSweepComparison,
    VWAPSweepPoint,
    VWAPSweepResult,
)
from aion.core.constants import MIN_QUALITY_SCORE
from aion.core.models import MarketSnapshot
from aion.regime.base import RegimeDetector
from aion.replay.models import LabelConfig
from aion.replay.runner import run_replay
from aion.strategies.base import StrategyEngine
from aion.strategies.filters import SessionFilter, SpreadFilter
from aion.strategies.models import OpeningRangeDefinition
from aion.strategies.opening_range import OpeningRangeEngine
from aion.strategies.vwap_fade import VWAPFadeDefinition, VWAPFadeEngine


def run_parameter_sweep(
    snapshots: list[MarketSnapshot],
    sweep_points: list[SweepPoint],
    *,
    regime_detector: RegimeDetector | None = None,
) -> SweepComparison:
    """Run replay for each SweepPoint and return all results.

    Parameters
    ----------
    snapshots:
        Historical snapshots (same list re-used for every point).
    sweep_points:
        List of parameter combinations to evaluate.
    regime_detector:
        Optional regime detector shared across all sweep runs.

    Returns
    -------
    SweepComparison
        Immutable container with one SweepResult per point, sortable
        via .ranked_by_win_rate(), .ranked_by_candidate_count(), etc.
    """
    results: list[SweepResult] = []

    for point in sweep_points:
        engine = _build_engine(point)
        label_cfg = LabelConfig(
            stop_pips=point.stop_pips,
            target_pips=point.target_pips,
            max_bars=point.max_label_bars,
        )
        replay_result = run_replay(
            snapshots,
            engine,
            regime_detector=regime_detector,
            label_config=label_cfg,
        )
        metrics = compute_metrics(replay_result.records, replay_result.labeled_outcomes)
        results.append(SweepResult(sweep_point=point, metrics=metrics))

    return SweepComparison(results=results)


def run_vwap_parameter_sweep(
    snapshots: list[MarketSnapshot],
    sweep_points: list[VWAPSweepPoint],
    *,
    regime_detector: RegimeDetector | None = None,
) -> VWAPSweepComparison:
    """Run replay for each VWAPSweepPoint and return all results.

    Parameters
    ----------
    snapshots:
        Historical snapshots (same list re-used for every point).
    sweep_points:
        List of VWAP Fade parameter combinations to evaluate.
    regime_detector:
        Optional regime detector shared across all sweep runs.

    Returns
    -------
    VWAPSweepComparison
        Immutable container with one VWAPSweepResult per point, sortable
        via .ranked_by_win_rate(), .ranked_by_candidate_count(), etc.
    """
    results: list[VWAPSweepResult] = []

    for point in sweep_points:
        engine = _build_vwap_engine(point)
        label_cfg = LabelConfig(
            stop_pips=point.stop_pips,
            target_pips=point.target_pips,
            max_bars=point.max_label_bars,
        )
        replay_result = run_replay(
            snapshots,
            engine,
            regime_detector=regime_detector,
            label_config=label_cfg,
        )
        metrics = compute_metrics(replay_result.records, replay_result.labeled_outcomes)
        results.append(VWAPSweepResult(sweep_point=point, metrics=metrics))

    return VWAPSweepComparison(results=results)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_engine(point: SweepPoint) -> StrategyEngine:
    """Construct a (possibly filtered) strategy engine from a SweepPoint.

    Build order (inner → outer):
      OpeningRangeEngine
        → SessionFilter  (if allowed_sessions is not None)
          → SpreadFilter (if max_spread_pips is not None)
    """
    or_def = OpeningRangeDefinition(
        strategy_id=f"sweep_{point.label}",
        session_name="LONDON",
        min_range_pips=point.min_range_pips,
        max_range_pips=point.max_range_pips,
        direction_bias=point.direction_bias,
        require_completed_range=point.require_completed_range,
        max_retest_penetration_points=point.max_retest_penetration_points,
    )
    engine: StrategyEngine = OpeningRangeEngine(
        or_def,
        min_quality_score=MIN_QUALITY_SCORE,
    )

    if point.allowed_sessions is not None:
        engine = SessionFilter(engine, allowed_sessions=set(point.allowed_sessions))

    if point.max_spread_pips is not None:
        engine = SpreadFilter(engine, max_spread_pips=point.max_spread_pips)

    return engine


def _build_vwap_engine(point: VWAPSweepPoint) -> StrategyEngine:
    """Construct a (possibly filtered) VWAP Fade engine from a VWAPSweepPoint.

    Build order (inner → outer):
      VWAPFadeEngine
        → SpreadFilter (if max_spread_pips is not None)

    Session targeting is handled inside VWAPFadeDefinition via session_name.
    """
    vwap_def = VWAPFadeDefinition(
        strategy_id=f"sweep_{point.label}",
        session_name=point.session_name,
        min_distance_to_vwap_pips=point.min_distance_to_vwap_pips,
        max_distance_to_vwap_pips=point.max_distance_to_vwap_pips,
        require_rejection=point.require_rejection,
        direction_bias=point.direction_bias,
    )
    engine: StrategyEngine = VWAPFadeEngine(
        vwap_def,
        min_quality_score=MIN_QUALITY_SCORE,
    )

    if point.max_spread_pips is not None:
        engine = SpreadFilter(engine, max_spread_pips=point.max_spread_pips)

    return engine
