"""
tests/unit/test_baseline_selection.py
───────────────────────────────────────
Unit tests for aion.analytics.baseline_selection.

Tests verify:
  - _composite_score returns None when win_rate or activation_rate is None
  - _composite_score increases with win_rate and activation_rate
  - expectancy bonus is applied only when avg_mfe/avg_mae are present
  - expectancy bonus is zero when expectancy is negative
  - rank_sweep_configs returns results sorted descending by score
  - rank_sweep_configs excludes entries below min_candidates
  - rank_sweep_configs excludes entries with None scores
  - select_best_opening_range_config returns the top-ranked SweepPoint
  - select_best_opening_range_config returns None when no valid points exist
  - select_best_vwap_fade_config returns the top-ranked VWAPSweepPoint
  - select_best_vwap_fade_config returns None when no valid points exist
  - StrategyBaselineProfile fields match the selected metrics
  - StrategyBaselineProfile is frozen
  - _or_session_label derives correct session string from SweepPoint
"""

from __future__ import annotations

import pytest

from aion.analytics.baseline_selection import (
    StrategyBaselineProfile,
    _composite_score,  # noqa: PLC2701
    _or_session_label,  # noqa: PLC2701
    rank_sweep_configs,
    select_best_opening_range_config,
    select_best_vwap_fade_config,
)
from aion.analytics.replay_models import (
    ReplayMetrics,
    SweepComparison,
    SweepPoint,
    SweepResult,
    VWAPSweepComparison,
    VWAPSweepPoint,
    VWAPSweepResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for building minimal ReplayMetrics
# ─────────────────────────────────────────────────────────────────────────────


def _metrics(
    *,
    candidate_count: int = 10,
    entry_activated_count: int = 8,
    win_count: int = 5,
    activation_rate: float | None = 0.8,
    win_rate_on_activated: float | None = 0.625,
    avg_mfe: float | None = 20.0,
    avg_mae: float | None = 10.0,
    avg_bars_to_resolution: float | None = 5.0,
) -> ReplayMetrics:
    loss_count = entry_activated_count - win_count
    return ReplayMetrics(
        total_records=100,
        no_trade_count=90 - candidate_count,
        candidate_count=candidate_count,
        insufficient_data_count=0,
        total_labeled=entry_activated_count + (candidate_count - entry_activated_count),
        entry_activated_count=entry_activated_count,
        win_count=win_count,
        loss_count=loss_count,
        timeout_count=0,
        entry_not_activated_count=candidate_count - entry_activated_count,
        activation_rate=activation_rate,
        win_rate_on_activated=win_rate_on_activated,
        avg_mfe=avg_mfe,
        avg_mae=avg_mae,
        avg_bars_to_entry=1.0,
        avg_bars_to_resolution=avg_bars_to_resolution,
    )


def _zero_metrics() -> ReplayMetrics:
    return ReplayMetrics(
        total_records=50,
        no_trade_count=50,
        candidate_count=0,
        insufficient_data_count=0,
        total_labeled=0,
        entry_activated_count=0,
        win_count=0,
        loss_count=0,
        timeout_count=0,
        entry_not_activated_count=0,
        activation_rate=None,
        win_rate_on_activated=None,
        avg_mfe=None,
        avg_mae=None,
        avg_bars_to_entry=None,
        avg_bars_to_resolution=None,
    )


def _sweep_point(label: str) -> SweepPoint:
    return SweepPoint(label=label, min_range_pips=5.0, max_range_pips=40.0)


def _vwap_sweep_point(label: str) -> VWAPSweepPoint:
    return VWAPSweepPoint(label=label, min_distance_to_vwap_pips=10.0)


# ─────────────────────────────────────────────────────────────────────────────
# _composite_score
# ─────────────────────────────────────────────────────────────────────────────


def test_composite_score_returns_none_when_win_rate_none():
    m = _metrics(win_rate_on_activated=None, activation_rate=0.5)
    assert _composite_score(m) is None


def test_composite_score_returns_none_when_activation_rate_none():
    m = _metrics(win_rate_on_activated=0.6, activation_rate=None)
    assert _composite_score(m) is None


def test_composite_score_base_is_product():
    m = _metrics(win_rate_on_activated=0.5, activation_rate=0.4, avg_mfe=None, avg_mae=None)
    score = _composite_score(m)
    assert score == pytest.approx(0.5 * 0.4)


def test_composite_score_increases_with_win_rate():
    low_wr = _metrics(win_rate_on_activated=0.4, activation_rate=0.5, avg_mfe=None, avg_mae=None)
    high_wr = _metrics(win_rate_on_activated=0.8, activation_rate=0.5, avg_mfe=None, avg_mae=None)
    assert _composite_score(high_wr) > _composite_score(low_wr)


def test_composite_score_increases_with_activation_rate():
    low_ar = _metrics(win_rate_on_activated=0.6, activation_rate=0.2, avg_mfe=None, avg_mae=None)
    high_ar = _metrics(win_rate_on_activated=0.6, activation_rate=0.8, avg_mfe=None, avg_mae=None)
    assert _composite_score(high_ar) > _composite_score(low_ar)


def test_composite_score_positive_expectancy_adds_bonus():
    no_exp = _metrics(
        win_rate_on_activated=0.6, activation_rate=0.5, avg_mfe=None, avg_mae=None
    )
    with_exp = _metrics(
        win_rate_on_activated=0.6, activation_rate=0.5, avg_mfe=20.0, avg_mae=10.0
    )
    assert _composite_score(with_exp) > _composite_score(no_exp)


def test_composite_score_negative_expectancy_no_bonus():
    """Negative expectancy should not reduce the base score (bonus capped at 0)."""
    no_exp = _metrics(
        win_rate_on_activated=0.3, activation_rate=0.5, avg_mfe=None, avg_mae=None
    )
    neg_exp = _metrics(
        win_rate_on_activated=0.3, activation_rate=0.5, avg_mfe=5.0, avg_mae=30.0
    )
    # Negative expectancy: 0.3*5 - 0.7*30 = 1.5 - 21 = -19.5 → bonus = 0
    assert _composite_score(neg_exp) == pytest.approx(_composite_score(no_exp))


# ─────────────────────────────────────────────────────────────────────────────
# rank_sweep_configs
# ─────────────────────────────────────────────────────────────────────────────


def test_rank_sweep_configs_sorted_descending():
    results = [
        ("low", _metrics(win_rate_on_activated=0.3, activation_rate=0.3, avg_mfe=None, avg_mae=None)),
        ("high", _metrics(win_rate_on_activated=0.9, activation_rate=0.8, avg_mfe=None, avg_mae=None)),
        ("mid", _metrics(win_rate_on_activated=0.6, activation_rate=0.5, avg_mfe=None, avg_mae=None)),
    ]
    ranked = rank_sweep_configs(results)
    labels = [label for _, label, _ in ranked]
    assert labels[0] == "high"
    assert labels[-1] == "low"


def test_rank_sweep_configs_excludes_below_min_candidates():
    results = [
        ("enough", _metrics(candidate_count=10, win_rate_on_activated=0.7, activation_rate=0.5,
                             avg_mfe=None, avg_mae=None)),
        ("too_few", _metrics(candidate_count=0, win_rate_on_activated=None, activation_rate=None,
                              avg_mfe=None, avg_mae=None)),
    ]
    ranked = rank_sweep_configs(results, min_candidates=5)
    labels = [label for _, label, _ in ranked]
    assert "too_few" not in labels
    assert "enough" in labels


def test_rank_sweep_configs_excludes_none_scores():
    """Entries with None win_rate are excluded entirely."""
    results = [
        ("no_score", _zero_metrics()),
        ("ok", _metrics(win_rate_on_activated=0.6, activation_rate=0.4, avg_mfe=None, avg_mae=None)),
    ]
    ranked = rank_sweep_configs(results, min_candidates=0)
    labels = [label for _, label, _ in ranked]
    assert "no_score" not in labels


def test_rank_sweep_configs_empty_input():
    assert rank_sweep_configs([]) == []


def test_rank_sweep_configs_returns_tuples():
    results = [("a", _metrics(avg_mfe=None, avg_mae=None))]
    ranked = rank_sweep_configs(results)
    score, label, metrics = ranked[0]
    assert isinstance(score, float)
    assert label == "a"
    assert isinstance(metrics, ReplayMetrics)


# ─────────────────────────────────────────────────────────────────────────────
# select_best_opening_range_config
# ─────────────────────────────────────────────────────────────────────────────


def test_select_best_or_config_returns_best_point():
    good = SweepResult(
        sweep_point=_sweep_point("good"),
        metrics=_metrics(win_rate_on_activated=0.8, activation_rate=0.7, avg_mfe=None, avg_mae=None),
    )
    bad = SweepResult(
        sweep_point=_sweep_point("bad"),
        metrics=_metrics(win_rate_on_activated=0.3, activation_rate=0.2, avg_mfe=None, avg_mae=None),
    )
    cmp = SweepComparison(results=[bad, good])
    result = select_best_opening_range_config(cmp)
    assert result is not None
    best_point, profile = result
    assert best_point.label == "good"


def test_select_best_or_config_returns_none_when_all_filtered():
    """All points have 0 candidates → nothing passes min_candidates filter."""
    pt = SweepResult(sweep_point=_sweep_point("empty"), metrics=_zero_metrics())
    cmp = SweepComparison(results=[pt])
    result = select_best_opening_range_config(cmp, min_candidates=1)
    assert result is None


def test_select_best_or_config_baseline_profile_fields():
    m = _metrics(
        win_rate_on_activated=0.7,
        activation_rate=0.6,
        avg_mfe=15.0,
        avg_mae=8.0,
        avg_bars_to_resolution=4.0,
    )
    cmp = SweepComparison(results=[SweepResult(sweep_point=_sweep_point("p"), metrics=m)])
    result = select_best_opening_range_config(cmp)
    assert result is not None
    _, profile = result
    assert isinstance(profile, StrategyBaselineProfile)
    assert profile.win_rate == pytest.approx(0.7)
    assert profile.activation_rate == pytest.approx(0.6)
    assert profile.avg_mfe == pytest.approx(15.0)
    assert profile.avg_mae == pytest.approx(8.0)
    assert profile.expected_resolution_bars == pytest.approx(4.0)


def test_select_best_or_config_session_all_when_no_filter():
    """SweepPoint with allowed_sessions=None → session label 'ALL'."""
    pt = SweepResult(
        sweep_point=SweepPoint(label="all_sess", allowed_sessions=None),
        metrics=_metrics(avg_mfe=None, avg_mae=None),
    )
    cmp = SweepComparison(results=[pt])
    result = select_best_opening_range_config(cmp)
    assert result is not None
    _, profile = result
    assert profile.session == "ALL"


def test_select_best_or_config_session_from_allowed_sessions():
    pt = SweepResult(
        sweep_point=SweepPoint(
            label="london_only",
            allowed_sessions=frozenset({"LONDON", "OVERLAP_LONDON_NY"}),
        ),
        metrics=_metrics(avg_mfe=None, avg_mae=None),
    )
    cmp = SweepComparison(results=[pt])
    result = select_best_opening_range_config(cmp)
    assert result is not None
    _, profile = result
    assert "LONDON" in profile.session


# ─────────────────────────────────────────────────────────────────────────────
# select_best_vwap_fade_config
# ─────────────────────────────────────────────────────────────────────────────


def test_select_best_vwap_config_returns_best_point():
    good = VWAPSweepResult(
        sweep_point=_vwap_sweep_point("good"),
        metrics=_metrics(win_rate_on_activated=0.75, activation_rate=0.6, avg_mfe=None, avg_mae=None),
    )
    bad = VWAPSweepResult(
        sweep_point=_vwap_sweep_point("bad"),
        metrics=_metrics(win_rate_on_activated=0.4, activation_rate=0.3, avg_mfe=None, avg_mae=None),
    )
    cmp = VWAPSweepComparison(results=[bad, good])
    result = select_best_vwap_fade_config(cmp)
    assert result is not None
    best_point, profile = result
    assert best_point.label == "good"


def test_select_best_vwap_config_returns_none_when_all_filtered():
    pt = VWAPSweepResult(
        sweep_point=_vwap_sweep_point("empty"),
        metrics=_zero_metrics(),
    )
    cmp = VWAPSweepComparison(results=[pt])
    result = select_best_vwap_fade_config(cmp, min_candidates=1)
    assert result is None


def test_select_best_vwap_config_session_from_sweep_point():
    pt = VWAPSweepResult(
        sweep_point=VWAPSweepPoint(label="overlap", session_name="OVERLAP_LONDON_NY"),
        metrics=_metrics(avg_mfe=None, avg_mae=None),
    )
    cmp = VWAPSweepComparison(results=[pt])
    result = select_best_vwap_fade_config(cmp)
    assert result is not None
    _, profile = result
    assert profile.session == "OVERLAP_LONDON_NY"


def test_select_best_vwap_config_strategy_id_contains_label():
    pt = VWAPSweepResult(
        sweep_point=_vwap_sweep_point("my_config"),
        metrics=_metrics(avg_mfe=None, avg_mae=None),
    )
    cmp = VWAPSweepComparison(results=[pt])
    result = select_best_vwap_fade_config(cmp)
    assert result is not None
    _, profile = result
    assert "my_config" in profile.strategy_id


# ─────────────────────────────────────────────────────────────────────────────
# StrategyBaselineProfile
# ─────────────────────────────────────────────────────────────────────────────


def test_baseline_profile_is_frozen():
    profile = StrategyBaselineProfile(
        strategy_id="test",
        session="LONDON",
        activation_rate=0.5,
        win_rate=0.6,
        avg_mfe=15.0,
        avg_mae=8.0,
        expected_resolution_bars=4.0,
    )
    with pytest.raises(Exception):
        profile.win_rate = 0.9  # type: ignore[misc]


def test_baseline_profile_fields_can_be_none():
    """All numeric fields are optional to handle no-activation cases."""
    profile = StrategyBaselineProfile(
        strategy_id="test",
        session="ALL",
        activation_rate=None,
        win_rate=None,
        avg_mfe=None,
        avg_mae=None,
        expected_resolution_bars=None,
    )
    assert profile.activation_rate is None
    assert profile.win_rate is None


# ─────────────────────────────────────────────────────────────────────────────
# _or_session_label
# ─────────────────────────────────────────────────────────────────────────────


def test_or_session_label_all_when_no_filter():
    pt = SweepPoint(label="x", allowed_sessions=None)
    assert _or_session_label(pt) == "ALL"


def test_or_session_label_joins_sorted_sessions():
    pt = SweepPoint(
        label="x",
        allowed_sessions=frozenset({"OVERLAP_LONDON_NY", "LONDON"}),
    )
    label = _or_session_label(pt)
    assert "LONDON" in label
    assert "OVERLAP_LONDON_NY" in label


def test_or_session_label_single_session():
    pt = SweepPoint(label="x", allowed_sessions=frozenset({"NEW_YORK"}))
    assert _or_session_label(pt) == "NEW_YORK"


# ─────────────────────────────────────────────────────────────────────────────
# max_retest_penetration_points propagation
# ─────────────────────────────────────────────────────────────────────────────


def test_select_best_or_config_preserves_max_retest_penetration_points():
    """The winning SweepPoint carries max_retest_penetration_points intact."""
    pt = SweepResult(
        sweep_point=SweepPoint(
            label="retest_10",
            max_retest_penetration_points=10.0,
        ),
        metrics=_metrics(avg_mfe=None, avg_mae=None),
    )
    cmp = SweepComparison(results=[pt])
    result = select_best_opening_range_config(cmp)
    assert result is not None
    best_point, _ = result
    assert best_point.max_retest_penetration_points == pytest.approx(10.0)


def test_select_best_or_config_retest_none_preserved():
    """SweepPoint with max_retest_penetration_points=None is returned as None."""
    pt = SweepResult(
        sweep_point=SweepPoint(label="no_retest", max_retest_penetration_points=None),
        metrics=_metrics(avg_mfe=None, avg_mae=None),
    )
    cmp = SweepComparison(results=[pt])
    result = select_best_opening_range_config(cmp)
    assert result is not None
    best_point, _ = result
    assert best_point.max_retest_penetration_points is None


def test_best_config_retest_wins_over_no_retest():
    """When a retest-filtered config scores higher it becomes the best config."""
    # Higher win_rate and activation_rate → higher composite score
    good = SweepResult(
        sweep_point=SweepPoint(label="with_retest", max_retest_penetration_points=10.0),
        metrics=_metrics(win_rate_on_activated=0.85, activation_rate=0.75,
                         avg_mfe=None, avg_mae=None),
    )
    bad = SweepResult(
        sweep_point=SweepPoint(label="no_retest", max_retest_penetration_points=None),
        metrics=_metrics(win_rate_on_activated=0.50, activation_rate=0.50,
                         avg_mfe=None, avg_mae=None),
    )
    cmp = SweepComparison(results=[bad, good])
    result = select_best_opening_range_config(cmp)
    assert result is not None
    best_point, _ = result
    assert best_point.label == "with_retest"
    assert best_point.max_retest_penetration_points == pytest.approx(10.0)
