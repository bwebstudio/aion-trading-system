"""
research.pattern_strategies.pattern_to_strategy
────────────────────────────────────────────────
Convert discovery results into StrategyCandidate instances.

Direction inference
───────────────────
mean_test_return > 0  → LONG
mean_test_return < 0  → SHORT
mean_test_return = 0  → rejected (no candidate)

Default execution rules
───────────────────────
stop_rule : 1.5 × ATR(14) against direction
exit_rule : take-profit at 2.5 × ATR(14) in direction, or
            forced exit after 20 bars (whichever comes first)

Both rules are overridable via the kwargs on the converter functions.
"""

from __future__ import annotations

from typing import Any

from research.pattern_discovery.multi_asset_validator import (
    MultiAssetPatternResult,
)
from research.pattern_discovery.pattern_types import (
    CompactPatternResult,
    PatternKey,
)
from research.pattern_strategies.strategy_candidate import StrategyCandidate


DEFAULT_STOP_MULT = 1.5
DEFAULT_TP_MULT = 2.5
DEFAULT_ATR_PERIOD = 14
DEFAULT_MAX_HOLD = 20


def _infer_direction(mean_test_return: float) -> str | None:
    if mean_test_return > 0:
        return "LONG"
    if mean_test_return < 0:
        return "SHORT"
    return None


def _entry_rule(pattern_key: PatternKey) -> dict[str, Any]:
    return {
        "type": "AND",
        "conditions": [
            {"column": col, "equals": val} for col, val in pattern_key
        ],
    }


def _default_stop_rule(
    stop_mult: float = DEFAULT_STOP_MULT,
    atr_period: int = DEFAULT_ATR_PERIOD,
) -> dict[str, Any]:
    return {
        "type": "atr_multiplier",
        "period": atr_period,
        "multiplier": stop_mult,
    }


def _default_exit_rule(
    tp_mult: float = DEFAULT_TP_MULT,
    atr_period: int = DEFAULT_ATR_PERIOD,
    max_hold: int = DEFAULT_MAX_HOLD,
) -> dict[str, Any]:
    return {
        "take_profit": {
            "type": "atr_multiplier",
            "period": atr_period,
            "multiplier": tp_mult,
        },
        "max_hold_bars": max_hold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Converters
# ─────────────────────────────────────────────────────────────────────────────


def convert_compact_result(
    result: CompactPatternResult,
    *,
    stop_mult: float = DEFAULT_STOP_MULT,
    tp_mult: float = DEFAULT_TP_MULT,
    atr_period: int = DEFAULT_ATR_PERIOD,
    max_hold: int = DEFAULT_MAX_HOLD,
) -> StrategyCandidate | None:
    """Convert a single-asset CompactPatternResult."""
    direction = _infer_direction(result.test_mean_return)
    if direction is None:
        return None

    expected_edge = {
        "sample_size": result.sample_size,
        "mean_test_return": float(result.test_mean_return),
        "mean_train_return": float(result.train_mean_return),
        "train_win_rate": float(result.train_win_rate),
        "test_win_rate": float(result.test_win_rate),
        "stability_score": (
            float(result.stability_score)
            if result.stability_score is not None
            else None
        ),
        "score": float(result.score) if result.score is not None else None,
        "assets_found": [],
    }

    return StrategyCandidate(
        pattern_key=result.key,
        direction=direction,
        entry_rule=_entry_rule(result.key),
        stop_rule=_default_stop_rule(stop_mult, atr_period),
        exit_rule=_default_exit_rule(tp_mult, atr_period, max_hold),
        expected_edge=expected_edge,
    )


def convert_multi_asset(
    result: MultiAssetPatternResult,
    *,
    stop_mult: float = DEFAULT_STOP_MULT,
    tp_mult: float = DEFAULT_TP_MULT,
    atr_period: int = DEFAULT_ATR_PERIOD,
    max_hold: int = DEFAULT_MAX_HOLD,
) -> StrategyCandidate | None:
    """Convert a cross-asset MultiAssetPatternResult."""
    direction = _infer_direction(result.mean_test_return)
    if direction is None:
        return None

    expected_edge = {
        "sample_size": result.total_samples,
        "mean_test_return": float(result.mean_test_return),
        "mean_score_across_assets": float(result.mean_score_across_assets),
        "sign_agreement": bool(result.sign_agreement),
        "assets_found": list(result.assets_found),
        "per_asset": [
            {
                "asset": s.asset,
                "sample_size": s.sample_size,
                "test_mean_return": float(s.test_mean_return),
                "test_win_rate": float(s.test_win_rate),
                "stability_score": (
                    float(s.stability_score)
                    if s.stability_score is not None
                    else None
                ),
            }
            for s in result.per_asset_stats
        ],
    }

    return StrategyCandidate(
        pattern_key=result.pattern_key,
        direction=direction,
        entry_rule=_entry_rule(result.pattern_key),
        stop_rule=_default_stop_rule(stop_mult, atr_period),
        exit_rule=_default_exit_rule(tp_mult, atr_period, max_hold),
        expected_edge=expected_edge,
    )
