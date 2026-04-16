"""
research.sequential_strategies.sequence_to_strategy
────────────────────────────────────────────────────
Convert a discovered SequenceResult (or its JSON dict form) into a
SequentialStrategyCandidate.

Direction inference
───────────────────
    mean_return > 0 → LONG
    mean_return < 0 → SHORT
    mean_return = 0 → rejected (returns None)

Default execution rules (overridable)
────────────────────────────────────
    stop_rule  : 1.5 * ATR(14) against direction
    exit_rule  : take-profit 2.5 * ATR(14) in direction,
                 OR forced exit after 20 bars

Both are identical to the pattern-based candidate defaults so the
downstream meta-strategy treats both kinds uniformly.
"""

from __future__ import annotations

from typing import Any

from research.sequential_discovery.sequence_evaluator import SequenceResult
from research.sequential_strategies.strategy_candidate import (
    SequenceKey,
    SequentialStrategyCandidate,
)


DEFAULT_STOP_MULT = 1.5
DEFAULT_TP_MULT = 2.5
DEFAULT_ATR_PERIOD = 14
DEFAULT_MAX_HOLD = 20


# ─────────────────────────────────────────────────────────────────────────────
# Rule builders
# ─────────────────────────────────────────────────────────────────────────────


def _entry_rule(sequence_key: SequenceKey) -> dict[str, Any]:
    """
    Build the entry rule.  Entry fires on the bar AFTER the final step
    (step L-1) is matched — i.e. at row (i + L) open, where i is the
    row where step 0 was observed.
    """
    return {
        "type": "SEQUENCE",
        "trigger": "after_last_step",
        "steps": [
            {"column": col, "equals": val} for col, val in sequence_key
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


def _infer_direction(mean_return: float) -> str | None:
    if mean_return > 0:
        return "LONG"
    if mean_return < 0:
        return "SHORT"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Converters
# ─────────────────────────────────────────────────────────────────────────────


def convert_sequence_result(
    result: SequenceResult,
    *,
    stop_mult: float = DEFAULT_STOP_MULT,
    tp_mult: float = DEFAULT_TP_MULT,
    atr_period: int = DEFAULT_ATR_PERIOD,
    max_hold: int = DEFAULT_MAX_HOLD,
) -> SequentialStrategyCandidate | None:
    """Build a SequentialStrategyCandidate from a SequenceResult."""
    direction = _infer_direction(result.mean_return)
    if direction is None:
        return None

    edge = {
        "sample_size": result.n_samples,
        "mean_return": float(result.mean_return),
        "expectancy": float(result.expectancy),
        "winrate": float(result.winrate),
        "profit_factor": (
            float(result.profit_factor)
            if result.profit_factor is not None
            else None
        ),
        "score": float(result.score),
        "length": result.length,
        "train_n": result.train_n,
        "test_n": result.test_n,
        "train_mean": (
            float(result.train_mean) if result.train_mean is not None else None
        ),
        "test_mean": (
            float(result.test_mean) if result.test_mean is not None else None
        ),
        "stability": (
            float(result.stability) if result.stability is not None else None
        ),
    }

    return SequentialStrategyCandidate(
        sequence_key=result.sequence,
        direction=direction,
        entry_rule=_entry_rule(result.sequence),
        stop_rule=_default_stop_rule(stop_mult, atr_period),
        exit_rule=_default_exit_rule(tp_mult, atr_period, max_hold),
        expected_edge=edge,
    )


def convert_sequence_dict(
    d: dict[str, Any],
    *,
    stop_mult: float = DEFAULT_STOP_MULT,
    tp_mult: float = DEFAULT_TP_MULT,
    atr_period: int = DEFAULT_ATR_PERIOD,
    max_hold: int = DEFAULT_MAX_HOLD,
) -> SequentialStrategyCandidate | None:
    """
    Build a candidate from the dict form stored in
    research/output/sequential_edges.json.
    """
    mean = d.get("mean_return")
    if mean is None:
        return None
    direction = _infer_direction(float(mean))
    if direction is None:
        return None

    sequence_key: SequenceKey = tuple(tuple(step) for step in d["sequence"])

    edge = {
        "sample_size": d.get("n_samples"),
        "mean_return": float(mean),
        "expectancy": float(d.get("expectancy", mean)),
        "winrate": float(d.get("winrate", 0.0)),
        "profit_factor": (
            float(d["profit_factor"])
            if d.get("profit_factor") is not None
            else None
        ),
        "score": float(d.get("score", 0.0)),
        "length": int(d.get("length", len(sequence_key))),
        "train_n": int(d.get("train_n", 0)),
        "test_n": int(d.get("test_n", 0)),
        "train_mean": (
            float(d["train_mean"]) if d.get("train_mean") is not None else None
        ),
        "test_mean": (
            float(d["test_mean"]) if d.get("test_mean") is not None else None
        ),
        "stability": (
            float(d["stability"]) if d.get("stability") is not None else None
        ),
    }

    return SequentialStrategyCandidate(
        sequence_key=sequence_key,
        direction=direction,
        entry_rule=_entry_rule(sequence_key),
        stop_rule=_default_stop_rule(stop_mult, atr_period),
        exit_rule=_default_exit_rule(tp_mult, atr_period, max_hold),
        expected_edge=edge,
    )


__all__ = [
    "convert_sequence_result",
    "convert_sequence_dict",
    "DEFAULT_STOP_MULT",
    "DEFAULT_TP_MULT",
    "DEFAULT_ATR_PERIOD",
    "DEFAULT_MAX_HOLD",
]
