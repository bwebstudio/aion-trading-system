"""
research.meta_strategy.unified
───────────────────────────────
Unified adapter for pattern-based and sequence-based strategy candidates.

Both kinds expose the same duck-typed interface so the meta selector
and backtest loop can treat them uniformly:

    unified.name              str   — prefixed with "[P]" / "[S]"
    unified.direction         str   — "LONG" | "SHORT"
    unified.expected_edge     dict
    unified.allowed_regimes   frozenset[str]
    unified.candidate_type    "pattern" | "sequence"
    unified.stop_rule         dict  (ATR stop multiplier)
    unified.exit_rule         dict  (TP + max_hold)
    unified.raw               underlying candidate (pattern or sequence)

Dispatch helpers
────────────────
    entry_mask_for(df, unified)   → boolean np.ndarray aligned to df rows
    backtest_for(df, unified)     → BacktestReport

Both helpers pick the right implementation based on `candidate_type`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from research.meta_strategy.strategy_selector import (
    allowed_regimes_for,
    allowed_regimes_for_sequence,
)
from research.pattern_strategies.backtest_pattern_strategy import (
    BacktestReport,
    _entry_mask,
    backtest_candidate,
)
from research.pattern_strategies.strategy_candidate import StrategyCandidate
from research.sequential_strategies.backtest_sequential_strategy import (
    _sequence_end_mask,
    backtest_sequential_candidate,
)
from research.sequential_strategies.strategy_candidate import (
    SequentialStrategyCandidate,
)

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


PATTERN = "pattern"
SEQUENCE = "sequence"


# ─────────────────────────────────────────────────────────────────────────────
# UnifiedCandidate
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class UnifiedCandidate:
    """
    Thin adapter wrapping either a `StrategyCandidate` (pattern) or
    a `SequentialStrategyCandidate` (sequence).  Exposes a single
    duck-typed interface to the meta-strategy selector and backtest.
    """

    name: str
    direction: str
    expected_edge: dict[str, Any]
    allowed_regimes: frozenset[str]
    candidate_type: str                       # PATTERN | SEQUENCE
    stop_rule: dict[str, Any]
    exit_rule: dict[str, Any]
    raw: StrategyCandidate | SequentialStrategyCandidate

    # NOTE: no `pattern_key` attribute — callers that need the raw
    # underlying structure should go through `.raw` or `.candidate_type`
    # and dispatch themselves.


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────


def _tag_name(name: str, kind: str) -> str:
    tag = "[P]" if kind == PATTERN else "[S]"
    # Avoid double-tagging if a caller passes a pre-tagged name.
    if name.startswith(tag):
        return name
    return f"{tag} {name}"


def wrap_pattern(candidate: StrategyCandidate) -> UnifiedCandidate:
    return UnifiedCandidate(
        name=_tag_name(candidate.name, PATTERN),
        direction=candidate.direction,
        expected_edge=dict(candidate.expected_edge or {}),
        allowed_regimes=frozenset(allowed_regimes_for(candidate)),
        candidate_type=PATTERN,
        stop_rule=candidate.stop_rule,
        exit_rule=candidate.exit_rule,
        raw=candidate,
    )


def wrap_sequential(candidate: SequentialStrategyCandidate) -> UnifiedCandidate:
    regimes = allowed_regimes_for_sequence(
        candidate.sequence_key, candidate.direction
    )
    return UnifiedCandidate(
        name=_tag_name(candidate.name, SEQUENCE),
        direction=candidate.direction,
        expected_edge=dict(candidate.expected_edge or {}),
        allowed_regimes=frozenset(regimes),
        candidate_type=SEQUENCE,
        stop_rule=candidate.stop_rule,
        exit_rule=candidate.exit_rule,
        raw=candidate,
    )


def wrap_many(
    pattern_cands: list[StrategyCandidate] | None = None,
    sequential_cands: list[SequentialStrategyCandidate] | None = None,
) -> list[UnifiedCandidate]:
    """Convenience: wrap both lists into a single unified list."""
    out: list[UnifiedCandidate] = []
    for c in pattern_cands or []:
        out.append(wrap_pattern(c))
    for c in sequential_cands or []:
        out.append(wrap_sequential(c))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch helpers
# ─────────────────────────────────────────────────────────────────────────────


def entry_mask_for(df: "pd.DataFrame", unified: UnifiedCandidate) -> "np.ndarray":
    """
    Return the boolean row-mask marking where this candidate's entry
    trigger fires.

    For PATTERN: row i's conditions all match → entry at bar i+1 open.
    For SEQUENCE: sequence completed at row j (end-mask) → entry at
                  bar j+1 open.

    Both masks share the same semantics: `mask[i] == True` means the
    candidate opens a position on the NEXT bar.  The meta backtest loop
    only needs this uniform signal.
    """
    if unified.candidate_type == PATTERN:
        return _entry_mask(df, unified.raw)  # type: ignore[arg-type]
    if unified.candidate_type == SEQUENCE:
        return _sequence_end_mask(df, unified.raw)  # type: ignore[arg-type]
    raise ValueError(f"Unknown candidate_type: {unified.candidate_type!r}")


def backtest_for(
    df: "pd.DataFrame", unified: UnifiedCandidate
) -> BacktestReport:
    """Run the full standalone backtest for a UnifiedCandidate."""
    if unified.candidate_type == PATTERN:
        rep = backtest_candidate(df, unified.raw)  # type: ignore[arg-type]
    elif unified.candidate_type == SEQUENCE:
        rep = backtest_sequential_candidate(df, unified.raw)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown candidate_type: {unified.candidate_type!r}")
    # Overwrite the name so the downstream per-strategy usage reporting
    # uses the tagged name.
    rep.candidate_name = unified.name
    return rep


__all__ = [
    "UnifiedCandidate",
    "PATTERN",
    "SEQUENCE",
    "wrap_pattern",
    "wrap_sequential",
    "wrap_many",
    "entry_mask_for",
    "backtest_for",
]
