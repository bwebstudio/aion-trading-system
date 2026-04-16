"""
research.sequential_strategies.strategy_candidate
──────────────────────────────────────────────────
`SequentialStrategyCandidate` — the sequence-based equivalent of
`research.pattern_strategies.StrategyCandidate`.

Differences vs. the pattern-based candidate
───────────────────────────────────────────
* `sequence_key` is ORDERED (tuple of (column, bin_value) applied to
  consecutive bars) — not a sorted set of conditions.
* `entry_rule.type` is "SEQUENCE" instead of "AND".
* Entry fires on the bar AFTER the sequence completes, i.e. at row
  (i + L) open, where i is the row the first step matched and L is
  the sequence length.

Everything else (stop_rule, exit_rule, expected_edge, JSON shape) is
intentionally symmetric with StrategyCandidate so downstream tooling
stays uniform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SequenceKey = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SequentialStrategyCandidate:
    """Executable candidate derived from a sequential (temporal) edge."""

    sequence_key: SequenceKey
    direction: str                          # "LONG" | "SHORT"
    entry_rule: dict[str, Any]
    stop_rule: dict[str, Any]
    exit_rule: dict[str, Any]
    expected_edge: dict[str, Any] = field(default_factory=dict)
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            desc = " -> ".join(f"{c}={v}" for c, v in self.sequence_key)
            object.__setattr__(self, "name", f"{self.direction}:{desc}")

    @property
    def length(self) -> int:
        return len(self.sequence_key)

    @property
    def description(self) -> str:
        return " -> ".join(f"{c}={v}" for c, v in self.sequence_key)


# ─────────────────────────────────────────────────────────────────────────────
# JSON serde
# ─────────────────────────────────────────────────────────────────────────────


def candidate_to_dict(c: SequentialStrategyCandidate) -> dict[str, Any]:
    return {
        "name": c.name,
        "direction": c.direction,
        "sequence_key": [list(step) for step in c.sequence_key],
        "length": c.length,
        "entry_rule": c.entry_rule,
        "stop_rule": c.stop_rule,
        "exit_rule": c.exit_rule,
        "expected_edge": c.expected_edge,
    }


def candidate_from_dict(d: dict[str, Any]) -> SequentialStrategyCandidate:
    key: SequenceKey = tuple(tuple(step) for step in d["sequence_key"])
    return SequentialStrategyCandidate(
        sequence_key=key,
        direction=d["direction"],
        entry_rule=d["entry_rule"],
        stop_rule=d["stop_rule"],
        exit_rule=d["exit_rule"],
        expected_edge=d.get("expected_edge", {}),
        name=d.get("name", ""),
    )


__all__ = [
    "SequentialStrategyCandidate",
    "SequenceKey",
    "candidate_to_dict",
    "candidate_from_dict",
]
