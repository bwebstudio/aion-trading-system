"""
research.pattern_strategies.strategy_candidate
───────────────────────────────────────────────
Dataclass + JSON serde for an executable strategy candidate derived
from a discovered PatternKey.

Design
──────
* Purely declarative — no behaviour, no live state.
* Entry / stop / exit rules are nested dicts (not code) so the JSON
  representation is loss-less and can be re-instantiated anywhere.
* `expected_edge` carries the discovery stats that justify the candidate
  (sample size, train/test means, stability, assets found).  The
  backtester does not read it; it is for downstream filtering and
  human review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research.pattern_discovery.pattern_types import PatternKey, format_pattern_key


Direction = str  # "LONG" | "SHORT"


@dataclass(frozen=True)
class StrategyCandidate:
    """
    A single executable candidate derived from a PatternKey.

    Fields
    ──────
    pattern_key:
        Canonical sorted tuple of (bin_column, bin_value) pairs.
    direction:
        "LONG" if discovery's test_mean_return > 0 else "SHORT".
    entry_rule:
        {"type": "AND", "conditions": [
            {"column": "momentum_3_bin", "equals": "NEG"}, ...
        ]}
    stop_rule:
        {"type": "atr_multiplier", "period": 14, "multiplier": 1.5}
    exit_rule:
        {"take_profit": {"type": "atr_multiplier", "period": 14,
                         "multiplier": 2.5},
         "max_hold_bars": 20}
    expected_edge:
        {"sample_size": ..., "mean_test_return": ..., "train_win_rate": ...,
         "test_win_rate": ..., "stability_score": ..., "assets_found": [...]}
    name:
        Stable human-readable id (auto-derived from direction + pattern).
    """

    pattern_key: PatternKey
    direction: Direction
    entry_rule: dict[str, Any]
    stop_rule: dict[str, Any]
    exit_rule: dict[str, Any]
    expected_edge: dict[str, Any] = field(default_factory=dict)
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            auto = f"{self.direction}:{format_pattern_key(self.pattern_key)}"
            object.__setattr__(self, "name", auto)

    @property
    def description(self) -> str:
        return format_pattern_key(self.pattern_key)


# ─────────────────────────────────────────────────────────────────────────────
# JSON serde
# ─────────────────────────────────────────────────────────────────────────────


def candidate_to_dict(c: StrategyCandidate) -> dict[str, Any]:
    """Serialise a StrategyCandidate into a JSON-ready dict."""
    return {
        "name": c.name,
        "direction": c.direction,
        "pattern_key": [list(pair) for pair in c.pattern_key],
        "entry_rule": c.entry_rule,
        "stop_rule": c.stop_rule,
        "exit_rule": c.exit_rule,
        "expected_edge": c.expected_edge,
    }


def candidate_from_dict(d: dict[str, Any]) -> StrategyCandidate:
    """Rebuild a StrategyCandidate from its JSON dict form."""
    key: PatternKey = tuple(tuple(pair) for pair in d["pattern_key"])
    return StrategyCandidate(
        pattern_key=key,
        direction=d["direction"],
        entry_rule=d["entry_rule"],
        stop_rule=d["stop_rule"],
        exit_rule=d["exit_rule"],
        expected_edge=d.get("expected_edge", {}),
        name=d.get("name", ""),
    )
