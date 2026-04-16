"""
research.sequential_strategies
───────────────────────────────
Convert sequential (temporal) edges into executable
SequentialStrategyCandidate instances and backtest them.
Research-only — must not touch production aion.* trading modules.

Public API
──────────
  SequentialStrategyCandidate     — dataclass
  candidate_to_dict / candidate_from_dict
  convert_sequence_result         — SequenceResult -> candidate
  convert_sequence_dict           — dict (from sequential_edges.json) -> candidate
  backtest_sequential_candidate   — bar-replay backtest on a compact df
  BacktestReport                  — re-exported for uniformity with pattern side
"""

from research.pattern_strategies.backtest_pattern_strategy import BacktestReport
from research.sequential_strategies.backtest_sequential_strategy import (
    backtest_sequential_candidate,
)
from research.sequential_strategies.sequence_to_strategy import (
    convert_sequence_dict,
    convert_sequence_result,
)
from research.sequential_strategies.strategy_candidate import (
    SequentialStrategyCandidate,
    candidate_from_dict,
    candidate_to_dict,
)

__all__ = [
    "SequentialStrategyCandidate",
    "candidate_to_dict",
    "candidate_from_dict",
    "convert_sequence_result",
    "convert_sequence_dict",
    "backtest_sequential_candidate",
    "BacktestReport",
]
