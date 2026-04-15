"""
research.pattern_strategies
────────────────────────────
Convert discovered PatternKeys into executable strategy candidates and
backtest them.  Research-only — must not import or mutate production
aion.* trading modules.

Public API
──────────
  StrategyCandidate        — dataclass describing one executable candidate
  candidate_to_dict / candidate_from_dict — JSON serde helpers
  convert_compact_result   — CompactPatternResult  -> StrategyCandidate
  convert_multi_asset      — MultiAssetPatternResult -> StrategyCandidate
  backtest_candidate       — bar-replay backtest of one candidate on a df
  BacktestReport           — dataclass with trade list + aggregate metrics
"""

from research.pattern_strategies.backtest_pattern_strategy import (
    BacktestReport,
    Trade,
    backtest_candidate,
)
from research.pattern_strategies.pattern_to_strategy import (
    convert_compact_result,
    convert_multi_asset,
)
from research.pattern_strategies.strategy_candidate import (
    StrategyCandidate,
    candidate_from_dict,
    candidate_to_dict,
)

__all__ = [
    "StrategyCandidate",
    "candidate_to_dict",
    "candidate_from_dict",
    "convert_compact_result",
    "convert_multi_asset",
    "backtest_candidate",
    "BacktestReport",
    "Trade",
]
