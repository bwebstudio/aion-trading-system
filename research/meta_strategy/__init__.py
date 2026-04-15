"""
research.meta_strategy
───────────────────────
Research-only meta-strategy layer: rule-based regime classifier +
regime-aware strategy selector + meta backtest.

Public API
──────────
  classify_rows          — vectorised regime labelling
  classify_row           — single-row helper
  StrategySelector       — ranks candidates within each regime
  allowed_regimes_for    — derive allowed regimes from a candidate
  backtest_meta          — end-to-end meta backtest
  MetaBacktestReport     — composite report
"""

from research.meta_strategy.meta_backtest import (
    CandidateFilter,
    MetaBacktestReport,
    backtest_meta,
    prefilter_candidates,
)
from research.meta_strategy.regime_classifier import (
    ALL_REGIMES,
    COMPRESSION,
    RANGE,
    TREND_DOWN,
    TREND_UP,
    classify_row,
    classify_rows,
)
from research.meta_strategy.strategy_selector import (
    EXTREME_VWAP_BINS,
    NO_TRADE,
    StrategySelector,
    allowed_regimes_for,
)

__all__ = [
    "classify_rows",
    "classify_row",
    "ALL_REGIMES",
    "TREND_UP",
    "TREND_DOWN",
    "RANGE",
    "COMPRESSION",
    "StrategySelector",
    "allowed_regimes_for",
    "EXTREME_VWAP_BINS",
    "NO_TRADE",
    "backtest_meta",
    "MetaBacktestReport",
    "CandidateFilter",
    "prefilter_candidates",
]
