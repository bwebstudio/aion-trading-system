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
    HEALTHY_DECAY_STATUSES,
    MetaBacktestReport,
    UNHEALTHY_DECAY_STATUSES,
    backtest_meta,
    decay_filter_candidates,
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
    allowed_regimes_for_sequence,
)
from research.meta_strategy.unified import (
    PATTERN,
    SEQUENCE,
    UnifiedCandidate,
    backtest_for,
    entry_mask_for,
    wrap_many,
    wrap_pattern,
    wrap_sequential,
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
    "allowed_regimes_for_sequence",
    "EXTREME_VWAP_BINS",
    "NO_TRADE",
    "UnifiedCandidate",
    "PATTERN",
    "SEQUENCE",
    "wrap_pattern",
    "wrap_sequential",
    "wrap_many",
    "entry_mask_for",
    "backtest_for",
    "backtest_meta",
    "MetaBacktestReport",
    "CandidateFilter",
    "prefilter_candidates",
    "decay_filter_candidates",
    "HEALTHY_DECAY_STATUSES",
    "UNHEALTHY_DECAY_STATUSES",
]
