"""
research.pattern_discovery
───────────────────────────
Research-only pattern discovery engine.  Not part of production AION.

Public API:
    FeatureBuilder     — build a feature matrix from snapshots
    PatternGenerator   — enumerate candidate feature-threshold patterns
    ForwardTester      — evaluate forward return metrics (spread/slippage aware)
    Pattern            — dataclass describing a single pattern
    PatternResult      — dataclass with mean_return, win_rate, sample_size, sharpe
"""

from research.pattern_discovery.feature_builder import FeatureBuilder
from research.pattern_discovery.forward_tester import ForwardTester
from research.pattern_discovery.multi_asset_validator import (
    MultiAssetPatternResult,
    PerAssetStats,
    validate_across_assets,
)
from research.pattern_discovery.pattern_generator import PatternGenerator
from research.pattern_discovery.pattern_types import (
    CompactPatternResult,
    Condition,
    Pattern,
    PatternKey,
    PatternResult,
    format_pattern_key,
)

__all__ = [
    "FeatureBuilder",
    "PatternGenerator",
    "ForwardTester",
    "Pattern",
    "PatternResult",
    "Condition",
    "PatternKey",
    "CompactPatternResult",
    "format_pattern_key",
    "MultiAssetPatternResult",
    "PerAssetStats",
    "validate_across_assets",
]
