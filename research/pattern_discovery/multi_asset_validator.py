"""
research.pattern_discovery.multi_asset_validator
─────────────────────────────────────────────────
Cross-asset validation of discovered patterns.

Given per-asset discovery results, find PatternKeys that appear in two or
more assets and summarise their aggregate edge.  The premise: structural
market edges generalise across symbols; single-asset top patterns are more
likely to be noise.

Input
─────
asset_results : dict[str, list[CompactPatternResult]]
    Keyed by asset name (canonical, e.g. "US100.cash").  Each value is a
    list of per-asset top-k CompactPatternResult items (as produced by
    ForwardTester.evaluate_patterns).

Output
──────
list[MultiAssetPatternResult] — only PatternKeys observed in
`min_assets` assets (default 2), sorted by mean_score descending.

Ranking
───────
mean_score_across_assets = mean(test_mean_return for asset in assets) \
                           * sqrt(total_samples)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from research.pattern_discovery.pattern_types import (
    CompactPatternResult,
    PatternKey,
    format_pattern_key,
)


@dataclass(frozen=True)
class PerAssetStats:
    """Per-asset snapshot of a pattern's performance."""

    asset: str
    sample_size: int
    train_mean_return: float
    test_mean_return: float
    train_win_rate: float
    test_win_rate: float
    stability_score: float | None
    score: float | None


@dataclass(frozen=True)
class MultiAssetPatternResult:
    """
    Aggregate result for a PatternKey observed in ≥ min_assets.

    Fields
    ──────
    pattern_key              : canonical tuple[(col, bin_value), ...]
    assets_found             : tuple of asset names containing the pattern
    per_asset_stats          : tuple of PerAssetStats (same order as assets_found)
    mean_test_return         : mean of per-asset test_mean_return values
    total_samples            : sum of per-asset sample_size values
    mean_score_across_assets : mean_test_return * sqrt(total_samples)
    sign_agreement           : True iff every per-asset test_mean_return
                               has the same sign (strongest structural signal)
    """

    pattern_key: PatternKey
    assets_found: tuple[str, ...]
    per_asset_stats: tuple[PerAssetStats, ...]
    mean_test_return: float
    total_samples: int
    mean_score_across_assets: float
    sign_agreement: bool

    @property
    def n_assets(self) -> int:
        return len(self.assets_found)

    @property
    def description(self) -> str:
        return format_pattern_key(self.pattern_key)

    def describe(self) -> str:
        stars = "*" if self.sign_agreement else " "
        per = ", ".join(
            f"{s.asset}:{s.test_mean_return:+.5f}(n={s.sample_size})"
            for s in self.per_asset_stats
        )
        return (
            f" {stars}  "
            f"assets={self.n_assets}  "
            f"mean_test={self.mean_test_return:+.5f}  "
            f"n_total={self.total_samples:>6}  "
            f"score={self.mean_score_across_assets:+.4f}  "
            f"| {self.description}  [{per}]"
        )


def validate_across_assets(
    asset_results: Mapping[str, list[CompactPatternResult]],
    *,
    min_assets: int = 2,
    require_sign_agreement: bool = False,
) -> list[MultiAssetPatternResult]:
    """
    Collapse per-asset results into cross-asset aggregates.

    Parameters
    ----------
    asset_results:
        Mapping from asset name → per-asset top-k results.
    min_assets:
        Minimum number of assets a PatternKey must appear in to be kept.
        Default 2.
    require_sign_agreement:
        If True, further filter to patterns whose per-asset test means
        all share the same sign.  Default False — sign agreement is
        reported but not required.

    Returns
    -------
    list[MultiAssetPatternResult]
        Sorted descending by |mean_score_across_assets|.
    """
    # key → list[(asset, CompactPatternResult)]
    occurrences: dict[PatternKey, list[tuple[str, CompactPatternResult]]] = {}
    for asset, results in asset_results.items():
        seen_this_asset: set[PatternKey] = set()
        for r in results:
            # Guard against duplicate keys per asset (same pattern listed twice).
            if r.key in seen_this_asset:
                continue
            seen_this_asset.add(r.key)
            occurrences.setdefault(r.key, []).append((asset, r))

    aggregated: list[MultiAssetPatternResult] = []
    for key, entries in occurrences.items():
        if len(entries) < min_assets:
            continue

        # Preserve insertion order of assets (dict iteration order = call order).
        stats = tuple(
            PerAssetStats(
                asset=asset,
                sample_size=r.sample_size,
                train_mean_return=r.train_mean_return,
                test_mean_return=r.test_mean_return,
                train_win_rate=r.train_win_rate,
                test_win_rate=r.test_win_rate,
                stability_score=r.stability_score,
                score=r.score,
            )
            for asset, r in entries
        )

        test_returns = [s.test_mean_return for s in stats]
        total_n = sum(s.sample_size for s in stats)
        mean_test = sum(test_returns) / len(test_returns)
        mean_score = mean_test * math.sqrt(total_n)

        signs = {1 if t > 0 else -1 if t < 0 else 0 for t in test_returns}
        sign_agreement = len(signs) == 1 and 0 not in signs

        if require_sign_agreement and not sign_agreement:
            continue

        aggregated.append(
            MultiAssetPatternResult(
                pattern_key=key,
                assets_found=tuple(s.asset for s in stats),
                per_asset_stats=stats,
                mean_test_return=mean_test,
                total_samples=total_n,
                mean_score_across_assets=mean_score,
                sign_agreement=sign_agreement,
            )
        )

    aggregated.sort(
        key=lambda r: abs(r.mean_score_across_assets),
        reverse=True,
    )
    return aggregated


__all__ = [
    "MultiAssetPatternResult",
    "PerAssetStats",
    "validate_across_assets",
]
