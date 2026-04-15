"""
research.pattern_discovery.pattern_generator
─────────────────────────────────────────────
Enumerate candidate feature-threshold patterns.

Thresholds (sigma-bins)
───────────────────
Continuous features are thresholded on their z-score columns (`{name}_z`),
where z_i = value_i / rolling_std_i (trailing window, no lookahead).
Fixed sigma-bins are used — no data-fitted thresholds:

    < -2sigma    < -1.5sigma    < -1sigma    > +1sigma    > +1.5sigma    > +2sigma

Boolean features → {True, False}.
Categorical features (session) → each distinct value that occurs often enough.

Combinations
────────────
Single-feature, pair, and triple patterns are enumerated.  Same feature
never appears twice in the same pattern.  Total output is capped at
`max_patterns` (default 3000) with a stable deterministic ordering:
singles first, then pairs, then triples.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Iterator

from research.pattern_discovery.pattern_types import Condition, Pattern, PatternKey

if TYPE_CHECKING:
    import pandas as pd


# Continuous features are compared through their z-score column.
# The condition is stored on the `_z` column (e.g. `distance_to_vwap_z`).
CONTINUOUS_FEATURES: tuple[str, ...] = (
    "distance_to_vwap",
    "distance_to_session_high",
    "distance_to_session_low",
    "momentum_3",
    "momentum_5",
)
BOOLEAN_FEATURES: tuple[str, ...] = ("range_compression",)
CATEGORICAL_FEATURES: tuple[str, ...] = ("session",)

# Fixed sigma-bins applied to `_z` columns.
SIGMA_BINS: tuple[tuple[str, float, str], ...] = (
    ("<", -2.0, "< -2sigma"),
    ("<", -1.5, "< -1.5sigma"),
    ("<", -1.0, "< -1sigma"),
    (">", 1.0, "> +1sigma"),
    (">", 1.5, "> +1.5sigma"),
    (">", 2.0, "> +2sigma"),
)


class PatternGenerator:
    """
    Generate candidate Patterns from a feature matrix.

    Parameters
    ----------
    max_patterns:
        Hard cap on the number of patterns returned.  Default 3000.
    min_bucket_occurrences:
        Drop a candidate condition if fewer than this many rows in the
        feature matrix satisfy it.  Default 20.
    """

    def __init__(
        self,
        max_patterns: int = 3000,
        min_bucket_occurrences: int = 20,
        *,
        max_conditions: int = 3,
        minimal_edge_threshold: float = 0.0002,
    ) -> None:
        self.max_patterns = max_patterns
        self.min_bucket_occurrences = min_bucket_occurrences
        self.max_conditions = max_conditions
        self.minimal_edge_threshold = minimal_edge_threshold

    # ─── Public API ───────────────────────────────────────────────────────────

    def generate(self, feature_matrix: list[dict[str, Any]]) -> list[Pattern]:
        """Return a list of candidate Patterns (order ≤ 3)."""
        conditions_by_feature = self._build_condition_pool(feature_matrix)
        features = [f for f, conds in conditions_by_feature.items() if conds]

        patterns: list[Pattern] = []

        # Single-feature patterns
        for feat in features:
            for cond in conditions_by_feature[feat]:
                patterns.append(
                    Pattern(conditions=(cond,), feature_names=(feat,))
                )

        # Pair patterns
        for f1, f2 in itertools.combinations(features, 2):
            for c1 in conditions_by_feature[f1]:
                for c2 in conditions_by_feature[f2]:
                    patterns.append(
                        Pattern(
                            conditions=(c1, c2),
                            feature_names=(f1, f2),
                        )
                    )

        # Triple patterns
        for f1, f2, f3 in itertools.combinations(features, 3):
            for c1 in conditions_by_feature[f1]:
                for c2 in conditions_by_feature[f2]:
                    for c3 in conditions_by_feature[f3]:
                        patterns.append(
                            Pattern(
                                conditions=(c1, c2, c3),
                                feature_names=(f1, f2, f3),
                            )
                        )

        if len(patterns) > self.max_patterns:
            patterns = patterns[: self.max_patterns]
        return patterns

    # ─── Condition pool construction ──────────────────────────────────────────

    def _build_condition_pool(
        self, rows: list[dict[str, Any]]
    ) -> dict[str, list[Condition]]:
        pool: dict[str, list[Condition]] = {}

        # Continuous features → sigma-bin conditions on the `_z` column.
        for feat in CONTINUOUS_FEATURES:
            z_col = f"{feat}_z"
            observed = sum(
                1 for r in rows if isinstance(r.get(z_col), (int, float))
            )
            if observed < self.min_bucket_occurrences:
                continue
            # Conditions read from `_z` column; meta label uses the
            # original feature name for readability.
            conds = [
                Condition(
                    feature=z_col,
                    op=op,
                    value=value,
                    meta=f"{feat} {label}",
                )
                for op, value, label in SIGMA_BINS
            ]
            pool[feat] = self._filter_by_occurrence(conds, rows)

        # Boolean features → True / False if both observed enough.
        for feat in BOOLEAN_FEATURES:
            vals = [r[feat] for r in rows if isinstance(r.get(feat), bool)]
            if len(vals) < self.min_bucket_occurrences:
                continue
            candidate_conds = [
                Condition(feature=feat, op="==", value=True),
                Condition(feature=feat, op="==", value=False),
            ]
            pool[feat] = self._filter_by_occurrence(candidate_conds, rows)

        # Categorical features → each distinct value that appears often enough.
        for feat in CATEGORICAL_FEATURES:
            counts: dict[Any, int] = {}
            for r in rows:
                v = r.get(feat)
                if v is None:
                    continue
                counts[v] = counts.get(v, 0) + 1
            conds = [
                Condition(feature=feat, op="==", value=v)
                for v, c in counts.items()
                if c >= self.min_bucket_occurrences
            ]
            if conds:
                pool[feat] = conds

        return pool

    def _filter_by_occurrence(
        self,
        conditions: list[Condition],
        rows: list[dict[str, Any]],
    ) -> list[Condition]:
        kept: list[Condition] = []
        for cond in conditions:
            hits = sum(1 for r in rows if cond.evaluate(r))
            if hits >= self.min_bucket_occurrences:
                kept.append(cond)
        return kept

    # ─────────────────────────────────────────────────────────────────────────
    # Fast path (v3) — operates on the compact DataFrame
    # ─────────────────────────────────────────────────────────────────────────

    def stream_keys(
        self,
        df: "pd.DataFrame",
        *,
        bin_columns: tuple[str, ...],
        max_patterns: int = 50_000,
        min_support: int = 100,
        max_order: int = 3,
    ) -> Iterator[PatternKey]:
        """
        Yield candidate PatternKeys on-the-fly, applying support pruning.

        Parameters
        ----------
        df:
            Compact feature DataFrame from FeatureBuilder.build_compact_matrix.
        bin_columns:
            The categorical columns in `df` that patterns may condition on.
        max_patterns:
            Hard cap on total keys yielded (across all orders).  Default 50_000.
        min_support:
            Minimum row count a (column, value) condition must satisfy to
            be used.  Pair/triple keys are pruned via an independence
            upper-bound before evaluation.  Default 100.
        max_order:
            Maximum number of conditions per pattern (1, 2, or 3).
            Default 3.

        Notes
        -----
        * No duplicates: each bin_column appears at most once per pattern.
        * Keys are emitted with their (column, value) pairs sorted
          lexicographically, making equivalent patterns identical.
        * The generator is lazy — callers can stop consuming at any point.
        """
        singletons = self._singleton_supports(df, bin_columns, min_support)
        total_rows = len(df)
        if total_rows == 0 or not singletons:
            return

        # Flatten (col, val, support) into a stable, sorted list of singletons.
        items: list[tuple[str, str, int]] = []
        for col in sorted(singletons):
            for val, sup in sorted(singletons[col].items()):
                items.append((col, val, sup))

        emitted = 0

        # ── Singles ───────────────────────────────────────────────────────────
        for col, val, _ in items:
            if emitted >= max_patterns:
                return
            yield ((col, val),)
            emitted += 1

        if max_order < 2:
            return

        # ── Pairs ────────────────────────────────────────────────────────────
        # Independence-based upper bound prune: if p(A) * p(B) * N < min_support
        # AND observed supports are low, skip.  For disjoint columns the
        # indep estimate is usable; we still bail out on observed joint
        # below threshold (but that costs a mask — we skip it here and let
        # the evaluator discard low-support matches via min_samples).
        by_col: dict[str, list[tuple[str, int]]] = {}
        for col, val, sup in items:
            by_col.setdefault(col, []).append((val, sup))

        cols = sorted(by_col.keys())
        for ca, cb in itertools.combinations(cols, 2):
            for va, supa in by_col[ca]:
                for vb, supb in by_col[cb]:
                    if emitted >= max_patterns:
                        return
                    # Independence upper bound — conservative prune.
                    exp = (supa / total_rows) * (supb / total_rows) * total_rows
                    if exp < min_support * 0.25:
                        # Very unlikely to hit min_support even with
                        # positive correlation.  Skip.
                        continue
                    key = self._normalise_key(((ca, va), (cb, vb)))
                    yield key
                    emitted += 1

        if max_order < 3:
            return

        # ── Triples ──────────────────────────────────────────────────────────
        for ca, cb, cc in itertools.combinations(cols, 3):
            for va, supa in by_col[ca]:
                for vb, supb in by_col[cb]:
                    for vc, supc in by_col[cc]:
                        if emitted >= max_patterns:
                            return
                        exp = (
                            (supa / total_rows)
                            * (supb / total_rows)
                            * (supc / total_rows)
                            * total_rows
                        )
                        if exp < min_support * 0.25:
                            continue
                        key = self._normalise_key(
                            ((ca, va), (cb, vb), (cc, vc))
                        )
                        yield key
                        emitted += 1

    # ─── Fast helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_key(pairs: tuple[tuple[str, str], ...]) -> PatternKey:
        return tuple(sorted(pairs))

    @staticmethod
    def _singleton_supports(
        df: "pd.DataFrame",
        bin_columns: tuple[str, ...],
        min_support: int,
    ) -> dict[str, dict[str, int]]:
        """
        Count rows per (column, bin_value) and drop values below min_support.

        Uses pandas value_counts — a single O(N) pass per column.
        """
        out: dict[str, dict[str, int]] = {}
        for col in bin_columns:
            if col not in df.columns:
                continue
            vc = df[col].value_counts(dropna=True)
            kept = {
                str(val): int(cnt)
                for val, cnt in vc.items()
                if int(cnt) >= min_support
            }
            if kept:
                out[col] = kept
        return out

    # ═════════════════════════════════════════════════════════════════════════
    # Level-wise (Apriori-style) generator — v5
    # ═════════════════════════════════════════════════════════════════════════

    def generate_patterns_levelwise(
        self,
        df: "pd.DataFrame",
        *,
        bin_columns: tuple[str, ...],
        batch_size: int = 1000,
        min_samples: int | None = None,
        minimal_edge_threshold: float | None = None,
        max_conditions: int | None = None,
    ) -> Iterator[list[PatternKey]]:
        """
        Apriori-style level-wise pattern expansion.

        At each level L (1..max_conditions):
            1. Build every L-condition pattern by adding one (column, bin_value)
               to a surviving (L-1)-pattern.
            2. Compute the sample size and mean forward return (cheap
               vectorised mask + sum on precomputed `forward_return_10`).
            3. A candidate SURVIVES the level iff:
                    sample_size >= min_samples
                    |mean_return| >= minimal_edge_threshold
            4. Only survivors are yielded AND only survivors feed expansion
               at level L+1.  This bounds the combinatorial explosion:
               unpromising parents are never expanded.

        The yielded batches are `list[PatternKey]` of up to `batch_size`
        items.  Memory is bounded by the current-level parent-mask buffer
        (dropped before moving to the next level).  The evaluator
        (ForwardTester.evaluate_patterns) consumes any iterable of
        PatternKeys — it does not need to change.

        Parameters
        ----------
        df:
            Compact feature DataFrame from FeatureBuilder.build_compact_matrix.
            Must contain `forward_return_10` and the categorical bin columns.
        bin_columns:
            Categorical columns that patterns may condition on.
        batch_size:
            Number of PatternKeys per yielded list.  Default 1000.
        min_samples:
            Per-pattern support threshold.  Defaults to `min_bucket_occurrences`.
        minimal_edge_threshold:
            Magnitude of mean return below which a pattern is pruned.
            Defaults to the class attribute (0.0002).
        max_conditions:
            Maximum pattern order (1, 2, or 3).  Defaults to the class attribute.

        Notes
        -----
        The cheap per-pattern mean here is intentionally *unfiltered*
        (no train/test split).  Its only role is to gate expansion.
        The downstream evaluator applies the full train/test stability
        filter.  This duplicated-work cost is small — the mask build is
        the dominant op and runs once per candidate per level.
        """
        import numpy as np

        if min_samples is None:
            min_samples = self.min_bucket_occurrences
        if minimal_edge_threshold is None:
            minimal_edge_threshold = self.minimal_edge_threshold
        if max_conditions is None:
            max_conditions = self.max_conditions
        if max_conditions < 1:
            return

        n = len(df)
        if n == 0:
            return

        returns = df["forward_return_10"].to_numpy(dtype=np.float32, copy=False)
        valid_ret = np.isfinite(returns)

        # Precompute int8 codes per column + value → code lookup.
        codes: dict[str, "np.ndarray"] = {}
        lookup: dict[str, dict[str, int]] = {}
        for col in bin_columns:
            if col not in df.columns or df[col].dtype.name != "category":
                continue
            cat = df[col].cat
            codes[col] = cat.codes.to_numpy(copy=False)
            lookup[col] = {str(v): i for i, v in enumerate(cat.categories)}

        cols_sorted = sorted(codes)
        if not cols_sorted:
            return

        # ── Level 1 ───────────────────────────────────────────────────────────
        batch: list[PatternKey] = []
        level_survivors: list[tuple[PatternKey, "np.ndarray"]] = []

        for col in cols_sorted:
            col_codes = codes[col]
            for val, code in lookup[col].items():
                mask = valid_ret & (col_codes == code)
                n_match = int(mask.sum())
                if n_match < min_samples:
                    continue
                mean = float(returns[mask].mean())
                if abs(mean) < minimal_edge_threshold:
                    continue
                key: PatternKey = ((col, val),)
                # Keep mask only if we will expand this pattern.
                if max_conditions >= 2:
                    level_survivors.append((key, mask))
                batch.append(key)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch
            batch = []

        # ── Higher levels ─────────────────────────────────────────────────────
        for level in range(2, max_conditions + 1):
            if not level_survivors:
                return
            next_survivors: list[tuple[PatternKey, "np.ndarray"]] = []
            seen: set[PatternKey] = set()
            keep_masks = level < max_conditions  # last level: drop masks

            for parent_key, parent_mask in level_survivors:
                used_cols = {c for c, _ in parent_key}
                for col in cols_sorted:
                    if col in used_cols:
                        continue
                    col_codes = codes[col]
                    for val, code in lookup[col].items():
                        new_key = tuple(
                            sorted(parent_key + ((col, val),))
                        )
                        if new_key in seen:
                            continue
                        seen.add(new_key)

                        mask = parent_mask & (col_codes == code)
                        n_match = int(mask.sum())
                        if n_match < min_samples:
                            continue
                        mean = float(returns[mask].mean())
                        if abs(mean) < minimal_edge_threshold:
                            continue

                        if keep_masks:
                            next_survivors.append((new_key, mask))
                        batch.append(new_key)
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
            if batch:
                yield batch
                batch = []

            # Drop previous level's masks before advancing.
            level_survivors = next_survivors
