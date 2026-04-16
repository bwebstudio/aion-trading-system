"""
research.sequential_discovery.sequence_generator
─────────────────────────────────────────────────
Level-wise (Apriori-style) sequence discovery for ordered events.

Levels
──────
  L=1 : base events (col, val) with support ≥ min_samples
        → NOT returned as sequences (length-1 is trivial)
        → used as building blocks for L=2 and beyond
  L=2 : all ordered pairs (A, B) with A != B
        evaluate → keep those with |mean| ≥ edge AND n ≥ min_samples
  L=3 : extend each L=2 survivor with every event C != last step
        evaluate → keep survivors
  L=4 : same expansion from L=3 survivors
  …up to `max_length`.

Extension uses `extend_end_mask`: a single left-shift + AND, no inner
Python loop per row.  Parent end-masks are cached per level and freed
before the next level starts.

A sequence is always ordered and canonical — (A, B) is a different
sequence from (B, A).  Identical consecutive steps are disallowed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from research.sequential_discovery.sequence_evaluator import (
    SequenceKey,
    SequenceResult,
    build_event_masks,
    evaluate_sequence,
    extend_end_mask,
)

if TYPE_CHECKING:
    import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _LevelSurvivor:
    """In-flight state held at each level during generation."""

    key: SequenceKey
    end_mask: np.ndarray  # valid-ret filtered


class SequenceGenerator:
    """
    Discover sequences of length 2..max_length.

    Parameters
    ----------
    min_samples:
        Minimum number of matching end-events required for a sequence
        to survive.
    minimal_edge_threshold:
        |mean_return| threshold.  Sequences below this are still evaluated
        but are not returned and are not expanded to the next level.
    max_length:
        Maximum sequence length (2..4 are practical).
    train_fraction:
        Split ratio used by `evaluate_sequence` to compute the train/test
        stability score.  Set to 0 to skip the split.
    """

    def __init__(
        self,
        *,
        min_samples: int = 100,
        minimal_edge_threshold: float = 0.0002,
        max_length: int = 3,
        train_fraction: float = 0.70,
    ) -> None:
        if max_length < 2:
            raise ValueError("max_length must be ≥ 2")
        self.min_samples = min_samples
        self.minimal_edge_threshold = minimal_edge_threshold
        self.max_length = max_length
        self.train_fraction = train_fraction

    # ── Public API ────────────────────────────────────────────────────────────

    def discover(
        self,
        df: "pd.DataFrame",
        bin_columns: tuple[str, ...],
    ) -> list[SequenceResult]:
        """
        Run level-wise sequence discovery over the compact matrix.

        Returns all sequences of length 2..max_length that pass the
        support + edge filters, sorted by |score| descending.
        """
        if "forward_return_10" not in df.columns:
            raise ValueError("df must contain 'forward_return_10' column")

        returns = df["forward_return_10"].to_numpy(
            dtype=np.float32, copy=False
        )
        valid_ret = np.isfinite(returns)

        # ── Level 1 — base events ────────────────────────────────────────────
        event_masks = build_event_masks(
            df, bin_columns, min_support=self.min_samples
        )
        if len(event_masks) < 2:
            return []

        # Precompute events as list for stable iteration + faster access.
        events: list[tuple[tuple[str, str], np.ndarray]] = list(
            event_masks.items()
        )

        all_results: list[SequenceResult] = []

        # ── Level 2 — ordered pairs ──────────────────────────────────────────
        level_survivors: list[_LevelSurvivor] = []
        for step_a, mask_a in events:
            for step_b, mask_b in events:
                if step_a == step_b:
                    continue  # disallow identical consecutive steps
                end_mask = extend_end_mask(mask_a, mask_b) & valid_ret
                n = int(end_mask.sum())
                if n < self.min_samples:
                    continue
                result = self._finalise(
                    (step_a, step_b), end_mask, returns
                )
                if result is None:
                    continue
                if abs(result.mean_return) < self.minimal_edge_threshold:
                    continue
                all_results.append(result)
                # Only survivors with edge are allowed to expand further.
                if self.max_length >= 3:
                    level_survivors.append(
                        _LevelSurvivor(key=result.sequence, end_mask=end_mask)
                    )

        # ── Levels 3 .. max_length ───────────────────────────────────────────
        for level in range(3, self.max_length + 1):
            if not level_survivors:
                break
            next_survivors: list[_LevelSurvivor] = []
            keep_masks = level < self.max_length
            for surv in level_survivors:
                last_step = surv.key[-1]
                for step_c, mask_c in events:
                    if step_c == last_step:
                        continue  # no identical consecutive step
                    new_end = extend_end_mask(surv.end_mask, mask_c)
                    # valid_ret already baked into surv.end_mask
                    n = int(new_end.sum())
                    if n < self.min_samples:
                        continue
                    new_key: SequenceKey = tuple(surv.key) + (step_c,)
                    result = self._finalise(new_key, new_end, returns)
                    if result is None:
                        continue
                    if abs(result.mean_return) < self.minimal_edge_threshold:
                        continue
                    all_results.append(result)
                    if keep_masks:
                        next_survivors.append(
                            _LevelSurvivor(key=new_key, end_mask=new_end)
                        )
            # Drop the previous level's cached masks before moving on.
            level_survivors = next_survivors

        all_results.sort(key=lambda r: abs(r.score), reverse=True)
        return all_results

    # ── Internal ─────────────────────────────────────────────────────────────

    def _finalise(
        self,
        key: SequenceKey,
        end_mask: np.ndarray,
        returns: np.ndarray,
    ) -> SequenceResult | None:
        """Compute aggregate metrics for a sequence given its end-mask."""
        n = int(end_mask.sum())
        if n < self.min_samples:
            return None

        rets = returns[end_mask]
        mean = float(rets.mean())
        winrate = float((rets > 0).mean())

        wins = rets[rets > 0]
        losses = rets[rets < 0]
        sum_w = float(wins.sum()) if len(wins) else 0.0
        sum_l = float(-losses.sum()) if len(losses) else 0.0
        pf = sum_w / sum_l if sum_l > 0 else None

        # Train/test stability.
        tm = te = None
        stab = None
        n_tr = n_te = 0
        if self.train_fraction > 0:
            n_total = len(returns)
            split_idx = int(n_total * self.train_fraction)
            train_mask = end_mask.copy()
            train_mask[split_idx:] = False
            test_mask = end_mask & ~train_mask
            n_tr = int(train_mask.sum())
            n_te = int(test_mask.sum())
            if n_tr > 0 and n_te > 0:
                tm = float(returns[train_mask].mean())
                te = float(returns[test_mask].mean())
                stab = te / tm if tm != 0.0 else None

        score = mean * math.sqrt(n)

        return SequenceResult(
            sequence=key,
            length=len(key),
            n_samples=n,
            mean_return=mean,
            expectancy=mean,
            profit_factor=pf,
            winrate=winrate,
            score=score,
            train_mean=tm,
            test_mean=te,
            train_n=n_tr,
            test_n=n_te,
            stability=stab,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────


def discover_sequences(
    df: "pd.DataFrame",
    bin_columns: tuple[str, ...],
    *,
    min_samples: int = 100,
    minimal_edge_threshold: float = 0.0002,
    max_length: int = 3,
    train_fraction: float = 0.70,
) -> list[SequenceResult]:
    """Shorthand for `SequenceGenerator(...).discover(df, bin_columns)`."""
    gen = SequenceGenerator(
        min_samples=min_samples,
        minimal_edge_threshold=minimal_edge_threshold,
        max_length=max_length,
        train_fraction=train_fraction,
    )
    return gen.discover(df, bin_columns)


__all__ = ["SequenceGenerator", "discover_sequences"]
