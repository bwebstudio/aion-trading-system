"""
research.sequential_discovery.sequence_evaluator
─────────────────────────────────────────────────
Vectorised scoring of a single sequence against the compact matrix.

Representation
──────────────
We encode each sequence by its "end mask": a boolean array where
`end_mask[j]` is True iff the sequence completed at row j (i.e. rows
`j-L+1 .. j` matched steps 0..L-1 in order).

Extension recurrence
────────────────────
Given a length-L end mask and a new final step C,

    new_end_mask[j] = end_mask[j - 1] AND C_mask[j]

so extending a sequence by one step is a single left-shift-by-one AND
a pointwise AND — O(N) per extension, no Python loops.

This lets the generator build level L+1 end masks directly from level L
survivors without re-scanning any row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


# Sequence = tuple of (column, bin_value) steps, ordered.
SequenceKey = tuple[tuple[str, str], ...]


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SequenceResult:
    """
    Metrics for a single sequence.

    * `sequence` is the canonical ordered tuple of steps.
    * Entry is applied after the LAST step; the forward return used is
      `forward_return_10[j]` where j is the row the sequence ended on.
    * `train_mean` / `test_mean` / `stability` populated only when
      `train_fraction` > 0; `stability = test_mean / train_mean`
      (None if train_mean is zero).
    """

    sequence: SequenceKey
    length: int
    n_samples: int
    mean_return: float
    expectancy: float
    profit_factor: float | None
    winrate: float
    score: float
    train_mean: float | None = None
    test_mean: float | None = None
    train_n: int = 0
    test_n: int = 0
    stability: float | None = None

    @property
    def description(self) -> str:
        return " -> ".join(f"{col}={val}" for col, val in self.sequence)

    def describe(self) -> str:
        stab = f"{self.stability:+.2f}" if self.stability is not None else " n/a"
        pf = (
            f"{self.profit_factor:.2f}"
            if self.profit_factor is not None
            else "  n/a"
        )
        return (
            f"L={self.length}  "
            f"n={self.n_samples:>5}  "
            f"mean={self.mean_return:+.5f}  "
            f"wr={self.winrate * 100:5.1f}%  "
            f"pf={pf:>5}  "
            f"stab={stab}  "
            f"score={self.score:+.4f}  "
            f"| {self.description}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Masks
# ─────────────────────────────────────────────────────────────────────────────


def build_event_masks(
    df: "pd.DataFrame",
    bin_columns: tuple[str, ...],
    *,
    min_support: int = 100,
) -> dict[tuple[str, str], np.ndarray]:
    """
    Precompute boolean row-masks for every (column, bin_value) event.

    Events with fewer than `min_support` matching rows are dropped.
    """
    out: dict[tuple[str, str], np.ndarray] = {}
    for col in bin_columns:
        if col not in df.columns or df[col].dtype.name != "category":
            continue
        cat = df[col].cat
        codes = cat.codes.to_numpy(copy=False)
        for i, val in enumerate(cat.categories):
            mask = codes == i
            n = int(mask.sum())
            if n < min_support:
                continue
            out[(col, str(val))] = mask
    return out


def _shift_right_by_one(mask: np.ndarray) -> np.ndarray:
    """Return `mask` shifted forward by one position (first entry False)."""
    out = np.zeros_like(mask)
    if len(mask) > 1:
        out[1:] = mask[:-1]
    return out


def extend_end_mask(
    parent_end_mask: np.ndarray,
    next_step_mask: np.ndarray,
) -> np.ndarray:
    """
    Given an end-mask of a (parent) length-L sequence and the per-row
    mask of the new step, return the end-mask of the length-(L+1) sequence.

        new[j] = parent[j-1] AND next[j]
    """
    return _shift_right_by_one(parent_end_mask) & next_step_mask


# ─────────────────────────────────────────────────────────────────────────────
# Single-sequence evaluation
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_sequence(
    sequence: SequenceKey,
    event_masks: dict[tuple[str, str], np.ndarray],
    returns: np.ndarray,
    valid_ret: np.ndarray,
    *,
    train_fraction: float = 0.70,
) -> SequenceResult | None:
    """
    Compute an end-mask for the sequence and return aggregate metrics.

    Returns None if any step's event-mask is missing (unknown value).
    """
    if not sequence:
        return None
    first = sequence[0]
    first_mask = event_masks.get(first)
    if first_mask is None:
        return None

    end_mask = first_mask.copy()
    for step in sequence[1:]:
        sm = event_masks.get(step)
        if sm is None:
            return None
        end_mask = extend_end_mask(end_mask, sm)

    end_mask &= valid_ret
    n = int(end_mask.sum())
    if n == 0:
        return None

    rets = returns[end_mask]
    mean = float(rets.mean())
    winrate = float((rets > 0).mean())

    wins = rets[rets > 0]
    losses = rets[rets < 0]
    sum_w = float(wins.sum()) if len(wins) else 0.0
    sum_l = float(-losses.sum()) if len(losses) else 0.0
    pf = sum_w / sum_l if sum_l > 0 else None

    # Train / test split on the row the sequence ENDED at.
    n_total = len(returns)
    split_idx = int(n_total * train_fraction)
    train_mask = end_mask.copy()
    train_mask[split_idx:] = False
    test_mask = end_mask & ~train_mask
    n_tr = int(train_mask.sum())
    n_te = int(test_mask.sum())
    if n_tr > 0 and n_te > 0:
        tm = float(returns[train_mask].mean())
        te = float(returns[test_mask].mean())
        stab = te / tm if tm != 0.0 else None
    else:
        tm = None
        te = None
        stab = None

    score = mean * math.sqrt(n)

    return SequenceResult(
        sequence=tuple(sequence),
        length=len(sequence),
        n_samples=n,
        mean_return=mean,
        expectancy=mean,  # per-trade expectancy = arithmetic mean
        profit_factor=pf,
        winrate=winrate,
        score=score,
        train_mean=tm,
        test_mean=te,
        train_n=n_tr,
        test_n=n_te,
        stability=stab,
    )


__all__ = [
    "SequenceKey",
    "SequenceResult",
    "build_event_masks",
    "extend_end_mask",
    "evaluate_sequence",
]
