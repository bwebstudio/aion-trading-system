"""
research.pattern_discovery.pattern_types
─────────────────────────────────────────
Core dataclasses + compact types for the pattern discovery engine.

v3 scaling notes
────────────────
For fast, memory-bounded discovery over tens of thousands of candidate
patterns, the canonical representation is the **PatternKey**:

    PatternKey = tuple[tuple[str, str], ...]

Example:
    (("distance_to_vwap_bin", "LT_NEG_2SIG"),
     ("momentum_3_bin",       "POS"))

PatternKey is sortable, hashable, and trivially comparable — the generator
and evaluator pass PatternKey values around in streams rather than
materialising full Pattern objects.

The heavier `Pattern` / `Condition` classes remain for human-readable
formatting and for the v1/v2 APIs.  They are not used on the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Compact pattern identity: sorted tuple of (column, bin_value) pairs.
PatternKey = tuple[tuple[str, str], ...]


def format_pattern_key(key: PatternKey) -> str:
    """Human-readable string for a PatternKey."""
    return " AND ".join(f"{col}={val}" for col, val in key)

Operator = str  # one of: ">", "<", ">=", "<=", "==", "!="


@dataclass(frozen=True)
class Condition:
    """A single feature threshold condition."""

    feature: str
    op: Operator
    value: Any
    meta: str = ""

    def describe(self) -> str:
        if self.meta:
            return self.meta
        if isinstance(self.value, float):
            return f"{self.feature} {self.op} {self.value:.4f}"
        return f"{self.feature} {self.op} {self.value}"

    def evaluate(self, row: dict[str, Any]) -> bool:
        v = row.get(self.feature)
        if v is None:
            return False
        try:
            if self.op == ">":
                return v > self.value
            if self.op == "<":
                return v < self.value
            if self.op == ">=":
                return v >= self.value
            if self.op == "<=":
                return v <= self.value
            if self.op == "==":
                return v == self.value
            if self.op == "!=":
                return v != self.value
        except TypeError:
            return False
        return False


@dataclass(frozen=True)
class Pattern:
    """
    A conjunction (AND) of conditions on one or more features.

    conditions:
        Tuple of Condition instances — every condition must evaluate True
        for the pattern to match the feature row.
    feature_names:
        Tuple of distinct feature names touched by the pattern.
        Length equals the pattern's "order" (1, 2, or 3).
    description:
        Human-readable summary (e.g. "momentum_3 > +1σ AND session == NY_OPEN").
    """

    conditions: tuple[Condition, ...]
    feature_names: tuple[str, ...]
    description: str = field(default="")

    def __post_init__(self) -> None:
        if not self.description:
            desc = " AND ".join(c.describe() for c in self.conditions)
            object.__setattr__(self, "description", desc)

    @property
    def order(self) -> int:
        return len(self.feature_names)

    def matches(self, row: dict[str, Any]) -> bool:
        return all(c.evaluate(row) for c in self.conditions)


@dataclass(frozen=True)
class PatternResult:
    """
    Forward-test result for a single pattern, split across train and test
    halves of the snapshot sequence.

    Fields
    ──────
    pattern:
        The Pattern this result belongs to.
    sample_size:
        Total matching occurrences that produced a valid forward return
        (train + test combined).
    mean_return:
        Mean forward return across train + test — kept for backwards
        compatibility and top-line reporting.
    win_rate:
        Fraction of (train + test) occurrences with forward return > 0.
    sharpe_estimate:
        mean_return / stdev_return across the combined sample, or None
        if stdev is zero.
    train_sample_size, test_sample_size:
        Per-split occurrence counts.
    train_mean_return, test_mean_return:
        Mean forward return in each split.
    train_win_rate, test_win_rate:
        Per-split win rate.
    stability_score:
        test_mean_return / train_mean_return.  Values near 1.0 indicate
        consistent performance across splits.  None if train_mean_return
        is zero (cannot be computed).
    score:
        Composite ranking score:
            score = test_mean_return * sqrt(sample_size_total) * stability_score
        None when stability_score is None.
    """

    pattern: Pattern
    sample_size: int
    mean_return: float
    win_rate: float
    sharpe_estimate: float | None

    train_sample_size: int = 0
    test_sample_size: int = 0
    train_mean_return: float = 0.0
    test_mean_return: float = 0.0
    train_win_rate: float = 0.0
    test_win_rate: float = 0.0
    stability_score: float | None = None
    score: float | None = None

    def describe(self) -> str:
        sh = (
            f"{self.sharpe_estimate:+.3f}"
            if self.sharpe_estimate is not None
            else "  n/a"
        )
        return (
            f"n={self.sample_size:>5}  "
            f"mean={self.mean_return:+.5f}  "
            f"wr={self.win_rate * 100:5.1f}%  "
            f"sharpe={sh}  |  {self.pattern.description}"
        )

    def describe_split(self) -> str:
        stab = (
            f"{self.stability_score:+.2f}"
            if self.stability_score is not None
            else " n/a"
        )
        score = f"{self.score:+.4f}" if self.score is not None else "   n/a"
        return (
            f"score={score}  "
            f"n={self.sample_size:>4} "
            f"(tr={self.train_sample_size}/te={self.test_sample_size})  "
            f"train={self.train_mean_return:+.5f} "
            f"test={self.test_mean_return:+.5f}  "
            f"wr tr/te={self.train_win_rate * 100:4.1f}%/"
            f"{self.test_win_rate * 100:4.1f}%  "
            f"stab={stab}  |  {self.pattern.description}"
        )


@dataclass(frozen=True)
class CompactPatternResult:
    """
    Lightweight result used on the fast/vectorised path.

    Holds the PatternKey rather than a full Pattern object — an
    explicit `Pattern` is only built when the caller needs one.

    Fields
    ──────
    key                 : PatternKey (sorted tuple of (col, bin_value))
    sample_size         : number of matching rows with valid forward_return
    mean_return         : mean of forward_return_10 across matches
    win_rate            : fraction of matches with forward_win_10 == 1
    sharpe              : mean_return / stdev  (None if stdev == 0)
    train_sample_size   : per-split counts
    test_sample_size
    train_mean_return
    test_mean_return
    train_win_rate
    test_win_rate
    stability_score     : test_mean / train_mean  (None if train_mean == 0)
    score               : test_mean * sqrt(n_total) * stability_score (None if n/a)
    """

    key: PatternKey
    sample_size: int
    mean_return: float
    win_rate: float
    sharpe: float | None

    train_sample_size: int = 0
    test_sample_size: int = 0
    train_mean_return: float = 0.0
    test_mean_return: float = 0.0
    train_win_rate: float = 0.0
    test_win_rate: float = 0.0
    stability_score: float | None = None
    score: float | None = None

    @property
    def description(self) -> str:
        return format_pattern_key(self.key)

    def describe(self) -> str:
        stab = (
            f"{self.stability_score:+.2f}"
            if self.stability_score is not None
            else " n/a"
        )
        score = f"{self.score:+.4f}" if self.score is not None else "   n/a"
        return (
            f"score={score}  "
            f"n={self.sample_size:>5} "
            f"(tr={self.train_sample_size}/te={self.test_sample_size})  "
            f"train={self.train_mean_return:+.5f} "
            f"test={self.test_mean_return:+.5f}  "
            f"wr tr/te={self.train_win_rate * 100:4.1f}%/"
            f"{self.test_win_rate * 100:4.1f}%  "
            f"stab={stab}  |  {self.description}"
        )
