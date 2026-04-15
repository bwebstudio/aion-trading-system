"""
research.pattern_discovery.forward_tester
──────────────────────────────────────────
Evaluate pattern forward-return metrics using the production ExecutionModel,
with a chronological train/test split and a stability filter.

Forward simulation (per match at feature_matrix index i):

    entry_snap  = snapshots[i + 1]
    exit_snap   = snapshots[i + 1 + N]   (N = forward_bars, default 10)

    spread_e    = ExecutionModel.estimate_spread(symbol, atr_1m)
    slip_e      = ExecutionModel.estimate_slippage(entry_bar, session, "retest")
    entry_price = entry_snap.latest_bar.open + spread_e/2 + slip_e

    spread_x    = ExecutionModel.estimate_spread(symbol, atr_1m_exit)
    slip_x      = ExecutionModel.estimate_slippage(exit_bar, session_x, "retest")
    exit_price  = exit_snap.latest_bar.close - spread_x/2 - slip_x

    forward_return = (exit_price - entry_price) / entry_price

All evaluation is long-direction.  A strongly negative mean_return
indicates a short-direction edge.

Train/Test split
────────────────
The snapshot sequence is split chronologically:

    train = first `train_fraction`     (default 0.70)
    test  = remainder                  (default 0.30)

Metrics are computed separately on each half.  The stability filter
discards patterns that fail any of:

    sample_size_train >= min_samples_train (default 100)
    sample_size_test  >= min_samples_test  (default 50)
    sign(train_mean_return) == sign(test_mean_return)
    |train_mean - test_mean| / |train_mean| < stability_tolerance
                                              (default 0.50)

Ranking score
─────────────
    score = test_mean_return * sqrt(sample_size_total) * stability_score
            where stability_score = test_mean_return / train_mean_return
"""

from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING, Any, Iterable

from aion.core.models import MarketSnapshot
from aion.execution.execution_model import ExecutionModel, detect_session
from research.pattern_discovery.pattern_types import (
    CompactPatternResult,
    Pattern,
    PatternKey,
    PatternResult,
)

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class ForwardTester:
    """
    Evaluate a batch of patterns against a snapshot sequence with a
    chronological train/test split and a stability filter.

    Parameters
    ----------
    execution_model:
        Instance of aion.execution.ExecutionModel.  Spread and slippage
        are always applied.
    forward_bars:
        Number of snapshots to hold after entry.  Default 10.
    train_fraction:
        Fraction (0, 1) of the snapshot sequence used for training.
        The remainder is the test set.  Default 0.70.
    min_samples_train, min_samples_test:
        Minimum per-split occurrence counts.  Defaults 100 / 50.
    stability_tolerance:
        Max allowed relative difference between train and test means,
        expressed as a fraction of |train_mean|.  Default 0.50.
    entry_type:
        Entry type passed to ExecutionModel.estimate_slippage.
        Default "retest".
    """

    def __init__(
        self,
        execution_model: ExecutionModel,
        *,
        forward_bars: int = 10,
        train_fraction: float = 0.70,
        min_samples_train: int = 100,
        min_samples_test: int = 50,
        stability_tolerance: float = 0.50,
        entry_type: str = "retest",
    ) -> None:
        if not 0.0 < train_fraction < 1.0:
            raise ValueError(f"train_fraction must be in (0, 1); got {train_fraction}")

        self.execution_model = execution_model
        self.forward_bars = forward_bars
        self.train_fraction = train_fraction
        self.min_samples_train = min_samples_train
        self.min_samples_test = min_samples_test
        self.stability_tolerance = stability_tolerance
        self.entry_type = entry_type

    # ─── Public API ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        snapshots: list[MarketSnapshot],
        feature_matrix: list[dict[str, Any]],
        patterns: list[Pattern],
    ) -> list[PatternResult]:
        """
        Return a PatternResult per pattern that passes the stability filter.
        Patterns that fail any of the stability rules are dropped.
        """
        returns_cache = self._precompute_returns(snapshots)
        split_idx = int(len(snapshots) * self.train_fraction)

        results: list[PatternResult] = []

        for pattern in patterns:
            train_sample: list[float] = []
            test_sample: list[float] = []

            for row in feature_matrix:
                if not pattern.matches(row):
                    continue
                ret = returns_cache.get(row["idx"])
                if ret is None:
                    continue
                if row["idx"] < split_idx:
                    train_sample.append(ret)
                else:
                    test_sample.append(ret)

            result = self._build_result(pattern, train_sample, test_sample)
            if result is None:
                continue
            results.append(result)

        return results

    # ─── Stability + ranking ──────────────────────────────────────────────────

    def _build_result(
        self,
        pattern: Pattern,
        train_sample: list[float],
        test_sample: list[float],
    ) -> PatternResult | None:
        n_train = len(train_sample)
        n_test = len(test_sample)

        # Sample-size filter
        if n_train < self.min_samples_train or n_test < self.min_samples_test:
            return None

        train_mean = sum(train_sample) / n_train
        test_mean = sum(test_sample) / n_test
        train_wins = sum(1 for r in train_sample if r > 0)
        test_wins = sum(1 for r in test_sample if r > 0)
        train_wr = train_wins / n_train
        test_wr = test_wins / n_test

        # Sign filter — both splits must agree in direction.
        if train_mean == 0.0 or test_mean == 0.0:
            return None
        if (train_mean > 0) != (test_mean > 0):
            return None

        # Magnitude filter — stability tolerance on |train − test| / |train|.
        rel_gap = abs(train_mean - test_mean) / abs(train_mean)
        if rel_gap >= self.stability_tolerance:
            return None

        stability_score = test_mean / train_mean

        # Combined metrics for backwards-compatible reporting.
        combined = train_sample + test_sample
        total_n = n_train + n_test
        combined_mean = sum(combined) / total_n
        combined_wr = (train_wins + test_wins) / total_n
        if total_n >= 2:
            var = sum((r - combined_mean) ** 2 for r in combined) / (total_n - 1)
            std = math.sqrt(var)
            sharpe = combined_mean / std if std > 0 else None
        else:
            sharpe = None

        score = test_mean * math.sqrt(total_n) * stability_score

        return PatternResult(
            pattern=pattern,
            sample_size=total_n,
            mean_return=combined_mean,
            win_rate=combined_wr,
            sharpe_estimate=sharpe,
            train_sample_size=n_train,
            test_sample_size=n_test,
            train_mean_return=train_mean,
            test_mean_return=test_mean,
            train_win_rate=train_wr,
            test_win_rate=test_wr,
            stability_score=stability_score,
            score=score,
        )

    # ─── Forward-return pre-computation ───────────────────────────────────────

    def _precompute_returns(
        self, snapshots: list[MarketSnapshot]
    ) -> dict[int, float]:
        """
        Map signal-row index → forward return.  Missing keys mean the
        forward simulation could not be run from that index.
        """
        n = len(snapshots)
        cache: dict[int, float] = {}
        for i in range(n):
            entry_idx = i + 1
            exit_idx = i + 1 + self.forward_bars
            if exit_idx >= n:
                break

            entry_snap = snapshots[entry_idx]
            exit_snap = snapshots[exit_idx]

            ret = self._simulate(entry_snap, exit_snap)
            if ret is not None:
                cache[i] = ret
        return cache

    def _simulate(
        self,
        entry_snap: MarketSnapshot,
        exit_snap: MarketSnapshot,
    ) -> float | None:
        entry_bar = entry_snap.latest_bar
        exit_bar = exit_snap.latest_bar
        symbol = entry_snap.symbol

        atr_entry = entry_snap.feature_vector.atr_14
        atr_exit = exit_snap.feature_vector.atr_14

        try:
            spread_e = self.execution_model.estimate_spread(symbol, atr_entry)
            slip_e = self.execution_model.estimate_slippage(
                entry_bar,
                session=detect_session(entry_bar.timestamp_utc),
                entry_type=self.entry_type,
                symbol=symbol,
            )
            spread_x = self.execution_model.estimate_spread(symbol, atr_exit)
            slip_x = self.execution_model.estimate_slippage(
                exit_bar,
                session=detect_session(exit_bar.timestamp_utc),
                entry_type=self.entry_type,
                symbol=symbol,
            )
        except Exception:
            return None

        entry_price = entry_bar.open + spread_e / 2.0 + slip_e
        exit_price = exit_bar.close - spread_x / 2.0 - slip_x

        if entry_price <= 0:
            return None
        return (exit_price - entry_price) / entry_price

    # ═════════════════════════════════════════════════════════════════════════
    # Fast / vectorised path (v3)
    # ═════════════════════════════════════════════════════════════════════════

    def evaluate_patterns(
        self,
        df: "pd.DataFrame",
        pattern_keys: Iterable[PatternKey],
        *,
        batch_size: int = 1000,
        top_k: int = 500,
        min_samples: int = 100,
        train_fraction: float = 0.70,
        min_samples_train: int = 100,
        min_samples_test: int = 50,
        stability_tolerance: float = 0.50,
        progress_every: int = 5000,
    ) -> list[CompactPatternResult]:
        """
        Vectorised, batched pattern evaluation with top-k retention.

        Consumes PatternKey values from `pattern_keys` (a generator) in
        chunks of `batch_size`, scores each with numpy masks on the
        compact DataFrame, and keeps only the top-k by composite score.

        Memory: only the compact DataFrame + a top-k heap of at most
        `top_k` results are held in RAM.  Pattern keys are streamed.

        Parameters mirror the v2 stability filter; patterns that fail
        any per-split rule (`min_samples_train`, `min_samples_test`,
        sign agreement, relative gap) are discarded before entering the heap.

        Returns
        -------
        list[CompactPatternResult]
            Results sorted descending by |score|, largest first.
        """
        import numpy as np

        # ── Precompute per-column int8 codes and return arrays ────────────────
        n = len(df)
        if n == 0:
            return []

        returns = df["forward_return_10"].to_numpy(dtype=np.float32, copy=False)
        wins = df["forward_win_10"].to_numpy(dtype=np.int8, copy=False)
        valid_ret = np.isfinite(returns)

        split_idx = int(n * train_fraction)
        is_train = np.zeros(n, dtype=bool)
        is_train[:split_idx] = True

        code_arrays: dict[str, "np.ndarray"] = {}
        code_lookup: dict[str, dict[str, int]] = {}
        for col in df.columns:
            if df[col].dtype.name != "category":
                continue
            cat = df[col].cat
            codes = cat.codes.to_numpy(copy=False)  # int8
            mapping = {str(v): i for i, v in enumerate(cat.categories)}
            code_arrays[col] = codes
            code_lookup[col] = mapping

        # ── Top-k heap: stores (sort_key, counter, result) ───────────────────
        # Sort key is -|score| so heap keeps *worst* at the top → replace it
        # when we find something better.  Counter breaks ties deterministically.
        heap: list[tuple[float, int, CompactPatternResult]] = []
        counter = 0
        evaluated = 0
        survived = 0

        # ── Iterate in batches ───────────────────────────────────────────────
        batch: list[PatternKey] = []
        for key in pattern_keys:
            batch.append(key)
            if len(batch) >= batch_size:
                counter, evaluated, survived = self._process_batch(
                    batch,
                    code_arrays,
                    code_lookup,
                    returns,
                    wins,
                    valid_ret,
                    is_train,
                    heap,
                    counter=counter,
                    evaluated=evaluated,
                    survived=survived,
                    min_samples=min_samples,
                    min_samples_train=min_samples_train,
                    min_samples_test=min_samples_test,
                    stability_tolerance=stability_tolerance,
                    top_k=top_k,
                )
                batch.clear()
                if progress_every and evaluated % progress_every < batch_size:
                    print(
                        f"    evaluated={evaluated:>6}  "
                        f"survived={survived:>5}  "
                        f"heap={len(heap):>4}"
                    )

        if batch:
            counter, evaluated, survived = self._process_batch(
                batch,
                code_arrays,
                code_lookup,
                returns,
                wins,
                valid_ret,
                is_train,
                heap,
                counter=counter,
                evaluated=evaluated,
                survived=survived,
                min_samples=min_samples,
                min_samples_train=min_samples_train,
                min_samples_test=min_samples_test,
                stability_tolerance=stability_tolerance,
                top_k=top_k,
            )

        results = [entry[2] for entry in heap]
        results.sort(
            key=lambda r: abs(r.score) if r.score is not None else 0.0,
            reverse=True,
        )
        return results

    # ─── Batch processor ──────────────────────────────────────────────────────

    @staticmethod
    def _process_batch(
        batch: list[PatternKey],
        code_arrays: dict[str, "np.ndarray"],
        code_lookup: dict[str, dict[str, int]],
        returns: "np.ndarray",
        wins: "np.ndarray",
        valid_ret: "np.ndarray",
        is_train: "np.ndarray",
        heap: list[tuple[float, int, CompactPatternResult]],
        *,
        counter: int,
        evaluated: int,
        survived: int,
        min_samples: int,
        min_samples_train: int,
        min_samples_test: int,
        stability_tolerance: float,
        top_k: int,
    ) -> tuple[int, int, int]:
        import numpy as np

        for key in batch:
            evaluated += 1

            # Build the boolean mask from per-column == comparisons.
            mask = valid_ret
            viable = True
            for col, val in key:
                codes = code_arrays.get(col)
                lookup = code_lookup.get(col)
                if codes is None or lookup is None:
                    viable = False
                    break
                code = lookup.get(val)
                if code is None:
                    viable = False
                    break
                mask = mask & (codes == code)
            if not viable:
                continue

            # Cheap min-sample gate before per-split work.
            total_n = int(mask.sum())
            if total_n < min_samples:
                continue

            train_mask = mask & is_train
            test_mask = mask & ~is_train
            n_tr = int(train_mask.sum())
            n_te = int(test_mask.sum())
            if n_tr < min_samples_train or n_te < min_samples_test:
                continue

            ret_tr = returns[train_mask]
            ret_te = returns[test_mask]
            win_tr = wins[train_mask]
            win_te = wins[test_mask]

            m_tr = float(ret_tr.mean())
            m_te = float(ret_te.mean())
            if m_tr == 0.0 or m_te == 0.0:
                continue
            if (m_tr > 0) != (m_te > 0):
                continue
            if abs(m_tr - m_te) / abs(m_tr) >= stability_tolerance:
                continue

            stability = m_te / m_tr
            score = m_te * math.sqrt(total_n) * stability

            combined = np.concatenate([ret_tr, ret_te])
            combined_mean = float(combined.mean())
            combined_wr = float(
                (int(win_tr.sum()) + int(win_te.sum())) / total_n
            )
            if total_n >= 2:
                std = float(combined.std(ddof=1))
                sharpe = combined_mean / std if std > 0 else None
            else:
                sharpe = None

            result = CompactPatternResult(
                key=tuple(key),
                sample_size=total_n,
                mean_return=combined_mean,
                win_rate=combined_wr,
                sharpe=sharpe,
                train_sample_size=n_tr,
                test_sample_size=n_te,
                train_mean_return=m_tr,
                test_mean_return=m_te,
                train_win_rate=float(win_tr.mean()) if n_tr else 0.0,
                test_win_rate=float(win_te.mean()) if n_te else 0.0,
                stability_score=stability,
                score=score,
            )
            survived += 1

            # Top-k heap keyed by |score| (min-heap — smallest |score| at root).
            heap_key = abs(score)
            counter += 1
            if len(heap) < top_k:
                heapq.heappush(heap, (heap_key, counter, result))
            elif heap_key > heap[0][0]:
                heapq.heapreplace(heap, (heap_key, counter, result))

        return counter, evaluated, survived
