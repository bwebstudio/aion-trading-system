"""
Synthetic smoke test for sequential discovery.

Builds a 4,000-row synthetic compact matrix and injects a deterministic
temporal edge:

    (range_compression_bin == TRUE) at row i
  → (momentum_3_bin       == POS)  at row i+1
    →  forward_return_10[i+1] skewed POSITIVE

This simulates a classic "compression → breakout" edge.  A noise-only
pair should NOT come out on top.

Assertions:
  * discover() returns at least one sequence
  * the injected sequence appears in the top-5 by |score|
  * vectorised masks behave consistently with naive evaluation
    on a small synthetic window

Run:
    python -m research.sequential_discovery._smoke_test
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from research.pattern_discovery.feature_builder import BIN_COLUMNS
from research.sequential_discovery import (
    SequenceGenerator,
    build_event_masks,
    evaluate_sequence,
)


def _build_df(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def choose(options):
        return rng.choice(options, size=n)

    df = pd.DataFrame(
        {
            "idx": np.arange(n, dtype=np.int32),
            "timestamp": pd.date_range(
                "2026-01-01", periods=n, freq="min", tz="UTC"
            ),
            "symbol": np.array(["SYNTH"] * n),
            "open": np.full(n, 100.0, dtype=np.float32),
            "high": np.full(n, 100.5, dtype=np.float32),
            "low": np.full(n, 99.5, dtype=np.float32),
            "close": np.full(n, 100.0, dtype=np.float32),
            "atr_14": np.full(n, 0.3, dtype=np.float32),
            "distance_to_vwap": rng.normal(0, 1, n).astype(np.float32),
            "distance_to_session_high": rng.normal(-100, 50, n).astype(np.float32),
            "distance_to_session_low": rng.normal(100, 50, n).astype(np.float32),
            "momentum_3": rng.normal(0, 1, n).astype(np.float32),
            "momentum_5": rng.normal(0, 1, n).astype(np.float32),
            "range_compression": rng.integers(0, 2, n).astype(bool),
            "forward_return_10": rng.normal(0, 0.001, n).astype(np.float32),
            "forward_win_10": rng.integers(0, 2, n).astype(np.int8),
            "distance_to_vwap_bin": choose(
                ["LT_NEG_2SIG", "LT_NEG_1SIG", "MID",
                 "GT_POS_1SIG", "GT_POS_2SIG"]
            ),
            "distance_to_session_high_bin": choose(["NEAR", "MID", "FAR"]),
            "distance_to_session_low_bin": choose(["NEAR", "MID", "FAR"]),
            "momentum_3_bin": choose(["NEG", "POS"]),
            "momentum_5_bin": choose(["NEG", "POS"]),
            "range_compression_bin": choose(["TRUE", "FALSE"]),
            "session_bin": choose(
                ["ASIA", "LONDON", "NY_OPEN", "NY_MID", "NY_CLOSE"]
            ),
            "time_of_day_bucket": choose(
                ["T_00_04", "T_04_08", "T_08_12",
                 "T_12_16", "T_16_20", "T_20_24"]
            ),
        }
    )
    for col in BIN_COLUMNS:
        df[col] = df[col].astype("category")

    # ── Inject a temporal edge ───────────────────────────────────────────
    # When row i has range_compression_bin=TRUE and row i+1 has
    # momentum_3_bin=POS, give forward_return_10[i+1] a strong positive bump.
    rc = (df["range_compression_bin"] == "TRUE").to_numpy()
    mom = (df["momentum_3_bin"] == "POS").to_numpy()
    # sequence ends at row i+1, so boost forward_return_10[i+1]
    end_mask = np.zeros(n, dtype=bool)
    end_mask[1:] = rc[:-1] & mom[1:]
    boost = np.where(end_mask, 0.004, 0.0).astype(np.float32)
    df["forward_return_10"] = (
        df["forward_return_10"].to_numpy() + boost
    ).astype(np.float32)
    return df


def _naive_sequence_count(df: pd.DataFrame, sequence) -> int:
    """Reference implementation: plain Python loop over rows."""
    n = len(df)
    L = len(sequence)
    cols = [df[col].astype(str).to_numpy() for col, _ in sequence]
    vals = [val for _, val in sequence]
    count = 0
    ret = df["forward_return_10"].to_numpy()
    for j in range(L - 1, n):
        ok = True
        for k in range(L):
            if cols[k][j - (L - 1 - k)] != vals[k]:
                ok = False
                break
        if ok and np.isfinite(ret[j]):
            count += 1
    return count


def main() -> None:
    print("sequential_discovery — smoke test")
    print("-" * 60)

    df = _build_df()
    print(f"df: {len(df):,} rows  {len(df.columns)} cols")

    # ── Consistency check: vectorised end-mask vs naive loop ─────────────
    event_masks = build_event_masks(df, BIN_COLUMNS, min_support=50)
    sample_seq = (
        ("range_compression_bin", "TRUE"),
        ("momentum_3_bin", "POS"),
    )
    returns = df["forward_return_10"].to_numpy(dtype=np.float32)
    valid = np.isfinite(returns)
    res = evaluate_sequence(sample_seq, event_masks, returns, valid)
    n_naive = _naive_sequence_count(df, sample_seq)
    print(
        f"injected pair    n_vectorised={res.n_samples}  n_naive={n_naive}"
    )
    assert res.n_samples == n_naive, (
        f"vectorised / naive mismatch: {res.n_samples} vs {n_naive}"
    )

    # ── Run discovery ────────────────────────────────────────────────────
    gen = SequenceGenerator(
        min_samples=100,
        minimal_edge_threshold=0.0002,
        max_length=3,
        train_fraction=0.70,
    )
    results = gen.discover(df, BIN_COLUMNS)
    print(f"discovered       {len(results):,} sequences")

    assert len(results) >= 1, "expected at least 1 sequence"

    top5 = results[:5]
    print()
    print("top 5:")
    for r in top5:
        print(f"  {r.describe()}")

    # The injected pair should be among the top 5 by |score|.
    injected = (
        ("range_compression_bin", "TRUE"),
        ("momentum_3_bin", "POS"),
    )
    top5_keys = [tuple(r.sequence) for r in top5]
    assert injected in top5_keys, (
        f"injected sequence missing from top-5; got {top5_keys}"
    )

    # Length-3 survivors should exist (expansion from the strong L2 edge).
    l3 = [r for r in results if r.length == 3]
    print(f"length-2: {sum(1 for r in results if r.length == 2):>4}")
    print(f"length-3: {len(l3):>4}")

    # At least one length-3 sequence extending the injected pair.
    has_l3_ext = any(
        r.sequence[:2] == injected for r in l3
    )
    print(f"length-3 extending injected pair: {has_l3_ext}")

    print()
    print("SMOKE OK")


if __name__ == "__main__":
    main()
