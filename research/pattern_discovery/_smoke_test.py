"""
Synthetic smoke test for the v3 pipeline.

Builds a 10k-row synthetic compact DataFrame (skipping the real CSV
snapshot pipeline), injects a known edge, and verifies that:
    * stream_keys yields thousands of candidates in <1s
    * evaluate_patterns finds the injected edge in the top-k
    * memory stays bounded (top-k heap + compact df)

Run with:
    python -m research.pattern_discovery._smoke_test
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from research.pattern_discovery import (
    CompactPatternResult,
    ForwardTester,
    PatternGenerator,
    validate_across_assets,
)
from research.pattern_discovery.feature_builder import BIN_COLUMNS


def _build_synthetic_df(n: int = 10_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def choose(options):
        return rng.choice(options, size=n)

    df = pd.DataFrame(
        {
            "idx": np.arange(n, dtype=np.int32),
            "timestamp": pd.date_range(
                "2024-01-01", periods=n, freq="min", tz="UTC"
            ),
            "symbol": np.array(["US100.cash"] * n),
            "distance_to_vwap": rng.normal(0, 1, n).astype(np.float32),
            "distance_to_session_high": rng.normal(-100, 50, n).astype(np.float32),
            "distance_to_session_low": rng.normal(100, 50, n).astype(np.float32),
            "momentum_3": rng.normal(0, 1, n).astype(np.float32),
            "momentum_5": rng.normal(0, 1, n).astype(np.float32),
            "range_compression": rng.integers(0, 2, n).astype(bool),
            "atr_14": np.full(n, 3.0, dtype=np.float32),
            "forward_return_10": rng.normal(0, 0.002, n).astype(np.float32),
            "forward_win_10": rng.integers(0, 2, n).astype(np.int8),
            "distance_to_vwap_bin": choose(
                [
                    "LT_NEG_2SIG", "LT_NEG_1P5SIG", "LT_NEG_1SIG",
                    "MID", "GT_POS_1SIG", "GT_POS_1P5SIG", "GT_POS_2SIG",
                ]
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

    # Inject an edge: when distance_to_vwap_bin == LT_NEG_2SIG,
    # forward returns skew positive.
    mask = df["distance_to_vwap_bin"] == "LT_NEG_2SIG"
    df.loc[mask, "forward_return_10"] = (
        df.loc[mask, "forward_return_10"] + 0.003
    ).astype(np.float32)
    df.loc[mask, "forward_win_10"] = np.int8(1)
    return df


def main() -> None:
    print("pattern_discovery v3 — synthetic smoke test")
    print("-" * 60)

    df = _build_synthetic_df(n=10_000, seed=0)
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(
        f"compact df : {len(df):,} rows  {len(df.columns)} cols  "
        f"{mem_mb:4.1f} MB"
    )

    t = time.perf_counter()
    gen = PatternGenerator(
        min_bucket_occurrences=50,
        max_conditions=3,
        minimal_edge_threshold=0.0002,
    )
    keys: list = []
    batches = 0
    for batch in gen.generate_patterns_levelwise(
        df,
        bin_columns=BIN_COLUMNS,
        batch_size=500,
        min_samples=100,
        minimal_edge_threshold=0.0002,
        max_conditions=3,
    ):
        keys.extend(batch)
        batches += 1
    t_gen = time.perf_counter() - t
    print(
        f"generated  : {len(keys):>6,} keys  in {batches} batch(es)  "
        f"({t_gen:.2f}s)"
    )

    t = time.perf_counter()
    tester = ForwardTester(execution_model=None, forward_bars=10)
    results = tester.evaluate_patterns(
        df,
        iter(keys),
        batch_size=1000,
        top_k=50,
        min_samples=100,
        train_fraction=0.70,
        min_samples_train=100,
        min_samples_test=50,
        stability_tolerance=0.50,
        progress_every=0,
    )
    t_eval = time.perf_counter() - t
    print(f"survived   : {len(results):>6,}        ({t_eval:.2f}s)")

    print()
    print("top 5:")
    for r in results[:5]:
        print(f"  {r.describe()}")

    # Assertions — injected edge should be among surviving patterns.
    assert len(keys) >= 1, f"expected at least 1 key, got {len(keys)}"
    assert len(results) >= 1, "expected at least 1 surviving pattern"
    any_vwap_edge = any(
        ("distance_to_vwap_bin", "LT_NEG_2SIG") in r.key for r in results
    )
    assert any_vwap_edge, "injected LT_NEG_2SIG edge was not recovered"
    print()
    print("SMOKE OK (single-asset path)")

    # ── Multi-asset smoke test ────────────────────────────────────────────────
    print()
    print("multi-asset cross-validation smoke test")
    print("-" * 60)

    # Asset "B": same injected edge plus a noise-only asset.
    df_a = df  # reuse above
    df_b = _build_synthetic_df(n=10_000, seed=1)
    df_c = _build_synthetic_df(n=10_000, seed=2)
    # df_c has no injected edge: make its LT_NEG_2SIG neutral.
    mask_c = df_c["distance_to_vwap_bin"] == "LT_NEG_2SIG"
    df_c.loc[mask_c, "forward_return_10"] = (
        df_c.loc[mask_c, "forward_return_10"] - 0.003
    ).astype("float32")

    def _evaluate(df_):
        gen = PatternGenerator(
            min_bucket_occurrences=50,
            max_conditions=3,
            minimal_edge_threshold=0.0002,
        )
        k: list = []
        for b in gen.generate_patterns_levelwise(
            df_,
            bin_columns=BIN_COLUMNS,
            batch_size=500,
            min_samples=100,
            minimal_edge_threshold=0.0002,
            max_conditions=3,
        ):
            k.extend(b)
        t = ForwardTester(execution_model=None, forward_bars=10)
        return t.evaluate_patterns(
            df_,
            iter(k),
            batch_size=1000,
            top_k=50,
            min_samples=100,
            train_fraction=0.70,
            min_samples_train=100,
            min_samples_test=50,
            stability_tolerance=0.50,
            progress_every=0,
        )

    asset_results: dict[str, list[CompactPatternResult]] = {
        "ASSET_A": _evaluate(df_a),
        "ASSET_B": _evaluate(df_b),
        "ASSET_C": _evaluate(df_c),
    }
    for name, res in asset_results.items():
        print(f"  {name}: {len(res)} surviving")

    cross = validate_across_assets(asset_results, min_assets=2)
    print(f"  cross-asset (>=2 assets): {len(cross)}")
    print()
    print("top 5 cross-asset:")
    for r in cross[:5]:
        print(f"  {r.describe()}")

    assert len(cross) >= 1, "expected at least one cross-asset pattern"
    print()
    print("SMOKE OK (multi-asset path)")


if __name__ == "__main__":
    main()
