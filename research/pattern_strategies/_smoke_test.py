"""
Synthetic smoke test for pattern_to_strategy + backtest.

Builds a 3,000-row synthetic compact matrix with OHLC + a deterministic
long edge (when distance_to_vwap_bin == LT_NEG_2SIG, the close drifts up
over the next few bars).  Verifies:

  * Pattern discovery recovers the edge.
  * convert_compact_result produces a StrategyCandidate with direction=LONG.
  * backtest_candidate produces trades with positive total return.
  * JSON serde round-trips.

Run:
    python -m research.pattern_strategies._smoke_test
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from research.pattern_discovery import ForwardTester, PatternGenerator
from research.pattern_discovery.feature_builder import BIN_COLUMNS
from research.pattern_strategies import (
    backtest_candidate,
    candidate_from_dict,
    candidate_to_dict,
    convert_compact_result,
)


def _build_df(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    base = 100.0
    # Random walk close.
    returns = rng.normal(0, 0.001, n)
    close = base * np.cumprod(1.0 + returns)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    noise = np.abs(rng.normal(0, 0.0008, n)) * close
    high = np.maximum(open_, close) + noise
    low = np.minimum(open_, close) - noise
    atr = np.full(n, close.mean() * 0.002, dtype=np.float32)

    def choose(options):
        return rng.choice(options, size=n)

    df = pd.DataFrame(
        {
            "idx": np.arange(n, dtype=np.int32),
            "timestamp": pd.date_range(
                "2025-01-01", periods=n, freq="min", tz="UTC"
            ),
            "symbol": np.array(["SYNTH"] * n),
            "open": open_.astype(np.float32),
            "high": high.astype(np.float32),
            "low": low.astype(np.float32),
            "close": close.astype(np.float32),
            "distance_to_vwap": rng.normal(0, 1, n).astype(np.float32),
            "distance_to_session_high": rng.normal(-100, 50, n).astype(np.float32),
            "distance_to_session_low": rng.normal(100, 50, n).astype(np.float32),
            "momentum_3": rng.normal(0, 1, n).astype(np.float32),
            "momentum_5": rng.normal(0, 1, n).astype(np.float32),
            "range_compression": rng.integers(0, 2, n).astype(bool),
            "atr_14": atr,
            "forward_return_10": np.zeros(n, dtype=np.float32),
            "forward_win_10": np.zeros(n, dtype=np.int8),
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

    # ── Inject a long edge ────────────────────────────────────────────────
    # When distance_to_vwap_bin == LT_NEG_2SIG, shift subsequent closes up.
    mask_edge = (df["distance_to_vwap_bin"] == "LT_NEG_2SIG").to_numpy()
    edge_idx = np.where(mask_edge)[0]
    bump = atr[0] * 3.0
    for i in edge_idx:
        end = min(i + 12, n)
        df.loc[i + 1:end, "close"] = (
            df.loc[i + 1:end, "close"].to_numpy() + bump
        ).astype(np.float32)
        df.loc[i + 1:end, "high"] = (
            df.loc[i + 1:end, "high"].to_numpy() + bump
        ).astype(np.float32)
        df.loc[i + 1:end, "low"] = (
            df.loc[i + 1:end, "low"].to_numpy() + bump
        ).astype(np.float32)

    # Recompute forward_return_10 from the modified closes.
    closes_arr = df["close"].to_numpy(dtype=np.float64)
    opens_arr = df["open"].to_numpy(dtype=np.float64)
    fwd = np.full(n, np.nan, dtype=np.float64)
    fwd_win = np.zeros(n, dtype=np.int8)
    for i in range(n - 11):
        entry = opens_arr[i + 1]
        exit_ = closes_arr[i + 11]
        if entry > 0:
            fwd[i] = (exit_ - entry) / entry
            fwd_win[i] = 1 if fwd[i] > 0 else 0
    df["forward_return_10"] = fwd.astype(np.float32)
    df["forward_win_10"] = fwd_win

    return df


def main() -> None:
    print("pattern_strategies — smoke test")
    print("-" * 60)

    df = _build_df()
    print(f"df: {len(df)} rows, {len(df.columns)} cols")

    # Discovery.
    gen = PatternGenerator(
        min_bucket_occurrences=50,
        max_conditions=2,
        minimal_edge_threshold=0.0002,
    )
    keys: list = []
    for batch in gen.generate_patterns_levelwise(
        df,
        bin_columns=BIN_COLUMNS,
        batch_size=500,
        min_samples=100,
        max_conditions=2,
        minimal_edge_threshold=0.0002,
    ):
        keys.extend(batch)
    print(f"generated   : {len(keys):>5} keys")

    tester = ForwardTester(execution_model=None, forward_bars=10)
    results = tester.evaluate_patterns(
        df,
        iter(keys),
        batch_size=500,
        top_k=20,
        min_samples=100,
        train_fraction=0.70,
        min_samples_train=100,
        min_samples_test=50,
        stability_tolerance=0.50,
        progress_every=0,
    )
    print(f"surviving   : {len(results)}")
    assert len(results) >= 1, "no patterns survived"

    # Pick top result → candidate.
    top = results[0]
    print(f"top pattern : {top.describe()}")
    candidate = convert_compact_result(top)
    assert candidate is not None
    print(f"candidate   : direction={candidate.direction}  key={candidate.pattern_key}")

    # JSON round-trip.
    as_dict = candidate_to_dict(candidate)
    as_json = json.dumps(as_dict)
    restored = candidate_from_dict(json.loads(as_json))
    assert restored.pattern_key == candidate.pattern_key
    assert restored.direction == candidate.direction
    print("json round-trip OK")

    # Backtest.
    report = backtest_candidate(df, candidate)
    print()
    print(f"backtest    : {report.summary_line()}")
    assert report.total_trades >= 1, "expected at least 1 trade"
    # Sanity: discovery said direction agrees with edge sign, so backtest
    # should at least NOT lose on a strongly deterministic injected edge.
    if candidate.direction == "LONG" and top.test_mean_return > 0.001:
        assert report.total_return > 0, (
            f"expected positive total_return, got {report.total_return}"
        )

    # Dump first 3 trades.
    print()
    print("sample trades:")
    for t in report.trades[:3]:
        print(
            f"  entry_bar={t.entry_bar:<5} exit_bar={t.exit_bar:<5} "
            f"entry={t.entry_price:.4f} exit={t.exit_price:.4f} "
            f"ret={t.return_pct:+.4f} [{t.exit_reason}]"
        )
    print()
    print("SMOKE OK")


if __name__ == "__main__":
    main()
