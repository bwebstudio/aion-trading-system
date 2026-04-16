"""
Synthetic smoke test for sequential_strategies.

Builds a compact matrix with a deterministic temporal edge
(compression → breakout), discovers the sequence, converts it to a
SequentialStrategyCandidate, and backtests it.  Verifies the full
loop end-to-end plus JSON round-trip.

Run:
    python -m research.sequential_strategies._smoke_test
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

from research.pattern_discovery.feature_builder import BIN_COLUMNS
from research.sequential_discovery import SequenceGenerator
from research.sequential_strategies import (
    backtest_sequential_candidate,
    candidate_from_dict,
    candidate_to_dict,
    convert_sequence_result,
)


def _build_df(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    base = 100.0
    returns = rng.normal(0, 0.001, n)
    close = base * np.cumprod(1.0 + returns)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    noise = np.abs(rng.normal(0, 0.0008, n)) * close
    high = np.maximum(open_, close) + noise
    low = np.minimum(open_, close) - noise
    atr = np.full(n, close.mean() * 0.003, dtype=np.float32)

    def choose(options):
        return rng.choice(options, size=n)

    df = pd.DataFrame(
        {
            "idx": np.arange(n, dtype=np.int32),
            "timestamp": pd.date_range(
                "2026-01-01", periods=n, freq="min", tz="UTC"
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

    # Inject compression → breakout: when row i has
    # range_compression_bin=TRUE AND row i+1 has momentum_3_bin=POS,
    # push prices upward for a few bars after i+1 so a LONG entry
    # on bar i+2 captures the move.
    rc = (df["range_compression_bin"] == "TRUE").to_numpy()
    mom = (df["momentum_3_bin"] == "POS").to_numpy()
    end_mask = np.zeros(n, dtype=bool)
    end_mask[1:] = rc[:-1] & mom[1:]
    end_idx = np.where(end_mask)[0]

    bump = atr[0] * 3.0
    for i in end_idx:
        hi = min(i + 11, n)
        df.loc[i + 1:hi, "close"] = (
            df.loc[i + 1:hi, "close"].to_numpy() + bump
        ).astype(np.float32)
        df.loc[i + 1:hi, "high"] = (
            df.loc[i + 1:hi, "high"].to_numpy() + bump
        ).astype(np.float32)
        df.loc[i + 1:hi, "low"] = (
            df.loc[i + 1:hi, "low"].to_numpy() + bump
        ).astype(np.float32)

    # Recompute forward_return_10 from modified closes.
    closes_arr = df["close"].to_numpy(dtype=np.float64)
    opens_arr = df["open"].to_numpy(dtype=np.float64)
    fwd = np.full(n, np.nan, dtype=np.float64)
    for i in range(n - 11):
        entry = opens_arr[i + 1]
        exit_ = closes_arr[i + 11]
        if entry > 0:
            fwd[i] = (exit_ - entry) / entry
    df["forward_return_10"] = fwd.astype(np.float32)
    df["forward_win_10"] = (fwd > 0).astype(np.int8)
    return df


def main() -> None:
    print("sequential_strategies — smoke test")
    print("-" * 60)

    df = _build_df()
    print(f"df: {len(df):,} rows  {len(df.columns)} cols")

    # Discover.
    gen = SequenceGenerator(
        min_samples=100,
        minimal_edge_threshold=0.0002,
        max_length=2,
        train_fraction=0.70,
    )
    results = gen.discover(df, BIN_COLUMNS)
    print(f"discovered   : {len(results):,} sequences")
    assert len(results) >= 1, "no sequences discovered"

    top = results[0]
    print(f"top          : {top.describe()}")

    # Convert.
    candidate = convert_sequence_result(top)
    assert candidate is not None
    print(
        f"candidate    : {candidate.name[:70]}  "
        f"direction={candidate.direction}"
    )

    # JSON round-trip.
    d = candidate_to_dict(candidate)
    s = json.dumps(d)
    restored = candidate_from_dict(json.loads(s))
    assert restored.sequence_key == candidate.sequence_key
    assert restored.direction == candidate.direction
    print("json round-trip OK")

    # Backtest.
    report = backtest_sequential_candidate(df, candidate)
    print()
    print(f"backtest     : {report.summary_line()}")
    assert report.total_trades >= 1, "no trades from backtest"

    # Sample trades.
    print()
    print("sample trades (first 3):")
    for t in report.trades[:3]:
        print(
            f"  entry_bar={t.entry_bar:<5} exit_bar={t.exit_bar:<5} "
            f"entry={t.entry_price:.4f} exit={t.exit_price:.4f} "
            f"ret={t.return_pct:+.4f} [{t.exit_reason}]"
        )

    # Sanity: on a strong injected long edge, total_return should be positive.
    if candidate.direction == "LONG":
        assert report.total_return > 0, (
            f"expected positive total_return for injected LONG edge; "
            f"got {report.total_return:+.4f}"
        )

    print()
    print("SMOKE OK")


if __name__ == "__main__":
    main()
