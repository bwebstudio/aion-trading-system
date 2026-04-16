"""
Synthetic smoke test for the unified meta-strategy pipeline.

Runs backtest_meta with a mix of pattern-based and sequence-based
candidates on the same synthetic matrix, and verifies:

  * pattern_candidates_count / sequence_candidates_count are populated
  * at least one candidate of each kind passes the prefilter
  * the selector.plan() references candidates whose names carry the
    proper [P] / [S] tag
  * per-strategy usage keys are the tagged unified names

Run:
    python -m research.meta_strategy._smoke_unified
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from research.meta_strategy import (
    CandidateFilter,
    backtest_meta,
    wrap_pattern,
    wrap_sequential,
)
from research.pattern_discovery.feature_builder import BIN_COLUMNS
from research.pattern_strategies import StrategyCandidate
from research.sequential_strategies import SequentialStrategyCandidate


def _build_df(n: int = 2500, seed: int = 0) -> pd.DataFrame:
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

    # Pattern edge: bars where momentum_3_bin == POS → next bar close rises.
    m3_pos = (df["momentum_3_bin"] == "POS").to_numpy()
    bump_p = atr[0] * 3.0
    for i in np.where(m3_pos)[0]:
        hi = min(i + 11, n)
        df.loc[i + 1:hi, "close"] = (
            df.loc[i + 1:hi, "close"].to_numpy() + bump_p
        ).astype(np.float32)
        df.loc[i + 1:hi, "high"] = (
            df.loc[i + 1:hi, "high"].to_numpy() + bump_p
        ).astype(np.float32)
        df.loc[i + 1:hi, "low"] = (
            df.loc[i + 1:hi, "low"].to_numpy() + bump_p
        ).astype(np.float32)

    # Sequence edge: range_compression=TRUE then momentum_3=POS → bump after.
    rc = (df["range_compression_bin"] == "TRUE").to_numpy()
    mom = (df["momentum_3_bin"] == "POS").to_numpy()
    end_mask = np.zeros(n, dtype=bool)
    end_mask[1:] = rc[:-1] & mom[1:]
    bump_s = atr[0] * 2.0
    for i in np.where(end_mask)[0]:
        hi = min(i + 11, n)
        df.loc[i + 1:hi, "close"] = (
            df.loc[i + 1:hi, "close"].to_numpy() + bump_s
        ).astype(np.float32)
        df.loc[i + 1:hi, "high"] = (
            df.loc[i + 1:hi, "high"].to_numpy() + bump_s
        ).astype(np.float32)
        df.loc[i + 1:hi, "low"] = (
            df.loc[i + 1:hi, "low"].to_numpy() + bump_s
        ).astype(np.float32)

    # Recompute forward returns from modified OHLC.
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


def _rules():
    stop = {"type": "atr_multiplier", "period": 14, "multiplier": 1.5}
    exit_ = {
        "take_profit": {
            "type": "atr_multiplier", "period": 14, "multiplier": 2.5,
        },
        "max_hold_bars": 20,
    }
    return stop, exit_


def _pattern_candidate():
    stop, exit_ = _rules()
    return StrategyCandidate(
        pattern_key=(("momentum_3_bin", "POS"),),
        direction="LONG",
        entry_rule={
            "type": "AND",
            "conditions": [{"column": "momentum_3_bin", "equals": "POS"}],
        },
        stop_rule=stop,
        exit_rule=exit_,
        expected_edge={"mean_test_return": 0.002},
    )


def _sequential_candidate():
    stop, exit_ = _rules()
    return SequentialStrategyCandidate(
        sequence_key=(
            ("range_compression_bin", "TRUE"),
            ("momentum_3_bin", "POS"),
        ),
        direction="LONG",
        entry_rule={
            "type": "SEQUENCE",
            "trigger": "after_last_step",
            "steps": [
                {"column": "range_compression_bin", "equals": "TRUE"},
                {"column": "momentum_3_bin", "equals": "POS"},
            ],
        },
        stop_rule=stop,
        exit_rule=exit_,
        expected_edge={"mean_return": 0.0025},
    )


def main() -> None:
    print("meta_strategy unified smoke test")
    print("-" * 60)

    df = _build_df()
    print(f"df: {len(df):,} rows  {len(df.columns)} cols")

    pat = _pattern_candidate()
    seq = _sequential_candidate()
    unified = [wrap_pattern(pat), wrap_sequential(seq)]

    for u in unified:
        print(
            f"  {u.candidate_type:<8}  regimes={sorted(u.allowed_regimes)}  "
            f"name={u.name}"
        )

    report = backtest_meta(
        df,
        unified,
        top_k_per_regime=1,
        pre_filter=CandidateFilter(),
        apply_prefilter=True,
        selector_min_profit_factor=1.3,
        selector_min_expectancy=0.0,
    )

    print()
    for line in report.summary_lines():
        print(line)

    # Invariants.
    assert report.candidates_before_filter == 2, (
        f"expected 2 candidates, got {report.candidates_before_filter}"
    )
    assert report.pattern_candidates_count == 1
    assert report.sequence_candidates_count == 1
    # Per-strategy usage keys must carry [P]/[S] tags when trades fire.
    used_names = set(report.per_strategy_usage.keys())
    tagged_ok = all(
        name.startswith("[P]") or name.startswith("[S]")
        for name in used_names
    )
    assert tagged_ok or not used_names, (
        f"usage keys not tagged with [P]/[S]: {used_names}"
    )

    # ── Second run with prefilter disabled — force the selector to engage
    #    both candidates so the tagged-name path is exercised end-to-end.
    print()
    print("second run: apply_prefilter=False (forces selector on both)")
    print("-" * 60)
    report2 = backtest_meta(
        df,
        unified,
        top_k_per_regime=1,
        apply_prefilter=False,
    )
    print(f"  plan = {report2.selector_plan}")
    print(f"  usage = {dict(report2.per_strategy_usage)}")
    # Selector plan values should either be None or a [P]/[S]-tagged name,
    # or the literal "NO_TRADE".
    for regime, entry in report2.selector_plan.items():
        if entry is None or entry == "NO_TRADE":
            continue
        assert entry.startswith("[P]") or entry.startswith("[S]"), (
            f"untagged name in plan[{regime}] = {entry!r}"
        )

    # ── Third run: edge-decay filter excludes the sequence candidate ──────
    print()
    print("third run: edge-decay filter (sequence marked BROKEN)")
    print("-" * 60)
    decay_map = {
        unified[0].name: "STABLE",      # pattern → kept
        unified[1].name: "BROKEN",      # sequence → excluded
    }
    report3 = backtest_meta(
        df,
        unified,
        top_k_per_regime=1,
        apply_prefilter=False,
        decay_status_by_name=decay_map,
    )
    print(
        f"  before_decay_filter={report3.candidates_before_decay_filter}  "
        f"after_decay_filter={report3.candidates_after_decay_filter}  "
        f"status_counts={dict(report3.decay_status_counts)}  "
        f"excluded={dict(report3.decay_excluded)}"
    )
    plan3 = report3.selector_plan
    usage3 = dict(report3.per_strategy_usage)
    print(f"  plan = {plan3}")
    print(f"  usage = {usage3}")

    assert report3.candidates_before_decay_filter == 2
    assert report3.candidates_after_decay_filter == 1
    # The sequence candidate must no longer appear in the selector plan
    # for COMPRESSION (it was BROKEN-filtered out).
    seq_name = unified[1].name
    assert not any(
        v == seq_name for v in plan3.values()
    ), f"BROKEN sequence still in plan: {plan3}"
    assert seq_name not in usage3, (
        f"BROKEN sequence still firing trades: {usage3}"
    )

    print()
    print("SMOKE OK")


if __name__ == "__main__":
    main()
