"""
Synthetic smoke test for the meta-strategy layer.

Builds a tiny compact matrix, crafts two candidates (a trend-follower and
a mean-reverter), and verifies:

  * regime classifier returns the expected distribution
  * selector.active_for() picks the correct candidate per regime
  * backtest_meta produces a report with per-regime breakdown and
    per-strategy usage counts

Run:
    python -m research.meta_strategy._smoke_test
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
    ALL_REGIMES,
    StrategySelector,
    allowed_regimes_for,
    backtest_meta,
    classify_rows,
)
from research.pattern_discovery.feature_builder import BIN_COLUMNS
from research.pattern_strategies import StrategyCandidate


def _build_df(n: int = 1000, seed: int = 0) -> pd.DataFrame:
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
    return df


def _make_candidates() -> list[StrategyCandidate]:
    trend_up = StrategyCandidate(
        pattern_key=(("momentum_3_bin", "POS"), ("momentum_5_bin", "POS")),
        direction="LONG",
        entry_rule={
            "type": "AND",
            "conditions": [
                {"column": "momentum_3_bin", "equals": "POS"},
                {"column": "momentum_5_bin", "equals": "POS"},
            ],
        },
        stop_rule={"type": "atr_multiplier", "period": 14, "multiplier": 1.5},
        exit_rule={
            "take_profit": {
                "type": "atr_multiplier", "period": 14, "multiplier": 2.5,
            },
            "max_hold_bars": 20,
        },
        expected_edge={"mean_test_return": 0.0025},
    )
    mean_reversion = StrategyCandidate(
        pattern_key=(("distance_to_vwap_bin", "LT_NEG_2SIG"),),
        direction="LONG",
        entry_rule={
            "type": "AND",
            "conditions": [
                {"column": "distance_to_vwap_bin", "equals": "LT_NEG_2SIG"},
            ],
        },
        stop_rule={"type": "atr_multiplier", "period": 14, "multiplier": 1.5},
        exit_rule={
            "take_profit": {
                "type": "atr_multiplier", "period": 14, "multiplier": 2.5,
            },
            "max_hold_bars": 20,
        },
        expected_edge={"mean_test_return": 0.0018},
    )
    compression = StrategyCandidate(
        pattern_key=(("range_compression_bin", "TRUE"),),
        direction="SHORT",
        entry_rule={
            "type": "AND",
            "conditions": [
                {"column": "range_compression_bin", "equals": "TRUE"},
            ],
        },
        stop_rule={"type": "atr_multiplier", "period": 14, "multiplier": 1.5},
        exit_rule={
            "take_profit": {
                "type": "atr_multiplier", "period": 14, "multiplier": 2.5,
            },
            "max_hold_bars": 20,
        },
        expected_edge={"mean_test_return": -0.0012},
    )
    return [trend_up, mean_reversion, compression]


def main() -> None:
    print("meta_strategy smoke test")
    print("-" * 60)

    df = _build_df()
    regimes = classify_rows(df)
    dist = {r: int((regimes == r).sum()) for r in ALL_REGIMES}
    print(f"regime distribution: {dist}")
    assert sum(dist.values()) == len(df)
    assert dist["TREND_UP"] > 0 and dist["TREND_DOWN"] > 0

    cands = _make_candidates()
    for c in cands:
        print(f"  candidate {c.name[:40]:<40}  allowed={allowed_regimes_for(c)}")

    sel = StrategySelector(cands)
    plan = sel.plan()
    print(f"  selector plan: {plan}")
    assert plan["TREND_UP"] is not None, "expected an active candidate for TREND_UP"
    assert plan["COMPRESSION"] is not None, "expected an active candidate for COMPRESSION"

    report = backtest_meta(df, cands)
    print()
    for line in report.summary_lines():
        print(line)

    # Invariants.
    total = report.global_report.total_trades
    assert total >= 0
    per_regime_total = sum(
        r.total_trades for r in report.per_regime.values()
    )
    assert per_regime_total == total, (
        f"per-regime trades ({per_regime_total}) != global ({total})"
    )
    assert report.candidates_before_filter == len(cands)
    assert report.candidates_after_filter <= report.candidates_before_filter

    # ── Pre-filter ablation: add obviously weak candidates ────────────────
    print()
    print("pre-filter ablation (adding 2 weak candidates)")
    print("-" * 60)
    from research.meta_strategy import CandidateFilter, prefilter_candidates

    weak_high_dd = StrategyCandidate(
        pattern_key=(("session_bin", "ASIA"),),
        direction="LONG",
        entry_rule={
            "type": "AND",
            "conditions": [{"column": "session_bin", "equals": "ASIA"}],
        },
        stop_rule={"type": "atr_multiplier", "period": 14, "multiplier": 3.0},
        exit_rule={
            "take_profit": {
                "type": "atr_multiplier", "period": 14, "multiplier": 0.3,
            },
            "max_hold_bars": 3,
        },
        expected_edge={"mean_test_return": 0.0001},
    )
    never_fires = StrategyCandidate(
        pattern_key=(("session_bin", "NONEXISTENT"),),
        direction="LONG",
        entry_rule={
            "type": "AND",
            "conditions": [{"column": "session_bin", "equals": "NONEXISTENT"}],
        },
        stop_rule={"type": "atr_multiplier", "period": 14, "multiplier": 1.5},
        exit_rule={
            "take_profit": {
                "type": "atr_multiplier", "period": 14, "multiplier": 2.5,
            },
            "max_hold_bars": 20,
        },
        expected_edge={"mean_test_return": 0.0010},
    )
    mixed_cands = list(cands) + [weak_high_dd, never_fires]

    filt = CandidateFilter()
    survivors, reports, reasons = prefilter_candidates(
        df, mixed_cands, filt=filt
    )
    print(f"  before = {len(mixed_cands)}  after = {len(survivors)}")
    for name, reason in reasons.items():
        rep = reports.get(name)
        t = rep.total_trades if rep else 0
        pf = (
            f"{rep.profit_factor:.2f}"
            if rep and rep.profit_factor is not None
            else "n/a"
        )
        dd = f"{rep.max_drawdown:+.4f}" if rep else "n/a"
        print(
            f"    [{reason:<12}] n={t:>4}  pf={pf:>5}  dd={dd:>8}  {name[:45]}"
        )
    assert weak_high_dd.name in reasons
    assert never_fires.name in reasons
    # Weak candidates must NOT be in survivors (either zero trades or
    # stats below thresholds).
    surviving_names = {c.name for c in survivors}
    assert never_fires.name not in surviving_names, (
        "never-firing candidate should not pass filter"
    )

    # ── NO_TRADE selector check ───────────────────────────────────────────
    print()
    print("selector NO_TRADE check (prefilter OFF, weak candidate present)")
    print("-" * 60)
    # Build candidates where the top one for COMPRESSION will FAIL quality.
    # We feed our standalone metrics so the selector can apply the gate.
    weak_compression = StrategyCandidate(
        pattern_key=(("range_compression_bin", "TRUE"),),
        direction="LONG",  # fights the synthetic edge → poor PF
        entry_rule={
            "type": "AND",
            "conditions": [{"column": "range_compression_bin", "equals": "TRUE"}],
        },
        stop_rule={"type": "atr_multiplier", "period": 14, "multiplier": 1.5},
        exit_rule={
            "take_profit": {
                "type": "atr_multiplier", "period": 14, "multiplier": 2.5,
            },
            "max_hold_bars": 20,
        },
        # Inflated edge so it ranks above the SHORT candidate within COMPRESSION.
        expected_edge={"mean_test_return": 0.005},
    )
    only_weak = [weak_compression]
    no_trade_report = backtest_meta(
        df,
        only_weak,
        apply_prefilter=False,                # let the weak one through
        selector_min_profit_factor=1.3,
        selector_min_expectancy=0.0,
    )
    plan_no_trade = no_trade_report.selector_plan
    print(f"  plan with weak-only candidates: {plan_no_trade}")
    # The weak candidate maps to COMPRESSION; selector should mark it
    # NO_TRADE because it has no metrics (apply_prefilter=False).
    # Without metrics, the v1 fallback is to trade — so let's instead
    # provide metrics by enabling prefilter and passing it through:
    # Permissive prefilter (lets weak survive) + impossible selector PF gate
    # → selector should emit NO_TRADE for COMPRESSION.
    from research.meta_strategy import CandidateFilter as _CF
    permissive = _CF(min_trades=1, min_profit_factor=0.0,
                     min_expectancy=-1.0, max_drawdown_abs=1.0)
    no_trade_report_2 = backtest_meta(
        df,
        only_weak,
        apply_prefilter=True,
        pre_filter=permissive,
        selector_min_profit_factor=10.0,      # impossible threshold
        selector_min_expectancy=0.0,
    )
    plan2 = no_trade_report_2.selector_plan
    print(f"  plan with impossible PF threshold: {plan2}")
    has_no_trade = any(v == "NO_TRADE" for v in plan2.values())
    assert has_no_trade, "expected at least one NO_TRADE entry under impossible PF threshold"
    # And the global trade count should be 0 because every regime is NO_TRADE.
    assert no_trade_report_2.global_report.total_trades == 0, (
        f"expected 0 global trades, got {no_trade_report_2.global_report.total_trades}"
    )

    print()
    print("SMOKE OK")


if __name__ == "__main__":
    main()
