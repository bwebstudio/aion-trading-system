"""
research.meta_strategy.meta_backtest
─────────────────────────────────────
Bar-replay backtest of the meta-strategy layer.

At each row:
    1. `regime_classifier.classify_rows` labels the row.
    2. `StrategySelector.active_for(regime)` returns the chosen candidate
       for that regime (or None).
    3. If the chosen candidate's entry conditions also match this row
       AND we are flat, open a position at the NEXT bar's open.
    4. Manage the position with ATR stop / ATR take-profit / max-hold
       timeout — same rules as backtest_pattern_strategy.py.

Pre-filter stage
────────────────
Before the selector sees any candidate, every candidate is individually
backtested on the same df.  Survivors must satisfy:

    total_trades   >= min_trades         (default 50)
    profit_factor  >= min_profit_factor  (default 1.3)
    expectancy     >  0
    |max_drawdown| <  max_drawdown_abs   (default 0.05)

Only survivors are handed to the StrategySelector.  This prevents the
meta layer from anchoring on weak, high-variance candidates.

Aggregate report
────────────────
Global metrics (total trades, winrate, expectancy, profit factor, max
drawdown) plus:
    * per_regime: metrics broken down by the regime at entry
    * per_strategy_usage: trade count per candidate name
    * regime_distribution: how many bars spent in each regime
    * candidates_before_filter / candidates_after_filter
    * prefilter_reports: per-candidate standalone metrics (diagnostic)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from research.meta_strategy.regime_classifier import ALL_REGIMES, classify_rows
from research.meta_strategy.strategy_selector import NO_TRADE, StrategySelector
from research.pattern_strategies.backtest_pattern_strategy import (
    BacktestReport,
    Trade,
    _build_report,
    _entry_mask,
    _rule_multiplier,
    backtest_candidate,
)
from research.pattern_strategies.strategy_candidate import StrategyCandidate


# ─────────────────────────────────────────────────────────────────────────────
# Pre-filter
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CandidateFilter:
    """Filter thresholds applied to each candidate's standalone backtest."""

    min_trades: int = 50
    min_profit_factor: float = 1.3
    min_expectancy: float = 0.0          # strictly > 0 required
    max_drawdown_abs: float = 0.05        # |dd| must be strictly less

    def passes(self, report: BacktestReport) -> tuple[bool, str]:
        """Return (passes, reason_if_failed)."""
        if report.total_trades < self.min_trades:
            return False, f"trades<{self.min_trades}"
        if report.expectancy <= self.min_expectancy:
            return False, "expectancy<=0"
        pf = report.profit_factor
        if pf is None or pf < self.min_profit_factor:
            return False, f"pf<{self.min_profit_factor}"
        if abs(report.max_drawdown) >= self.max_drawdown_abs:
            return False, f"|dd|>={self.max_drawdown_abs}"
        return True, "ok"


def prefilter_candidates(
    df: "pd.DataFrame",
    candidates: list[StrategyCandidate],
    *,
    filt: CandidateFilter | None = None,
) -> tuple[list[StrategyCandidate], dict[str, BacktestReport], dict[str, str]]:
    """
    Standalone-backtest each candidate; return
    (survivors, reports_by_name, rejection_reasons_by_name).

    Candidates that raise during replay (e.g. due to bin values absent
    from the df) are treated as failing with reason "error".
    """
    filt = filt or CandidateFilter()
    survivors: list[StrategyCandidate] = []
    reports: dict[str, BacktestReport] = {}
    reasons: dict[str, str] = {}
    for cand in candidates:
        try:
            rep = backtest_candidate(df, cand)
        except Exception as exc:  # noqa: BLE001
            reasons[cand.name] = f"error: {exc}"
            continue
        reports[cand.name] = rep
        ok, reason = filt.passes(rep)
        reasons[cand.name] = reason
        if ok:
            survivors.append(cand)
    return survivors, reports, reasons

if TYPE_CHECKING:
    import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MetaBacktestReport:
    """Composite report for the meta-strategy backtest."""

    global_report: BacktestReport
    per_regime: dict[str, BacktestReport] = field(default_factory=dict)
    per_strategy_usage: dict[str, int] = field(default_factory=dict)
    regime_distribution: dict[str, int] = field(default_factory=dict)
    selector_plan: dict[str, str | None] = field(default_factory=dict)
    # Pre-filter diagnostics
    candidates_before_filter: int = 0
    candidates_after_filter: int = 0
    prefilter_reports: dict[str, BacktestReport] = field(default_factory=dict)
    prefilter_rejections: dict[str, str] = field(default_factory=dict)

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        lines.append(
            f"candidates_before_filter : {self.candidates_before_filter}"
        )
        lines.append(
            f"candidates_after_filter  : {self.candidates_after_filter}"
        )
        lines.append("")
        g = self.global_report
        lines.append(f"GLOBAL   {g.summary_line()}")
        lines.append("")
        lines.append("per-regime:")
        for regime in ALL_REGIMES:
            r = self.per_regime.get(regime)
            bars = self.regime_distribution.get(regime, 0)
            raw_plan = self.selector_plan.get(regime)
            if raw_plan is None:
                plan = "(none — no candidate maps to regime)"
            elif raw_plan == "NO_TRADE":
                plan = "NO_TRADE (top candidate fails quality gate)"
            else:
                plan = raw_plan
            if r is None or r.total_trades == 0:
                lines.append(
                    f"  {regime:<12} bars={bars:>5,}  trades=0   plan={plan}"
                )
            else:
                lines.append(
                    f"  {regime:<12} bars={bars:>5,}  "
                    f"n={r.total_trades:>4}  "
                    f"wr={r.winrate * 100:5.1f}%  "
                    f"avg={r.avg_return:+.5f}  "
                    f"pf={r.profit_factor if r.profit_factor is not None else 'n/a':>5}  "
                    f"dd={r.max_drawdown:+.4f}  "
                    f"plan={plan}"
                )
        lines.append("")
        lines.append("per-strategy usage:")
        if not self.per_strategy_usage:
            lines.append("  (none)")
        else:
            for name, n in sorted(
                self.per_strategy_usage.items(), key=lambda x: -x[1]
            ):
                lines.append(f"  {n:>5}× {name}")
        return lines


# ─────────────────────────────────────────────────────────────────────────────
# Backtest engine
# ─────────────────────────────────────────────────────────────────────────────


def _pseudo_candidate_with_trades(
    name: str, direction: str, trades: list[Trade]
) -> BacktestReport:
    """Wrap a trade list into a BacktestReport without re-running replay."""
    dummy = StrategyCandidate(
        pattern_key=(("_", "_"),),
        direction=direction,
        entry_rule={},
        stop_rule={},
        exit_rule={},
        expected_edge={},
        name=name,
    )
    return _build_report(dummy, trades)


def backtest_meta(
    df: "pd.DataFrame",
    candidates: list[StrategyCandidate],
    *,
    top_k_per_regime: int = 1,
    pre_filter: CandidateFilter | None = None,
    apply_prefilter: bool = True,
    selector_min_profit_factor: float = 1.3,
    selector_min_expectancy: float = 0.0,
) -> MetaBacktestReport:
    """
    Run the meta-strategy backtest.

    Parameters
    ----------
    df:
        Compact matrix with OHLC, ATR, and categorical bin columns.
    candidates:
        StrategyCandidate list (typically loaded from
        research/output/strategy_candidates.json).
    top_k_per_regime:
        Number of candidates kept per regime.  Only top-1 is used for
        per-row activation, but the shortlist is exposed via the selector.
    pre_filter:
        Thresholds applied per-candidate before the selector sees them.
        Defaults to `CandidateFilter()` (min_trades=50, pf>=1.3,
        expectancy>0, |dd|<0.05).
    apply_prefilter:
        If False, skip the pre-filter stage and pass all candidates
        straight to the selector (useful for ablation comparisons).
    """
    required = {"open", "high", "low", "close", "atr_14"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    n = len(df)
    before_count = len(candidates)

    # ── Pre-filter stage ────────────────────────────────────────────────────
    if apply_prefilter:
        filt = pre_filter or CandidateFilter()
        survivors, prefilter_reports, prefilter_reasons = prefilter_candidates(
            df, candidates, filt=filt
        )
    else:
        survivors = list(candidates)
        prefilter_reports = {}
        prefilter_reasons = {}

    after_count = len(survivors)

    if n == 0 or not survivors:
        return MetaBacktestReport(
            global_report=_pseudo_candidate_with_trades("META", "LONG", []),
            selector_plan={},
            candidates_before_filter=before_count,
            candidates_after_filter=after_count,
            prefilter_reports=prefilter_reports,
            prefilter_rejections=prefilter_reasons,
        )

    # After the filter, `candidates` refers to survivors only.
    candidates = survivors

    # ── Regime stream ───────────────────────────────────────────────────────
    regimes = classify_rows(df)
    regime_distribution: dict[str, int] = {r: 0 for r in ALL_REGIMES}
    for r in ALL_REGIMES:
        regime_distribution[r] = int((regimes == r).sum())

    # ── Selector — top-1 per regime, with NO_TRADE quality gate ───────────
    selector = StrategySelector(
        candidates,
        top_k_per_regime=top_k_per_regime,
        candidate_metrics=prefilter_reports if prefilter_reports else None,
        min_profit_factor=selector_min_profit_factor,
        min_expectancy=selector_min_expectancy,
    )
    plan = selector.plan()

    # ── Precompute per-candidate entry masks (O(N) each, once) ─────────────
    # Only candidates that ACTIVELY trade (not NO_TRADE / None) need a mask.
    active_cands: dict[str, StrategyCandidate] = {}
    for regime in ALL_REGIMES:
        cand = selector.active_for(regime)
        if cand is None or cand is NO_TRADE:
            continue
        active_cands[cand.name] = cand
    entry_masks: dict[str, np.ndarray] = {
        name: _entry_mask(df, cand) for name, cand in active_cands.items()
    }

    # ── Per-regime active candidate name (None / NO_TRADE / name) ─────────
    regime_to_cand_name: dict[str, str | None] = {}
    for regime in ALL_REGIMES:
        cand = selector.active_for(regime)
        if cand is None or cand is NO_TRADE:
            regime_to_cand_name[regime] = None  # both treated as skip
        else:
            regime_to_cand_name[regime] = cand.name

    # ── Numpy views ────────────────────────────────────────────────────────
    opens = df["open"].to_numpy(dtype=np.float64, copy=False)
    highs = df["high"].to_numpy(dtype=np.float64, copy=False)
    lows = df["low"].to_numpy(dtype=np.float64, copy=False)
    closes = df["close"].to_numpy(dtype=np.float64, copy=False)
    atrs = df["atr_14"].to_numpy(dtype=np.float64, copy=False)

    all_trades: list[Trade] = []
    trades_by_regime: dict[str, list[Trade]] = {r: [] for r in ALL_REGIMES}
    trades_by_strategy: dict[str, list[Trade]] = {
        name: [] for name in active_cands
    }
    usage_count: dict[str, int] = {name: 0 for name in active_cands}

    i = 0
    while i < n - 1:
        regime = regimes[i]
        cand_name = regime_to_cand_name.get(regime)
        if cand_name is None:
            i += 1
            continue

        emask = entry_masks[cand_name]
        if not emask[i]:
            i += 1
            continue

        cand = active_cands[cand_name]
        stop_mult = _rule_multiplier(cand.stop_rule)
        tp_rule = cand.exit_rule.get("take_profit", {})
        tp_mult = _rule_multiplier(tp_rule)
        max_hold = int(cand.exit_rule.get("max_hold_bars", 20))
        long_ = cand.direction == "LONG"

        entry_bar = i + 1
        if entry_bar >= n:
            break
        entry_price = opens[entry_bar]
        atr = atrs[i]
        if not np.isfinite(atr) or atr <= 0 or not np.isfinite(entry_price):
            i += 1
            continue

        if long_:
            stop = entry_price - stop_mult * atr
            tp = entry_price + tp_mult * atr
        else:
            stop = entry_price + stop_mult * atr
            tp = entry_price - tp_mult * atr

        exit_bar: int | None = None
        exit_price: float | None = None
        exit_reason = "TIMEOUT"
        last_allowed = min(entry_bar + max_hold - 1, n - 1)

        for j in range(entry_bar, last_allowed + 1):
            hi, lo = highs[j], lows[j]
            if long_:
                stop_hit = lo <= stop
                tp_hit = hi >= tp
            else:
                stop_hit = hi >= stop
                tp_hit = lo <= tp
            if stop_hit:
                exit_bar, exit_price, exit_reason = j, stop, "STOP"
                break
            if tp_hit:
                exit_bar, exit_price, exit_reason = j, tp, "TP"
                break

        if exit_bar is None:
            exit_bar = last_allowed
            exit_price = closes[exit_bar]
            exit_reason = "TIMEOUT"

        if long_:
            ret = (exit_price - entry_price) / entry_price
        else:
            ret = (entry_price - exit_price) / entry_price

        trade = Trade(
            entry_bar=entry_bar,
            exit_bar=exit_bar,
            entry_price=float(entry_price),
            exit_price=float(exit_price),
            direction=cand.direction,
            stop_price=float(stop),
            tp_price=float(tp),
            return_pct=float(ret),
            exit_reason=exit_reason,
        )
        all_trades.append(trade)
        trades_by_regime[regime].append(trade)
        trades_by_strategy[cand_name].append(trade)
        usage_count[cand_name] = usage_count.get(cand_name, 0) + 1

        i = exit_bar + 1

    # ── Aggregate ──────────────────────────────────────────────────────────
    global_report = _pseudo_candidate_with_trades("META", "MIXED", all_trades)

    per_regime: dict[str, BacktestReport] = {}
    for regime, trades in trades_by_regime.items():
        per_regime[regime] = _pseudo_candidate_with_trades(
            f"META:{regime}", "MIXED", trades
        )

    return MetaBacktestReport(
        global_report=global_report,
        per_regime=per_regime,
        per_strategy_usage=usage_count,
        regime_distribution=regime_distribution,
        selector_plan=plan,
        candidates_before_filter=before_count,
        candidates_after_filter=after_count,
        prefilter_reports=prefilter_reports,
        prefilter_rejections=prefilter_reasons,
    )


__all__ = [
    "MetaBacktestReport",
    "backtest_meta",
]
