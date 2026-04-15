"""
research.pattern_strategies.backtest_pattern_strategy
──────────────────────────────────────────────────────
Bar-replay backtest for a single StrategyCandidate.

Replay model
────────────
For each row i where the candidate's entry conditions evaluate True,
open a position at the OPEN of bar i+1.  Intra-bar stop / take-profit
checks use high/low of subsequent bars.  Max hold caps the position to
`max_hold_bars` after entry; on timeout the position closes at the
bar's close.

Single position at a time — a new entry is only considered after the
prior position has closed (no overlap, no pyramiding).  This is a
deliberate simplification: the discovery engine already forecasts
one bar ahead, so overlap rarely improves statistics and adds noise.

Transaction costs
─────────────────
The ExecutionModel is already baked into `forward_return_10` at
discovery time, so the discovered expected_edge is net of slippage.
For the EXECUTABLE backtest here we use raw OHLC (no slippage),
because the goal is to surface the candidate's raw PnL profile —
production execution cost is added back once the candidate enters
the live aion.strategies pipeline.

Metrics
───────
total_trades, winrate, avg_return, expectancy (per-trade E[R]),
max_drawdown (peak-to-trough on cumulative return), profit_factor
(Σwins / |Σlosses|).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from research.pattern_strategies.strategy_candidate import StrategyCandidate

if TYPE_CHECKING:
    import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Trade:
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    direction: str
    stop_price: float
    tp_price: float
    return_pct: float
    exit_reason: str  # "STOP" | "TP" | "TIMEOUT"


@dataclass
class BacktestReport:
    candidate_name: str
    direction: str
    total_trades: int = 0
    winrate: float = 0.0
    avg_return: float = 0.0
    expectancy: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float | None = None
    total_return: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    trades: list[Trade] = field(default_factory=list)

    def summary_line(self) -> str:
        pf = f"{self.profit_factor:.2f}" if self.profit_factor is not None else "  n/a"
        return (
            f"n={self.total_trades:>4}  "
            f"wr={self.winrate * 100:5.1f}%  "
            f"avg={self.avg_return:+.5f}  "
            f"E={self.expectancy:+.5f}  "
            f"pf={pf}  "
            f"dd={self.max_drawdown:+.4f}  "
            f"[{self.direction}] {self.candidate_name}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry condition mask
# ─────────────────────────────────────────────────────────────────────────────


def _entry_mask(df: "pd.DataFrame", candidate: StrategyCandidate) -> np.ndarray:
    """Boolean array of rows where the AND'd entry conditions fire."""
    n = len(df)
    mask = np.ones(n, dtype=bool)
    for col, val in candidate.pattern_key:
        if col not in df.columns:
            return np.zeros(n, dtype=bool)
        series = df[col]
        if series.dtype.name == "category":
            codes = series.cat.codes.to_numpy(copy=False)
            cats = {str(v): i for i, v in enumerate(series.cat.categories)}
            code = cats.get(val)
            if code is None:
                return np.zeros(n, dtype=bool)
            mask &= codes == code
        else:
            mask &= (series.to_numpy(copy=False) == val)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Replay
# ─────────────────────────────────────────────────────────────────────────────


def _rule_multiplier(rule: dict[str, Any], key: str = "multiplier") -> float:
    return float(rule.get(key, 0.0))


def backtest_candidate(
    df: "pd.DataFrame",
    candidate: StrategyCandidate,
) -> BacktestReport:
    """
    Replay `candidate` over `df` (compact matrix with OHLC + ATR columns).

    The df must contain: open, high, low, close, atr_14, plus the
    categorical bin columns referenced by candidate.pattern_key.
    """
    required = {"open", "high", "low", "close", "atr_14"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    stop_mult = _rule_multiplier(candidate.stop_rule)
    tp_rule = candidate.exit_rule.get("take_profit", {})
    tp_mult = _rule_multiplier(tp_rule)
    max_hold = int(candidate.exit_rule.get("max_hold_bars", 20))
    direction = candidate.direction

    entries = _entry_mask(df, candidate)
    opens = df["open"].to_numpy(dtype=np.float64, copy=False)
    highs = df["high"].to_numpy(dtype=np.float64, copy=False)
    lows = df["low"].to_numpy(dtype=np.float64, copy=False)
    closes = df["close"].to_numpy(dtype=np.float64, copy=False)
    atrs = df["atr_14"].to_numpy(dtype=np.float64, copy=False)

    n = len(df)
    trades: list[Trade] = []
    i = 0
    long_ = direction == "LONG"

    while i < n - 1:
        if not entries[i]:
            i += 1
            continue

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
            # Stop wins ties (conservative).
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

        trades.append(
            Trade(
                entry_bar=entry_bar,
                exit_bar=exit_bar,
                entry_price=float(entry_price),
                exit_price=float(exit_price),
                direction=direction,
                stop_price=float(stop),
                tp_price=float(tp),
                return_pct=float(ret),
                exit_reason=exit_reason,
            )
        )
        # No overlapping positions: resume scanning after exit.
        i = exit_bar + 1

    return _build_report(candidate, trades)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def _build_report(
    candidate: StrategyCandidate,
    trades: list[Trade],
) -> BacktestReport:
    report = BacktestReport(
        candidate_name=candidate.name,
        direction=candidate.direction,
        trades=trades,
    )
    n = len(trades)
    if n == 0:
        return report

    rets = np.array([t.return_pct for t in trades], dtype=np.float64)
    wins = rets[rets > 0]
    losses = rets[rets < 0]

    report.total_trades = n
    report.winrate = float(len(wins) / n)
    report.avg_return = float(rets.mean())
    report.total_return = float(rets.sum())
    report.avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    report.avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    # Expectancy per trade = E[R].  With wins/losses it's algebraically
    # identical to the plain mean when both sides are present.
    report.expectancy = report.avg_return

    # Profit factor.
    sum_wins = float(wins.sum()) if len(wins) else 0.0
    sum_losses_abs = float(-losses.sum()) if len(losses) else 0.0
    report.profit_factor = (
        sum_wins / sum_losses_abs if sum_losses_abs > 0 else None
    )

    # Max drawdown on the equity curve (additive — returns are small).
    equity = np.cumsum(rets)
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity - running_max
    report.max_drawdown = float(drawdowns.min()) if len(drawdowns) else 0.0

    return report


__all__ = [
    "Trade",
    "BacktestReport",
    "backtest_candidate",
]
