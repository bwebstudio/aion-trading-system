"""
research.edge_decay.rolling_metrics
────────────────────────────────────
Trade-by-trade rolling performance windows.

Given the trade list from a backtest report, compute rolling metrics
(window_size trades wide) across the timeline:

    rolling winrate
    rolling expectancy        (arithmetic mean of per-trade returns)
    rolling profit_factor     (Σ wins / |Σ losses|)
    rolling max_drawdown      (peak-to-trough on cumulative returns
                               WITHIN each window — self-contained)

For each metric we produce the latest value and a slope signal that
downstream classification uses to decide IMPROVING / STABLE / DECAYING.

No bar-level approximation — all metrics come straight from the trade
returns that the backtest actually closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WindowMetrics:
    """
    Rolling-window metrics for one window size.

    * `latest_*` is the value over the final window_size trades.
    * `series_*` is the rolling series: metric computed at every
      trade index i >= window_size - 1 using trades[i-w+1..i].
      Length: max(0, n_trades - window_size + 1).
    * `slope_*` is the linear-regression slope of `series_*` vs.
      its integer index (slope per window step).
    * `rel_change_*` is `(series[-1] - series[0]) / max(|series[0]|, eps)`
      — more robust than raw slope for short series.
    """

    window_size: int
    n_trades: int
    valid: bool                       # True iff n_trades >= window_size
    latest_winrate: float | None = None
    latest_expectancy: float | None = None
    latest_profit_factor: float | None = None
    latest_max_drawdown: float | None = None
    series_expectancy: list[float] = field(default_factory=list)
    series_profit_factor: list[float] = field(default_factory=list)
    series_max_drawdown: list[float] = field(default_factory=list)
    slope_expectancy: float = 0.0
    slope_profit_factor: float = 0.0
    rel_change_expectancy: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Core math
# ─────────────────────────────────────────────────────────────────────────────


def _window_profit_factor(rets: np.ndarray) -> float | None:
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    sum_w = float(wins.sum()) if len(wins) else 0.0
    sum_l = float(-losses.sum()) if len(losses) else 0.0
    if sum_l <= 0:
        return None
    return sum_w / sum_l


def _window_max_drawdown(rets: np.ndarray) -> float:
    """Max drawdown of the cumulative-sum equity curve of one window."""
    if len(rets) == 0:
        return 0.0
    equity = np.cumsum(rets)
    running_max = np.maximum.accumulate(equity)
    dd = equity - running_max
    return float(dd.min()) if len(dd) else 0.0


def _slope(series: list[float]) -> float:
    """Plain OLS slope of `series` vs. its integer index."""
    n = len(series)
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    y = np.asarray(series, dtype=np.float64)
    if not np.isfinite(y).all():
        mask = np.isfinite(y)
        if mask.sum() < 3:
            return 0.0
        x = x[mask]
        y = y[mask]
    # np.polyfit returns highest-order first.
    try:
        m, _b = np.polyfit(x, y, 1)
    except np.linalg.LinAlgError:
        return 0.0
    return float(m)


def _rel_change(series: list[float]) -> float:
    if len(series) < 2:
        return 0.0
    first = series[0]
    last = series[-1]
    denom = max(abs(first), 1e-9)
    return float((last - first) / denom)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def compute_rolling_metrics(
    returns: Iterable[float],
    window_size: int,
) -> WindowMetrics:
    """
    Build rolling-window metrics over a list of per-trade returns.

    Parameters
    ----------
    returns:
        Per-trade `return_pct` values in chronological order (entry →
        exit direction already baked in).
    window_size:
        Trades per window.
    """
    rets = np.asarray(list(returns), dtype=np.float64)
    n = len(rets)

    if n < window_size or window_size <= 0:
        return WindowMetrics(
            window_size=window_size,
            n_trades=n,
            valid=False,
        )

    n_windows = n - window_size + 1
    series_exp: list[float] = []
    series_pf: list[float] = []
    series_dd: list[float] = []

    for i in range(n_windows):
        w = rets[i: i + window_size]
        series_exp.append(float(w.mean()))
        pf = _window_profit_factor(w)
        # Represent "no losses" as a large-but-finite PF so the series
        # is plottable / slope-able; downstream caps via clamp.
        series_pf.append(pf if pf is not None else 999.0)
        series_dd.append(_window_max_drawdown(w))

    last_w = rets[-window_size:]
    last_winrate = float((last_w > 0).mean())
    last_exp = float(last_w.mean())
    last_pf = _window_profit_factor(last_w)
    last_dd = _window_max_drawdown(last_w)

    return WindowMetrics(
        window_size=window_size,
        n_trades=n,
        valid=True,
        latest_winrate=last_winrate,
        latest_expectancy=last_exp,
        latest_profit_factor=last_pf,
        latest_max_drawdown=last_dd,
        series_expectancy=series_exp,
        series_profit_factor=series_pf,
        series_max_drawdown=series_dd,
        slope_expectancy=_slope(series_exp),
        slope_profit_factor=_slope(series_pf),
        rel_change_expectancy=_rel_change(series_exp),
    )


def compute_windows(
    returns: Iterable[float],
    window_sizes: Iterable[int] = (50, 100, 200),
) -> dict[int, WindowMetrics]:
    """Convenience: compute one WindowMetrics per size, keyed by size."""
    rets = list(returns)
    return {w: compute_rolling_metrics(rets, w) for w in window_sizes}


__all__ = [
    "WindowMetrics",
    "compute_rolling_metrics",
    "compute_windows",
]
