"""
research.edge_decay.decay_report
─────────────────────────────────
Classify each candidate's rolling-metric timeline into one of four
decay statuses and compute a continuous `decay_score ∈ [-1, +1]`.

Classification rules
────────────────────
Given the primary window's metrics:

    BROKEN     if latest_pf < 1.0
               or latest_expectancy <= 0
               or |latest_max_drawdown| > broken_dd_threshold

    DECAYING   if rel_change_expectancy < -decaying_threshold
               or the drawdown series is monotonically worsening
               (latest is the most negative window-dd seen)

    IMPROVING  if latest_pf >= 1.3
               and rel_change_expectancy > improving_threshold

    STABLE     otherwise (latest is acceptable, trend is flat)

A candidate without enough trades for any window size is tagged
INSUFFICIENT_DATA.

decay_score
───────────
Continuous signal combining:

    pf_signal    = clip((latest_pf - 1.0) / 0.5,         -1, +1)
    exp_signal   = tanh(latest_expectancy * exp_scale)
    trend_signal = clip(rel_change_expectancy / 0.5,     -1, +1)
    dd_signal    = clip(1 - |latest_dd| / broken_dd_threshold, -1, +1)

    decay_score  = clip( 0.30 * pf_signal
                       + 0.30 * exp_signal
                       + 0.25 * trend_signal
                       + 0.15 * dd_signal, -1, +1 )

Values near +1 indicate a strong, improving edge; values near -1
indicate a broken edge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import math

from research.edge_decay.rolling_metrics import (
    WindowMetrics,
    compute_windows,
)


# ── Status constants ────────────────────────────────────────────────────────
STATUS_STABLE = "STABLE"
STATUS_IMPROVING = "IMPROVING"
STATUS_DECAYING = "DECAYING"
STATUS_BROKEN = "BROKEN"
STATUS_INSUFFICIENT = "INSUFFICIENT_DATA"


# ── Default thresholds ──────────────────────────────────────────────────────
BROKEN_DD_THRESHOLD = 0.10           # |window max dd| above this = BROKEN
IMPROVING_REL_CHANGE = 0.10           # +10% series growth → IMPROVING
DECAYING_REL_CHANGE = 0.10            # -10% series shrink → DECAYING
STABLE_PF_MIN = 1.3
EXP_SIGNAL_SCALE = 5000.0             # expectancy ≈ 0.0002 → tanh(1)


# ─────────────────────────────────────────────────────────────────────────────
# Report dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DecayReport:
    """
    Single-candidate decay analysis.

    Fields
    ──────
    candidate_name / candidate_type / asset : identity
    total_trades                            : full trade count
    windows                                 : {window_size -> WindowMetrics}
    primary_window                          : largest window with enough trades
                                              (used for latest_* + status)
    latest_*                                : metric values at end of timeline
    rel_change_expectancy                   : series movement from start to end
    status                                  : one of STATUS_*
    decay_score                             : continuous ∈ [-1, +1]
    """

    candidate_name: str
    candidate_type: str
    asset: str
    total_trades: int
    windows: dict[int, WindowMetrics]
    primary_window: int | None
    latest_winrate: float | None
    latest_expectancy: float | None
    latest_profit_factor: float | None
    latest_max_drawdown: float | None
    rel_change_expectancy: float
    slope_expectancy: float
    status: str
    decay_score: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "candidate_name": self.candidate_name,
            "candidate_type": self.candidate_type,
            "asset": self.asset,
            "total_trades": self.total_trades,
            "primary_window": self.primary_window,
            "latest_winrate": self.latest_winrate,
            "latest_expectancy": self.latest_expectancy,
            "latest_profit_factor": self.latest_profit_factor,
            "latest_max_drawdown": self.latest_max_drawdown,
            "rel_change_expectancy": self.rel_change_expectancy,
            "slope_expectancy": self.slope_expectancy,
            "status": self.status,
            "decay_score": self.decay_score,
            "windows": {
                str(k): _window_to_dict(v) for k, v in self.windows.items()
            },
        }

    def summary_line(self) -> str:
        pf = (
            f"{self.latest_profit_factor:5.2f}"
            if self.latest_profit_factor is not None
            else "  n/a"
        )
        exp = (
            f"{self.latest_expectancy:+.5f}"
            if self.latest_expectancy is not None
            else "     n/a"
        )
        return (
            f"{self.candidate_type:<8}  "
            f"{self.candidate_name[:48]:<48}  "
            f"trades={self.total_trades:>4}  "
            f"pf={pf}  "
            f"exp={exp}  "
            f"score={self.decay_score:+.2f}  "
            f"{self.status}"
        )


def _window_to_dict(w: WindowMetrics) -> dict[str, Any]:
    return {
        "window_size": w.window_size,
        "n_trades": w.n_trades,
        "valid": w.valid,
        "latest_winrate": w.latest_winrate,
        "latest_expectancy": w.latest_expectancy,
        "latest_profit_factor": w.latest_profit_factor,
        "latest_max_drawdown": w.latest_max_drawdown,
        "slope_expectancy": w.slope_expectancy,
        "slope_profit_factor": w.slope_profit_factor,
        "rel_change_expectancy": w.rel_change_expectancy,
        "n_windows": len(w.series_expectancy),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _pick_primary_window(
    windows: dict[int, WindowMetrics],
) -> WindowMetrics | None:
    """Prefer the LARGEST valid window (most statistically robust)."""
    valid = [w for w in windows.values() if w.valid]
    if not valid:
        return None
    return max(valid, key=lambda w: w.window_size)


def _dd_worsening(series: list[float]) -> bool:
    """
    True if the drawdown series is monotonically worsening — i.e. the
    latest window has the deepest drawdown of the entire timeline.
    `series` values are negative-or-zero window drawdowns.
    """
    if len(series) < 3:
        return False
    return series[-1] == min(series)


def _classify(
    w: WindowMetrics,
    *,
    broken_dd_threshold: float,
    improving_threshold: float,
    decaying_threshold: float,
    stable_pf_min: float,
) -> str:
    if not w.valid:
        return STATUS_INSUFFICIENT

    pf = w.latest_profit_factor
    exp = w.latest_expectancy
    dd = w.latest_max_drawdown or 0.0

    # ── BROKEN ──────────────────────────────────────────────────────────────
    if pf is not None and pf < 1.0:
        return STATUS_BROKEN
    if exp is not None and exp <= 0.0:
        return STATUS_BROKEN
    if abs(dd) > broken_dd_threshold:
        return STATUS_BROKEN

    # ── DECAYING ────────────────────────────────────────────────────────────
    # Either the expectancy series has shrunk materially, or the window
    # drawdown is monotonically worsening.
    if w.rel_change_expectancy < -decaying_threshold:
        return STATUS_DECAYING
    if _dd_worsening(w.series_max_drawdown):
        # Only mark as DECAYING if PF is weakening too.
        if pf is None or pf < stable_pf_min:
            return STATUS_DECAYING

    # ── IMPROVING ───────────────────────────────────────────────────────────
    if (
        pf is not None and pf >= stable_pf_min
        and w.rel_change_expectancy > improving_threshold
    ):
        return STATUS_IMPROVING

    # ── STABLE ──────────────────────────────────────────────────────────────
    return STATUS_STABLE


def _decay_score(
    w: WindowMetrics,
    *,
    broken_dd_threshold: float,
) -> float:
    if not w.valid:
        return 0.0

    pf = w.latest_profit_factor if w.latest_profit_factor is not None else 1.0
    exp = w.latest_expectancy if w.latest_expectancy is not None else 0.0
    dd = w.latest_max_drawdown or 0.0

    pf_signal = _clip((pf - 1.0) / 0.5, -1.0, 1.0)
    exp_signal = math.tanh(exp * EXP_SIGNAL_SCALE)
    trend_signal = _clip(w.rel_change_expectancy / 0.5, -1.0, 1.0)
    dd_signal = _clip(1.0 - abs(dd) / max(broken_dd_threshold, 1e-6),
                      -1.0, 1.0)

    score = (
        0.30 * pf_signal
        + 0.30 * exp_signal
        + 0.25 * trend_signal
        + 0.15 * dd_signal
    )
    return _clip(score, -1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def build_report(
    *,
    candidate_name: str,
    candidate_type: str,
    asset: str,
    trade_returns: list[float],
    window_sizes: tuple[int, ...] = (50, 100, 200),
    broken_dd_threshold: float = BROKEN_DD_THRESHOLD,
    improving_threshold: float = IMPROVING_REL_CHANGE,
    decaying_threshold: float = DECAYING_REL_CHANGE,
    stable_pf_min: float = STABLE_PF_MIN,
) -> DecayReport:
    """
    Compute rolling metrics and produce a DecayReport for one candidate.
    """
    windows = compute_windows(trade_returns, window_sizes)
    primary = _pick_primary_window(windows)

    total_trades = len(trade_returns)

    if primary is None:
        return DecayReport(
            candidate_name=candidate_name,
            candidate_type=candidate_type,
            asset=asset,
            total_trades=total_trades,
            windows=windows,
            primary_window=None,
            latest_winrate=None,
            latest_expectancy=None,
            latest_profit_factor=None,
            latest_max_drawdown=None,
            rel_change_expectancy=0.0,
            slope_expectancy=0.0,
            status=STATUS_INSUFFICIENT,
            decay_score=0.0,
        )

    status = _classify(
        primary,
        broken_dd_threshold=broken_dd_threshold,
        improving_threshold=improving_threshold,
        decaying_threshold=decaying_threshold,
        stable_pf_min=stable_pf_min,
    )
    score = _decay_score(primary, broken_dd_threshold=broken_dd_threshold)

    return DecayReport(
        candidate_name=candidate_name,
        candidate_type=candidate_type,
        asset=asset,
        total_trades=total_trades,
        windows=windows,
        primary_window=primary.window_size,
        latest_winrate=primary.latest_winrate,
        latest_expectancy=primary.latest_expectancy,
        latest_profit_factor=primary.latest_profit_factor,
        latest_max_drawdown=primary.latest_max_drawdown,
        rel_change_expectancy=primary.rel_change_expectancy,
        slope_expectancy=primary.slope_expectancy,
        status=status,
        decay_score=score,
    )


__all__ = [
    "STATUS_STABLE",
    "STATUS_IMPROVING",
    "STATUS_DECAYING",
    "STATUS_BROKEN",
    "STATUS_INSUFFICIENT",
    "BROKEN_DD_THRESHOLD",
    "IMPROVING_REL_CHANGE",
    "DECAYING_REL_CHANGE",
    "STABLE_PF_MIN",
    "DecayReport",
    "build_report",
]
