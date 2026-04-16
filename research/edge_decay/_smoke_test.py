"""
Synthetic smoke test for the edge-decay module.

Builds four candidates with deterministically-shaped trade histories:

    STABLE     — consistent +edge across the timeline
    IMPROVING  — edge ramps UP through time
    DECAYING   — edge ramps DOWN through time (still > 1.0 PF overall)
    BROKEN     — edge flips negative in the final window

Verifies that `build_report(...)` classifies each into the expected
status and produces sensible decay_score signs.

Run:
    python -m research.edge_decay._smoke_test
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from research.edge_decay import (
    STATUS_BROKEN,
    STATUS_DECAYING,
    STATUS_IMPROVING,
    STATUS_STABLE,
    build_report,
)


def _stable_trades(n: int = 400, seed: int = 0) -> list[float]:
    """Consistent small positive edge with no temporal drift.

    Uses a deterministic 5-trade cycle (3 wins / 2 losses of fixed
    magnitude) so every 50-trade window sees exactly the same
    composition — rel_change of every metric is ≈ 0.
    """
    # Pattern repeats every 5 trades: +0.004, +0.004, +0.004, -0.003, -0.003
    # Expectancy per trade = (3*0.004 - 2*0.003)/5 = +0.0012, PF = 12/6 = 2.0
    base = [0.004, 0.004, 0.004, -0.003, -0.003]
    out = [base[i % 5] for i in range(n)]
    # Tiny deterministic jitter so the series isn't perfectly flat
    # (avoids degenerate slope/variance corner cases); amplitude small
    # enough that rel_change stays well under 10%.
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0, 0.00001, n)
    return (np.asarray(out) + jitter).tolist()


def _improving_trades(n: int = 400, seed: int = 1) -> list[float]:
    """Start weak, end strong."""
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        # Winrate ramps from 50% at i=0 to 75% at i=n-1.
        wr = 0.50 + 0.25 * (i / (n - 1))
        # Win size ramps from +0.003 to +0.006.
        win_mag = 0.003 + 0.003 * (i / (n - 1))
        if rng.random() < wr:
            out[i] = win_mag + rng.normal(0, 0.0005)
        else:
            out[i] = -0.003 + rng.normal(0, 0.0005)
    return out.astype(float).tolist()


def _decaying_trades(n: int = 400, seed: int = 2) -> list[float]:
    """Start strong, end weaker — still overall PF > 1.0."""
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        wr = 0.75 - 0.20 * (i / (n - 1))    # 75% → 55%
        win_mag = 0.006 - 0.003 * (i / (n - 1))  # +0.006 → +0.003
        if rng.random() < wr:
            out[i] = win_mag + rng.normal(0, 0.0005)
        else:
            out[i] = -0.004 + rng.normal(0, 0.0005)
    return out.astype(float).tolist()


def _broken_trades(n: int = 400, seed: int = 3) -> list[float]:
    """Recent window is net-negative."""
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        # Early: 60% winners.  Late: 30% winners.
        frac = i / (n - 1)
        wr = 0.60 - 0.30 * frac
        if rng.random() < wr:
            out[i] = 0.003 + rng.normal(0, 0.0005)
        else:
            out[i] = -0.005 + rng.normal(0, 0.0005)
    return out.astype(float).tolist()


CASES = [
    ("STABLE_candidate",    "pattern",  STATUS_STABLE,    _stable_trades()),
    ("IMPROVING_candidate", "pattern",  STATUS_IMPROVING, _improving_trades()),
    ("DECAYING_candidate",  "sequence", STATUS_DECAYING,  _decaying_trades()),
    ("BROKEN_candidate",    "sequence", STATUS_BROKEN,    _broken_trades()),
]


def main() -> None:
    print("edge_decay smoke test")
    print("-" * 60)
    print(
        f"{'name':<22} {'expected':<10} {'got':<10} "
        f"{'pf':>5} {'exp':>10} {'rel_dx':>7} {'score':>6}"
    )

    failures: list[str] = []
    for name, ctype, expected, rets in CASES:
        report = build_report(
            candidate_name=name,
            candidate_type=ctype,
            asset="SYNTH",
            trade_returns=rets,
            window_sizes=(50, 100, 200),
        )
        pf_str = (
            f"{report.latest_profit_factor:5.2f}"
            if report.latest_profit_factor is not None
            else "  n/a"
        )
        exp_str = (
            f"{report.latest_expectancy:+.5f}"
            if report.latest_expectancy is not None
            else "      n/a"
        )
        mark = " " if report.status == expected else "!"
        print(
            f"{mark} {name:<20} {expected:<10} {report.status:<10} "
            f"{pf_str} {exp_str} "
            f"{report.rel_change_expectancy:+.3f} "
            f"{report.decay_score:+.2f}"
        )
        if report.status != expected:
            failures.append(
                f"{name}: expected {expected}, got {report.status}"
            )

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
