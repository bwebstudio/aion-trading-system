"""
aion/app/summary.py
────────────────────
Result models and summary formatting for the paper trading loop.

Model hierarchy:
  StrategyBreakdown  — per-strategy metrics from one loop run
  PaperTradingSummary — aggregate summary of a complete run
  PaperTradingResult  — top-level container with summary + live state + journal

format_summary(result) -> str renders a human-readable report suitable for
printing to the console or displaying on a non-technical dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aion.execution.journal import ExecutionJournal
from aion.execution.state import ExecutionState


# ─────────────────────────────────────────────────────────────────────────────
# StrategyBreakdown
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StrategyBreakdown:
    """Per-strategy metrics from a single paper trading loop run."""

    strategy_id: str
    signals: int
    """Number of CANDIDATE results produced by this strategy."""

    risk_approved: int
    """Signals that passed all risk rules."""

    executed: int
    """Positions actually opened (filled orders)."""

    closed: int
    """Positions that were closed during the run."""

    pnl: float
    """Realized P&L from closed positions (account currency)."""

    win_count: int
    """Closed positions with pnl_amount > 0."""

    loss_count: int
    """Closed positions with pnl_amount < 0."""


# ─────────────────────────────────────────────────────────────────────────────
# PaperTradingSummary
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PaperTradingSummary:
    """Aggregate summary of a complete paper trading loop run."""

    run_id: str
    snapshots_evaluated: int
    total_signals: int
    """Total CANDIDATE results across all strategies."""

    risk_approved: int
    """Signals approved by the risk allocator."""

    total_executed: int
    """Positions actually opened (filled)."""

    positions_closed: int
    """Positions closed during the run (stop, target, timeout)."""

    positions_still_open: int
    """Positions still open at end of data (not yet closed)."""

    total_pnl: float
    """Sum of realized P&L from all closed positions."""

    win_count: int
    loss_count: int

    avg_r_multiple: float | None
    """Average R-multiple across closed positions.  None if no closed positions."""

    strategy_breakdown: list[StrategyBreakdown]
    elapsed_seconds: float


# ─────────────────────────────────────────────────────────────────────────────
# PaperTradingResult
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PaperTradingResult:
    """
    Top-level output of run_paper_loop().

    Not frozen — holds references to the live ExecutionState and
    ExecutionJournal which are mutable objects.
    """

    summary: PaperTradingSummary
    state: ExecutionState
    journal: ExecutionJournal


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────


def format_summary(result: PaperTradingResult) -> str:
    """
    Render a human-readable summary of a paper trading loop run.

    Designed to be clear for a non-technical reader.  Returns a multi-line
    string suitable for print() or dashboard display.
    """
    s = result.summary
    lines: list[str] = []

    def _line(text: str = "") -> None:
        lines.append(text)

    _line("Paper Trading Loop - Results")
    _line("=" * 35)
    _line(f"Run ID             : {s.run_id}")
    _line(f"Snapshots evaluated: {s.snapshots_evaluated}")
    _line()

    _line("Signal Flow")
    _line("-" * 20)
    _line(f"Signals generated  : {s.total_signals}")
    _line(f"Risk approved      : {s.risk_approved}")
    _line(f"Positions opened   : {s.total_executed}")
    _line(f"Positions closed   : {s.positions_closed}")
    _line(f"Still open (end)   : {s.positions_still_open}")
    _line()

    _line("P&L Summary")
    _line("-" * 20)
    pnl_sign = "+" if s.total_pnl >= 0 else ""
    _line(f"Total realized P&L : {pnl_sign}${s.total_pnl:.2f}")

    if s.avg_r_multiple is not None:
        r_sign = "+" if s.avg_r_multiple >= 0 else ""
        _line(f"Avg R-multiple     : {r_sign}{s.avg_r_multiple:.2f}R")

    total_decided = s.win_count + s.loss_count
    if total_decided > 0:
        win_pct = 100 * s.win_count / total_decided
        _line(f"Win rate           : {s.win_count}/{total_decided} ({win_pct:.0f}%)")
    else:
        _line("Win rate           : n/a")

    _line(f"Wins               : {s.win_count}")
    _line(f"Losses             : {s.loss_count}")
    _line()

    if s.strategy_breakdown:
        _line("Strategy Breakdown")
        _line("-" * 20)
        for bd in s.strategy_breakdown:
            pnl_sign = "+" if bd.pnl >= 0 else ""
            _line(f"  {bd.strategy_id}")
            _line(f"    Signals  : {bd.signals}")
            _line(f"    Approved : {bd.risk_approved}")
            _line(f"    Executed : {bd.executed}")
            _line(f"    Closed   : {bd.closed}")
            _line(f"    P&L      : {pnl_sign}${bd.pnl:.2f}")
            _line(f"    W / L    : {bd.win_count} / {bd.loss_count}")

    return "\n".join(lines)
