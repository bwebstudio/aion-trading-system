"""
aion.app
─────────
Paper Trading Loop v1.

Public API
----------
  run_paper_loop(snapshots, engines, config) -> PaperTradingResult

  PaperTradingConfig   — loop configuration (risk, instrument, stop/target, pip_size)
  PaperTradingResult   — run output (summary + state + journal)
  PaperTradingSummary  — aggregate metrics
  StrategyBreakdown    — per-strategy metrics
  format_summary       — render a human-readable result string
"""

from aion.app.loop import run_paper_loop
from aion.app.orchestrator import PaperTradingConfig
from aion.app.summary import (
    PaperTradingResult,
    PaperTradingSummary,
    StrategyBreakdown,
    format_summary,
)

__all__ = [
    "run_paper_loop",
    "PaperTradingConfig",
    "PaperTradingResult",
    "PaperTradingSummary",
    "StrategyBreakdown",
    "format_summary",
]
