"""
aion.risk
──────────
Risk Allocation v1.

Public API
----------
  evaluate(candidate, profile, state, instrument, stop_distance_points,
           target_distance_points=None) -> RiskDecision

  RiskProfile     — account-level risk configuration
  PortfolioState  — current portfolio snapshot
  RiskDecision    — allocator output (approved/rejected + sizing)
"""

from aion.risk.allocator import evaluate
from aion.risk.models import PortfolioState, RiskDecision, RiskProfile

__all__ = [
    "evaluate",
    "PortfolioState",
    "RiskDecision",
    "RiskProfile",
]
