"""
aion/risk/sizing.py
────────────────────
Position sizing for Risk Allocation v1.

Sizing model: fixed fractional risk.

  risk_amount   = account_equity * (max_risk_per_trade_pct / 100)
  position_size = risk_amount / (stop_distance_points * point_value)

Where:
  point_value (from InstrumentSpec) = account currency per lot per point.
    EURUSD standard lot:  point_value = 10.0  USD/pip/lot
    US100.cash (example): point_value = 1.0   USD/point/lot  (broker-dependent)

Rounding rule:
  Raw lots are rounded DOWN to the nearest lot_step to never exceed the
  risk budget.  If the result falls below min_lot, min_lot is used.

  Note: enforcing min_lot on a small account with a wide stop will result
  in slightly more risk than intended.  This is expected and should be logged.

Example (EURUSD, $10 000 account, 1% risk, 10-pip stop):
  risk_amount   = 10 000 * 1% = 100 USD
  position_size = 100 / (10 * 10.0) = 1.00 lot
"""

from __future__ import annotations

import math

from aion.core.models import InstrumentSpec
from aion.risk.models import RiskProfile


def compute_risk_amount(profile: RiskProfile) -> float:
    """Monetary amount to risk on this trade (account currency).

    risk_amount = account_equity * (max_risk_per_trade_pct / 100)
    """
    return profile.account_equity * (profile.max_risk_per_trade_pct / 100.0)


def compute_position_size(
    risk_amount: float,
    stop_distance_points: float,
    instrument: InstrumentSpec,
) -> float:
    """Compute position size in lots.

    Formula
    -------
    raw_lots = risk_amount / (stop_distance_points * point_value)

    The result is floored to the nearest lot_step (conservative: never exceed
    the risk budget), then clipped to at least min_lot.

    Parameters
    ----------
    risk_amount:
        Monetary amount to risk (account currency).
    stop_distance_points:
        Stop distance in points (pips for forex, index points for indices).
        Must be positive — callers should validate this before calling.
    instrument:
        InstrumentSpec supplying point_value, min_lot, and lot_step.

    Returns
    -------
    float
        Position size in lots, rounded to lot_step precision.
    """
    raw_lots = risk_amount / (stop_distance_points * instrument.point_value)

    # Floor to nearest lot_step (never overshoot the risk budget)
    steps = math.floor(raw_lots / instrument.lot_step)
    lots = steps * instrument.lot_step

    # Enforce minimum tradeable lot size
    lots = max(instrument.min_lot, lots)

    # Snap to lot_step decimal precision to eliminate floating-point noise
    precision = _lot_step_precision(instrument.lot_step)
    return round(lots, precision)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _lot_step_precision(lot_step: float) -> int:
    """Return the number of decimal places implied by lot_step.

    Examples:
      lot_step=0.01  → 2
      lot_step=0.1   → 1
      lot_step=1.0   → 0
      lot_step=0.001 → 3
    """
    s = f"{lot_step:.10f}".rstrip("0")
    if "." in s:
        return len(s.split(".")[1])
    return 0
