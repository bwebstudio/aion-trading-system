"""
aion/app/orchestrator.py
─────────────────────────
Configuration model for the paper trading loop.

PaperTradingConfig is the single object passed to run_paper_loop().
It holds everything the loop needs to run without external dependencies:
  - risk rules (RiskProfile)
  - instrument specification (for sizing and price conversion)
  - fixed stop/target distances in pips
  - pip_size for converting pips → price levels
  - optional timeout in bars

Design notes:
  - pip_size is the price distance per pip in native price units:
      EURUSD standard lot : pip_size = tick_size * 10 = 0.0001
      US100.cash (pip=1pt): pip_size = tick_size * 1  = 0.01
  - stop_distance_points and target_distance_points use the same unit
    as the risk allocator's stop_distance_points argument (pips / points).
  - max_bars_open caps position lifetime.  None means no timeout.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from aion.core.models import InstrumentSpec
from aion.risk.models import RiskProfile


class PaperTradingConfig(BaseModel, frozen=True):
    """
    Complete configuration for one paper trading loop run.

    Example (EURUSD, conservative setup):
        PaperTradingConfig(
            risk_profile=RiskProfile(account_equity=10_000.0),
            instrument=eurusd_spec,
            stop_distance_points=10.0,
            target_distance_points=20.0,
            pip_size=0.0001,
            max_bars_open=50,
        )
    """

    risk_profile: RiskProfile
    """Account risk configuration (equity, limits, percentages)."""

    instrument: InstrumentSpec
    """Instrument being traded (for sizing via point_value, min_lot, lot_step)."""

    stop_distance_points: float
    """Fixed stop distance in pips / points.  Same unit as risk allocator."""

    target_distance_points: float | None = None
    """Fixed target distance in pips / points.  None means no take-profit."""

    pip_size: float = 0.0001
    """
    Price distance per pip in native instrument price units.
    EURUSD (5-decimal, pip_multiplier=10): pip_size = 0.00001 * 10 = 0.0001
    US100.cash (pip_multiplier=1):          pip_size = tick_size * 1
    """

    max_bars_open: int | None = None
    """
    Maximum number of bars a position can remain open before forced close.
    None means positions are held until stop, target, or end of data.
    """

    slippage_points: float = 0.0
    """
    Execution slippage applied to entry fills, in price points.
    Applied adversely: LONG fills higher, SHORT fills lower.
    Default 0.0 preserves zero-slippage behaviour.
    """

    @field_validator("stop_distance_points")
    @classmethod
    def stop_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"stop_distance_points must be positive, got {v}.")
        return v

    @field_validator("pip_size")
    @classmethod
    def pip_size_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"pip_size must be positive, got {v}.")
        return v
