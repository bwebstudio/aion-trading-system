"""
aion.execution.execution_model
───────────────────────────────
Realistic execution cost model.

Estimates per-fill spread and slippage from OHLC bar context:
  - bar volatility (range)
  - bar momentum (body / range)
  - session of the market
  - entry type (limit / retest / breakout)

Formula
-------
range_m1            = bar.high - bar.low
momentum            = |bar.close - bar.open| / max(range_m1, 1e-6)
volatility_component = k_vol * range_m1
momentum_component   = k_mom * momentum * range_m1
base_slippage        = volatility_component + momentum_component
slippage             = base_slippage
                       * session_multiplier[session]
                       * entry_multiplier[entry_type]
                       * random.uniform(0.8, 1.2)

Spread
------
spread = max(min_spread_symbol, 0.02 * atr_1m)

Fallback
--------
If any input is missing or malformed, both spread and slippage default to 0
so existing zero-slippage backtests continue to work unchanged.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Session handling
# ─────────────────────────────────────────────────────────────────────────────

ASIA = "ASIA"
LONDON = "LONDON"
NY_OPEN = "NY_OPEN"
NY_MID = "NY_MID"
NY_CLOSE = "NY_CLOSE"
OFF_HOURS = "OFF_HOURS"

SESSION_MULTIPLIERS: dict[str, float] = {
    ASIA: 0.7,
    LONDON: 1.0,
    NY_OPEN: 1.4,
    NY_MID: 1.0,
    NY_CLOSE: 1.2,
    OFF_HOURS: 1.0,
}

ENTRY_MULTIPLIERS: dict[str, float] = {
    "limit": 0.5,
    "retest": 1.0,
    "breakout": 1.5,
}

DEFAULT_MIN_SPREAD: dict[str, float] = {
    "US100.cash": 1.5,
    "XAUUSD": 8.0,
    "BTCUSD": 80.0,
}


def detect_session(timestamp: datetime) -> str:
    """
    Return the session label for a UTC timestamp.

    Windows (UTC):
      ASIA      00:00 – 07:00
      LONDON    07:00 – 13:30
      NY_OPEN   13:30 – 15:00
      NY_MID    15:00 – 19:00
      NY_CLOSE  19:00 – 21:00
      else      OFF_HOURS
    """
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)

    t = timestamp.time()
    if time(0, 0) <= t < time(7, 0):
        return ASIA
    if time(7, 0) <= t < time(13, 30):
        return LONDON
    if time(13, 30) <= t < time(15, 0):
        return NY_OPEN
    if time(15, 0) <= t < time(19, 0):
        return NY_MID
    if time(19, 0) <= t < time(21, 0):
        return NY_CLOSE
    return OFF_HOURS


# ─────────────────────────────────────────────────────────────────────────────
# Per-symbol parameters
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SymbolParams:
    min_spread: float
    k_vol: float = 0.05
    k_mom: float = 0.05


DEFAULT_PARAMS = SymbolParams(min_spread=0.0, k_vol=0.05, k_mom=0.05)


# ─────────────────────────────────────────────────────────────────────────────
# Execution model
# ─────────────────────────────────────────────────────────────────────────────


class ExecutionModel:
    """
    Dynamic spread + slippage estimator.

    Usage:
        model = ExecutionModel.from_config("aion/config/execution_config.yaml")
        spread   = model.estimate_spread("US100.cash", atr_1m=3.2)
        slippage = model.estimate_slippage(bar, session="NY_OPEN",
                                           entry_type="breakout",
                                           symbol="US100.cash")
    """

    def __init__(
        self,
        params_by_symbol: dict[str, SymbolParams] | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self._params = params_by_symbol or {}
        self._rng = rng or random.Random()

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, path: str | Path) -> ExecutionModel:
        """Build an ExecutionModel from a YAML config file."""
        p = Path(path)
        if not p.exists() or yaml is None:
            return cls()
        try:
            with p.open("r", encoding="utf-8") as fh:
                raw: dict[str, Any] = yaml.safe_load(fh) or {}
        except Exception:
            return cls()

        params: dict[str, SymbolParams] = {}
        for symbol, cfg in raw.items():
            if not isinstance(cfg, dict):
                continue
            try:
                params[symbol] = SymbolParams(
                    min_spread=float(cfg.get("min_spread", 0.0)),
                    k_vol=float(cfg.get("k_vol", 0.05)),
                    k_mom=float(cfg.get("k_mom", 0.05)),
                )
            except (TypeError, ValueError):
                continue
        return cls(params_by_symbol=params)

    # ── Parameter access ──────────────────────────────────────────────────────

    def params_for(self, symbol: str) -> SymbolParams:
        if symbol in self._params:
            return self._params[symbol]
        if symbol in DEFAULT_MIN_SPREAD:
            return SymbolParams(min_spread=DEFAULT_MIN_SPREAD[symbol])
        return DEFAULT_PARAMS

    # ── Spread ────────────────────────────────────────────────────────────────

    def estimate_spread(self, symbol: str, atr_1m: float | None) -> float:
        """
        Estimate spread in price points.

        spread = max(min_spread_symbol, 0.02 * atr_1m)

        If atr_1m is missing or invalid, returns min_spread for the symbol.
        """
        params = self.params_for(symbol)
        try:
            if atr_1m is None or atr_1m <= 0:
                return params.min_spread
            dynamic = 0.02 * float(atr_1m)
            return max(params.min_spread, dynamic)
        except (TypeError, ValueError):
            return 0.0

    # ── Slippage ──────────────────────────────────────────────────────────────

    def estimate_slippage(
        self,
        bar: Any,
        session: str,
        entry_type: str,
        symbol: str | None = None,
    ) -> float:
        """
        Estimate entry slippage in price points.

        Parameters
        ----------
        bar:
            Any object with numeric `open`, `high`, `low`, `close` attributes.
        session:
            Session label (ASIA, LONDON, NY_OPEN, NY_MID, NY_CLOSE).
            Unknown labels fall back to multiplier 1.0.
        entry_type:
            One of "limit", "retest", "breakout".  Unknown types
            fall back to multiplier 1.0.
        symbol:
            Optional symbol key for per-symbol k_vol / k_mom overrides.
        """
        try:
            high = float(bar.high)
            low = float(bar.low)
            open_ = float(bar.open)
            close = float(bar.close)
        except (AttributeError, TypeError, ValueError):
            return 0.0

        range_m1 = high - low
        if range_m1 <= 0:
            return 0.0

        momentum = abs(close - open_) / max(range_m1, 1e-6)

        params = self.params_for(symbol) if symbol else DEFAULT_PARAMS
        volatility_component = params.k_vol * range_m1
        momentum_component = params.k_mom * momentum * range_m1
        base_slippage = volatility_component + momentum_component

        session_adjustment = SESSION_MULTIPLIERS.get(session, 1.0)
        entry_adjustment = ENTRY_MULTIPLIERS.get(entry_type, 1.0)

        slippage = base_slippage * session_adjustment * entry_adjustment
        slippage *= self._rng.uniform(0.8, 1.2)
        return max(slippage, 0.0)


__all__ = [
    "ExecutionModel",
    "SymbolParams",
    "detect_session",
    "SESSION_MULTIPLIERS",
    "ENTRY_MULTIPLIERS",
    "DEFAULT_MIN_SPREAD",
    "ASIA",
    "LONDON",
    "NY_OPEN",
    "NY_MID",
    "NY_CLOSE",
    "OFF_HOURS",
]
