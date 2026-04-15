"""
aion.core.enums
───────────────
All enumerations used across the platform.

Rules:
- All enums inherit from (str, Enum) so they are JSON-serialisable
  and compare naturally with string literals.
- Never import from aion.data or any sub-package here.
"""

from enum import Enum


# ─────────────────────────────────────────────
# Instrument / Market
# ─────────────────────────────────────────────


class AssetClass(str, Enum):
    FOREX = "FOREX"
    INDICES = "INDICES"
    COMMODITIES = "COMMODITIES"
    CRYPTO = "CRYPTO"
    STOCKS = "STOCKS"


class Timeframe(str, Enum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"


class DataSource(str, Enum):
    CSV = "CSV"
    MT5 = "MT5"
    SYNTHETIC = "SYNTHETIC"


# ─────────────────────────────────────────────
# Sessions
# ─────────────────────────────────────────────


class SessionName(str, Enum):
    ASIA = "ASIA"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    OVERLAP_LONDON_NY = "OVERLAP_LONDON_NY"
    OFF_HOURS = "OFF_HOURS"


# ─────────────────────────────────────────────
# Regime (placeholder — ML classifier added later)
# ─────────────────────────────────────────────


class RegimeLabel(str, Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    COMPRESSION = "COMPRESSION"
    EXPANSION = "EXPANSION"
    REVERSAL = "REVERSAL"
    CHAOTIC = "CHAOTIC"
    UNKNOWN = "UNKNOWN"


# ─────────────────────────────────────────────
# System
# ─────────────────────────────────────────────


class SystemMode(str, Enum):
    PAPER = "PAPER"
    LIVE = "LIVE"


class SystemEnvironment(str, Enum):
    RESEARCH = "RESEARCH"
    PRODUCTION = "PRODUCTION"


# ─────────────────────────────────────────────
# Decisions (skeleton for Meta-AI layer)
# ─────────────────────────────────────────────


class TradeDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class DecisionOutcome(str, Enum):
    TRADE = "TRADE"
    NO_TRADE = "NO_TRADE"
    RISK_REJECTED = "RISK_REJECTED"
