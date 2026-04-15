"""
aion.core.constants
───────────────────
Platform-wide constants.

Rules:
- No business logic here — only fixed values.
- All values are module-level constants (UPPER_SNAKE_CASE).
- Import freely from anywhere inside aion/.
"""

from aion.core.enums import Timeframe

# ─────────────────────────────────────────────
# Timeframe metadata
# ─────────────────────────────────────────────

# Duration in minutes for each timeframe.
# Used for resampling, bar gap detection, and time-based logic.
TIMEFRAME_MINUTES: dict[Timeframe, int] = {
    Timeframe.M1: 1,
    Timeframe.M5: 5,
    Timeframe.M15: 15,
    Timeframe.M30: 30,
    Timeframe.H1: 60,
    Timeframe.H4: 240,
    Timeframe.D1: 1440,
}

# ─────────────────────────────────────────────
# Feature engineering parameters
# ─────────────────────────────────────────────

ATR_PERIOD: int = 14
ROLLING_RANGE_SHORT: int = 10
ROLLING_RANGE_LONG: int = 20
SPREAD_LOOKBACK: int = 20
VOLATILITY_PERCENTILE_LOOKBACK: int = 20
RETURN_SHORT_BARS: int = 1
RETURN_LONG_BARS: int = 5

# Opening range: first N minutes of the primary session are the "opening range"
OPENING_RANGE_MINUTES: int = 30

# ─────────────────────────────────────────────
# Session definitions (local timezone + local times)
# ─────────────────────────────────────────────
# Each session is defined in its *own* timezone so DST is
# handled correctly by zoneinfo when converting to UTC.
# Format: (hour, minute) in 24h local time.

SESSION_DEFINITIONS: dict[str, dict] = {
    "ASIA": {
        "timezone": "Asia/Tokyo",
        "open": (9, 0),    # 09:00 JST  = 00:00 UTC (no DST in Tokyo)
        "close": (18, 0),  # 18:00 JST  = 09:00 UTC
    },
    "LONDON": {
        "timezone": "Europe/London",
        "open": (8, 0),    # 08:00 GMT/BST → 08:00 or 07:00 UTC (DST-aware)
        "close": (16, 30), # 16:30 GMT/BST
    },
    "NEW_YORK": {
        "timezone": "America/New_York",
        "open": (9, 30),   # 09:30 EST/EDT → 14:30 or 13:30 UTC (DST-aware)
        "close": (17, 0),  # 17:00 EST/EDT
    },
}

# ─────────────────────────────────────────────
# Snapshot / versioning
# ─────────────────────────────────────────────

SNAPSHOT_VERSION: str = "1.0"
FEATURE_SET_VERSION: str = "1.0.0"

# ─────────────────────────────────────────────
# Data quality thresholds
# ─────────────────────────────────────────────

# A bar is considered "stale" if OHLC values are all identical to
# the previous bar (indicating no market activity or feed issue).
STALE_BAR_CONSECUTIVE_THRESHOLD: int = 3

# A bar is a "spike" if its high-low range exceeds this multiple of
# the rolling ATR.  Heuristic — can be tuned per instrument.
SPIKE_ATR_MULTIPLIER: float = 10.0

# Maximum allowed gap (in bar periods) before it is flagged as missing.
MAX_GAP_BARS: int = 1

# Minimum quality score to consider a dataset usable for live decisions.
MIN_QUALITY_SCORE: float = 0.90

# ─────────────────────────────────────────────
# Bars to keep in memory per snapshot
# ─────────────────────────────────────────────

SNAPSHOT_BARS_M1: int = 100
SNAPSHOT_BARS_M5: int = 100
SNAPSHOT_BARS_M15: int = 100
