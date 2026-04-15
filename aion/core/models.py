"""
aion.core.models
────────────────
All domain models (Pydantic v2).

These are the contracts between modules.  Any data crossing a module
boundary MUST be represented as one of these models.

Rules:
- All models are immutable (frozen=True).
- Optional fields use `float | None` — never bare Optional.
- datetime fields are always UTC-aware (validated in normalizer).
- No business logic inside models — only data + simple derived properties.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aion.core.enums import (
    AssetClass,
    DataSource,
    SessionName,
    Timeframe,
)
from aion.core.ids import new_snapshot_id


# ─────────────────────────────────────────────────────────────────────────────
# Instrument
# ─────────────────────────────────────────────────────────────────────────────


class InstrumentSpec(BaseModel):
    """
    Static specification of a tradeable instrument.

    Loaded once at startup from a config file or registry.
    Never mutated at runtime.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    """Canonical symbol used throughout the platform.  E.g. 'EURUSD'."""

    broker_symbol: str
    """Symbol as the broker names it.  May differ (e.g. 'EURUSDm')."""

    asset_class: AssetClass

    # Timezone strings (IANA tz database names, e.g. 'America/New_York')
    price_timezone: str
    """Timezone in which prices are quoted / published."""

    market_timezone: str
    """Primary exchange or reference market timezone."""

    broker_timezone: str
    """Timezone used by the broker for bar timestamps."""

    # Contract specification
    tick_size: float
    """Minimum price increment.  E.g. 0.00001 for EURUSD."""

    point_value: float
    """Monetary value of one full point move per lot.  E.g. 10.0 for EURUSD."""

    contract_size: float
    """Units of base currency per lot.  E.g. 100_000 for standard forex lot."""

    min_lot: float
    """Minimum tradeable lot size.  E.g. 0.01."""

    lot_step: float
    """Lot size increment.  E.g. 0.01."""

    currency_profit: str
    """Currency in which profit/loss is realised.  E.g. 'USD'."""

    currency_margin: str
    """Currency used to calculate margin.  E.g. 'USD'."""

    # Session / calendar
    session_calendar: str
    """Key into the session calendar registry.  E.g. 'forex_standard'."""

    trading_hours_label: str
    """Human-readable description.  E.g. 'Sun 22:00 – Fri 22:00 UTC'."""


# ─────────────────────────────────────────────────────────────────────────────
# Bars
# ─────────────────────────────────────────────────────────────────────────────


class RawBar(BaseModel):
    """
    A single OHLCV bar as received from the data source.

    - `timestamp` may be broker-local or UTC — it depends on the source.
    - The normalizer converts RawBar → MarketBar with guaranteed UTC.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    timestamp: datetime
    """Timestamp in the broker's timezone (may be naive or tz-aware)."""

    open: float
    high: float
    low: float
    close: float
    tick_volume: float
    real_volume: float = 0.0
    spread: float = 0.0
    source: DataSource


class MarketBar(BaseModel):
    """
    A normalised, validated OHLCV bar.

    - `timestamp_utc` is always UTC-aware.
    - `timestamp_market` is in the instrument's market timezone.
    - `is_valid` is False if OHLC constraints were violated during normalisation.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    timestamp_utc: datetime
    timestamp_market: datetime
    timeframe: Timeframe
    open: float
    high: float
    low: float
    close: float
    tick_volume: float
    real_volume: float
    spread: float
    source: DataSource
    is_valid: bool = True

    # ── Derived properties (computed, not stored) ──────────────────────────

    @property
    def body(self) -> float:
        """Absolute distance between open and close."""
        return abs(self.close - self.open)

    @property
    def full_range(self) -> float:
        """High-to-low range of the bar."""
        return self.high - self.low

    @property
    def upper_wick(self) -> float:
        """Distance from the top of the body to the high."""
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        """Distance from the bottom of the body to the low."""
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open


# ─────────────────────────────────────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────────────────────────────────────


class SessionContext(BaseModel):
    """
    Describes the market session state at a specific moment in time.

    Built by aion.data.sessions.build_session_context().
    """

    model_config = ConfigDict(frozen=True)

    trading_day: date
    """The calendar date of the trading day (UTC date)."""

    broker_time: datetime
    market_time: datetime
    local_time: datetime

    # Active sessions
    is_asia: bool
    is_london: bool
    is_new_york: bool

    # Composite state
    is_session_open_window: bool
    """True if any major session is currently open."""

    opening_range_active: bool
    """True during the first OPENING_RANGE_MINUTES of the primary session."""

    opening_range_completed: bool
    """True after the opening range period has elapsed."""

    session_name: SessionName
    """Primary session label for this moment."""

    session_open_utc: datetime | None = None
    """UTC time when the current session opened today."""

    session_close_utc: datetime | None = None
    """UTC time when the current session will close today."""


# ─────────────────────────────────────────────────────────────────────────────
# Features
# ─────────────────────────────────────────────────────────────────────────────


class FeatureVector(BaseModel):
    """
    Computed features for a single bar/timeframe/symbol combination.

    All values are `float | None`.  None means the feature could not be
    computed (insufficient history, missing data).  Consumers must handle None.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    timestamp_utc: datetime
    timeframe: Timeframe

    # ── Volatility ────────────────────────────
    atr_14: float | None
    rolling_range_10: float | None
    rolling_range_20: float | None
    volatility_percentile_20: float | None

    # ── Session context ────────────────────────
    session_high: float | None
    session_low: float | None
    opening_range_high: float | None
    opening_range_low: float | None
    vwap_session: float | None

    # ── Spread ────────────────────────────────
    spread_mean_20: float | None
    spread_zscore_20: float | None

    # ── Returns ───────────────────────────────
    return_1: float | None
    return_5: float | None

    # ── Candle structure ──────────────────────
    candle_body: float | None
    upper_wick: float | None
    lower_wick: float | None

    # ── Distance to session extremes ──────────
    distance_to_session_high: float | None
    distance_to_session_low: float | None

    # ── Versioning ────────────────────────────
    feature_set_version: str


# ─────────────────────────────────────────────────────────────────────────────
# Data Quality
# ─────────────────────────────────────────────────────────────────────────────


class DataQualityReport(BaseModel):
    """
    Summary of data quality issues found during validation.

    `quality_score` is a float in [0.0, 1.0].
    1.0 = perfect data.  0.0 = completely unusable.

    The score formula is defined in aion.data.validator.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    timeframe: Timeframe
    rows_checked: int
    missing_bars: int
    duplicate_timestamps: int
    out_of_order_rows: int
    stale_bars: int
    spike_bars: int
    null_rows: int
    quality_score: float
    warnings: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot — top-level output of the Market Engine
# ─────────────────────────────────────────────────────────────────────────────


class MarketSnapshot(BaseModel):
    """
    The complete, self-contained view of one symbol at one moment in time.

    This is the primary input to every downstream module:
    - Regime Detector
    - Strategy Engines
    - Meta-AI Layer

    Everything a strategy needs is here.  No strategy should need to
    query the database or call the market engine separately.
    """

    model_config = ConfigDict(frozen=True)

    snapshot_id: str = Field(default_factory=new_snapshot_id)
    symbol: str
    timestamp_utc: datetime

    base_timeframe: Timeframe
    """The lowest-resolution timeframe from which higher TFs were built."""

    instrument: InstrumentSpec
    session_context: SessionContext
    latest_bar: MarketBar

    # Multi-timeframe bar windows (most recent N bars, oldest first)
    bars_m1: list[MarketBar]
    bars_m5: list[MarketBar]
    bars_m15: list[MarketBar]

    feature_vector: FeatureVector
    quality_report: DataQualityReport

    snapshot_version: str = "1.0"

    # ── Convenience accessors ─────────────────

    @property
    def is_usable(self) -> bool:
        """False if data quality is too low for reliable decisions."""
        from aion.core.constants import MIN_QUALITY_SCORE
        return self.quality_report.quality_score >= MIN_QUALITY_SCORE

    def bars_for(self, timeframe: Timeframe) -> list[MarketBar]:
        """Return the bar list for the requested timeframe."""
        mapping: dict[Timeframe, list[MarketBar]] = {
            Timeframe.M1: self.bars_m1,
            Timeframe.M5: self.bars_m5,
            Timeframe.M15: self.bars_m15,
        }
        if timeframe not in mapping:
            raise ValueError(
                f"Timeframe {timeframe} not available in snapshot. "
                f"Available: {list(mapping.keys())}"
            )
        return mapping[timeframe]
