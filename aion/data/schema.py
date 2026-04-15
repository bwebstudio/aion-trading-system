"""
aion.data.schema
────────────────
Column name constants and DataFrame schema validation helpers.

Why this exists:
- Column names are referenced in many places (adapters, normalizer,
  resampler, features, persistence).  A single source of truth prevents
  typos and makes refactoring trivial.
- Validation helpers make it easy to assert a DataFrame is in the
  expected shape before processing.

Rules:
- No pandas import at module level (keep import cost low when used as
  constants only).  Pandas is imported inside functions that need it.
- All constants are plain strings.
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────────
# Raw bar columns  (as loaded from CSV / MT5 before normalisation)
# ─────────────────────────────────────────────────────────────────────────────


class RawBarCols:
    TIMESTAMP = "timestamp"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    TICK_VOLUME = "tick_volume"
    REAL_VOLUME = "real_volume"
    SPREAD = "spread"

    REQUIRED: frozenset[str] = frozenset(
        {TIMESTAMP, OPEN, HIGH, LOW, CLOSE, TICK_VOLUME}
    )
    OPTIONAL: frozenset[str] = frozenset({REAL_VOLUME, SPREAD})
    NUMERIC: frozenset[str] = frozenset(
        {OPEN, HIGH, LOW, CLOSE, TICK_VOLUME, REAL_VOLUME, SPREAD}
    )
    ALL: frozenset[str] = REQUIRED | OPTIONAL


# ─────────────────────────────────────────────────────────────────────────────
# MarketBar columns  (after normalisation, persisted as Parquet)
# ─────────────────────────────────────────────────────────────────────────────


class MarketBarCols:
    SYMBOL = "symbol"
    TIMESTAMP_UTC = "timestamp_utc"
    TIMESTAMP_MARKET = "timestamp_market"
    TIMEFRAME = "timeframe"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    TICK_VOLUME = "tick_volume"
    REAL_VOLUME = "real_volume"
    SPREAD = "spread"
    SOURCE = "source"
    IS_VALID = "is_valid"

    REQUIRED: frozenset[str] = frozenset(
        {
            SYMBOL,
            TIMESTAMP_UTC,
            TIMEFRAME,
            OPEN,
            HIGH,
            LOW,
            CLOSE,
            TICK_VOLUME,
            SPREAD,
            SOURCE,
            IS_VALID,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# FeatureVector columns
# ─────────────────────────────────────────────────────────────────────────────


class FeatureCols:
    SYMBOL = "symbol"
    TIMESTAMP_UTC = "timestamp_utc"
    TIMEFRAME = "timeframe"
    ATR_14 = "atr_14"
    ROLLING_RANGE_10 = "rolling_range_10"
    ROLLING_RANGE_20 = "rolling_range_20"
    SESSION_HIGH = "session_high"
    SESSION_LOW = "session_low"
    OPENING_RANGE_HIGH = "opening_range_high"
    OPENING_RANGE_LOW = "opening_range_low"
    VWAP_SESSION = "vwap_session"
    SPREAD_MEAN_20 = "spread_mean_20"
    SPREAD_ZSCORE_20 = "spread_zscore_20"
    RETURN_1 = "return_1"
    RETURN_5 = "return_5"
    VOLATILITY_PERCENTILE_20 = "volatility_percentile_20"
    CANDLE_BODY = "candle_body"
    UPPER_WICK = "upper_wick"
    LOWER_WICK = "lower_wick"
    DISTANCE_TO_SESSION_HIGH = "distance_to_session_high"
    DISTANCE_TO_SESSION_LOW = "distance_to_session_low"
    FEATURE_SET_VERSION = "feature_set_version"


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────


class SchemaError(Exception):
    """Raised when a DataFrame does not match the expected schema."""


def assert_raw_bar_schema(df: object) -> None:
    """
    Raise SchemaError if `df` is missing required raw bar columns.

    Accepts any object with a `.columns` attribute (pandas DataFrame).
    """
    _assert_columns(df, RawBarCols.REQUIRED, "RawBar")


def assert_market_bar_schema(df: object) -> None:
    """Raise SchemaError if `df` is missing required market bar columns."""
    _assert_columns(df, MarketBarCols.REQUIRED, "MarketBar")


def _assert_columns(
    df: object, required: frozenset[str], schema_name: str
) -> None:
    actual = frozenset(getattr(df, "columns", []))
    missing = required - actual
    if missing:
        raise SchemaError(
            f"{schema_name} schema violation — missing columns: {sorted(missing)}"
        )
