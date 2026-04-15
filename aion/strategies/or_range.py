"""
aion.strategies.or_range
─────────────────────────
Parametrisable Opening Range computation.

The OR (Opening Range) is a price range defined at the start of a trading
session.  Different strategies use different methods to compute it:

  SINGLE_CANDLE  — One candle at a specific time.
                   Example: M5 bar at 16:30 broker time.
                   OR high = candle.high, OR low = candle.low.

  CANDLE_BLOCK   — A block of consecutive M1 bars starting at a specific time.
                   Example: M1 bars from 16:30 to 16:34 (5 bars).
                   OR high = max(highs), OR low = min(lows).

This module is pure computation — no signals, no state, no strategy logic.
It answers one question: "What is the OR for this session?"

Usage:
    from aion.strategies.or_range import (
        OpeningRangeConfig, ORMethod, compute_opening_range,
    )

    config = OpeningRangeConfig(
        method=ORMethod.SINGLE_CANDLE,
        reference_time=time(16, 30),
        candle_timeframe=Timeframe.M5,
    )

    level = compute_opening_range(bars, config)
    if level is not None:
        print(f"OR: {level.or_high} / {level.or_low}")
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aion.core.enums import Timeframe
from aion.core.models import MarketBar


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class ORMethod(str, Enum):
    """How the Opening Range is computed from bar data."""

    SINGLE_CANDLE = "SINGLE_CANDLE"
    """OR = one candle at reference_time (high/low of that candle)."""

    CANDLE_BLOCK = "CANDLE_BLOCK"
    """OR = max(highs) / min(lows) of N consecutive bars from reference_time."""


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


class OpeningRangeConfig(BaseModel):
    """
    Parametrises how the Opening Range is identified from bar data.

    The reference_time is matched against bar timestamps using the
    timezone_source field:
      "broker"  → compare against bar.timestamp_utc (assumes broker=UTC)
      "market"  → compare against bar.timestamp_market

    For MT5 brokers on UTC+2, the bar timestamps are already converted to
    UTC by the normaliser, so reference_time should be in UTC.
    """

    model_config = ConfigDict(frozen=True)

    method: ORMethod

    reference_time: time
    """
    Time-of-day that marks the start of the Opening Range.
    Example: time(13, 30) for NY open at 13:30 UTC.
    """

    timezone_source: str = "broker"
    """
    Which bar timestamp to compare against reference_time.
      'broker' → timestamp_utc   (default; correct when broker TZ = UTC)
      'market' → timestamp_market (use when reference_time is in market TZ)
    """

    # ── SINGLE_CANDLE parameters ────────────────────────────────────────────

    candle_timeframe: Timeframe | None = None
    """
    Required for SINGLE_CANDLE.  The timeframe of the candle to capture.
    The function searches for a bar matching this timeframe at reference_time.
    """

    # ── CANDLE_BLOCK parameters ─────────────────────────────────────────────

    block_duration_minutes: int | None = None
    """
    Required for CANDLE_BLOCK.  Duration of the block in minutes.
    Example: 5 → capture M1 bars from reference_time to reference_time + 4min.
    """

    block_timeframe: Timeframe = Timeframe.M1
    """Timeframe of individual bars in the block.  Default M1."""

    # ── Range validation ────────────────────────────────────────────────────

    min_range_points: float = 0.0
    """Minimum OR range in price points.  Below this → OR rejected."""

    max_range_points: float | None = None
    """Maximum OR range in price points.  Above this → OR rejected.  None = no limit."""

    @field_validator("block_duration_minutes")
    @classmethod
    def _block_must_be_positive(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("block_duration_minutes must be > 0.")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────


class OpeningRangeLevel(BaseModel):
    """
    The computed Opening Range — immutable, no signal, no direction.

    Downstream engines consume this to decide breakout direction,
    stop placement, etc.
    """

    model_config = ConfigDict(frozen=True)

    or_high: float
    or_low: float
    midpoint: float
    range_points: float
    method: ORMethod
    computed_at: datetime
    """Timestamp of the last bar used to compute this OR."""

    source_bars: int
    """Number of bars that contributed to this OR level."""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def compute_opening_range(
    bars: list[MarketBar],
    config: OpeningRangeConfig,
) -> OpeningRangeLevel | None:
    """
    Compute the Opening Range from a list of bars using the given config.

    Parameters
    ----------
    bars:
        Bars sorted ascending by timestamp.  Must include the bar(s) at
        the reference_time.  The function scans the list for matching bars.
    config:
        How to identify and validate the OR.

    Returns
    -------
    OpeningRangeLevel if the OR was found and passes validation.
    None if no matching bars were found or the range failed validation.
    """
    if not bars:
        return None

    if config.method == ORMethod.SINGLE_CANDLE:
        return _compute_single_candle(bars, config)
    elif config.method == ORMethod.CANDLE_BLOCK:
        return _compute_candle_block(bars, config)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Internal — SINGLE_CANDLE
# ─────────────────────────────────────────────────────────────────────────────


def _compute_single_candle(
    bars: list[MarketBar],
    config: OpeningRangeConfig,
) -> OpeningRangeLevel | None:
    """Find one bar matching reference_time + candle_timeframe."""
    if config.candle_timeframe is None:
        return None

    ref = config.reference_time

    for bar in bars:
        bar_time = _bar_time(bar, config.timezone_source)
        if bar_time == ref and bar.timeframe == config.candle_timeframe:
            return _build_level(
                or_high=bar.high,
                or_low=bar.low,
                computed_at=bar.timestamp_utc,
                source_bars=1,
                method=ORMethod.SINGLE_CANDLE,
                config=config,
            )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Internal — CANDLE_BLOCK
# ─────────────────────────────────────────────────────────────────────────────


def _compute_candle_block(
    bars: list[MarketBar],
    config: OpeningRangeConfig,
) -> OpeningRangeLevel | None:
    """Collect bars within [reference_time, reference_time + duration) and aggregate."""
    if config.block_duration_minutes is None:
        return None

    ref = config.reference_time
    end = _add_minutes_to_time(ref, config.block_duration_minutes)

    matched: list[MarketBar] = []
    for bar in bars:
        if bar.timeframe != config.block_timeframe:
            continue
        bar_time = _bar_time(bar, config.timezone_source)
        if _time_in_range(bar_time, ref, end):
            matched.append(bar)

    if not matched:
        return None

    or_high = max(b.high for b in matched)
    or_low = min(b.low for b in matched)
    computed_at = matched[-1].timestamp_utc

    return _build_level(
        or_high=or_high,
        or_low=or_low,
        computed_at=computed_at,
        source_bars=len(matched),
        method=ORMethod.CANDLE_BLOCK,
        config=config,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _bar_time(bar: MarketBar, timezone_source: str) -> time:
    """Extract the time-of-day from a bar using the configured source."""
    if timezone_source == "market":
        return bar.timestamp_market.time()
    # Default: broker → timestamp_utc
    return bar.timestamp_utc.time()


def _add_minutes_to_time(t: time, minutes: int) -> time:
    """Add minutes to a time-of-day, wrapping at midnight."""
    dt = datetime(2000, 1, 1, t.hour, t.minute, t.second)
    result = dt + timedelta(minutes=minutes)
    return result.time()


def _time_in_range(t: time, start: time, end: time) -> bool:
    """Check if t is in [start, end).  Does NOT handle midnight wrap."""
    return start <= t < end


def _build_level(
    or_high: float,
    or_low: float,
    computed_at: datetime,
    source_bars: int,
    method: ORMethod,
    config: OpeningRangeConfig,
) -> OpeningRangeLevel | None:
    """Build and validate an OpeningRangeLevel."""
    range_points = or_high - or_low

    if range_points < config.min_range_points:
        return None

    if config.max_range_points is not None and range_points > config.max_range_points:
        return None

    midpoint = round((or_high + or_low) / 2, 5)

    return OpeningRangeLevel(
        or_high=or_high,
        or_low=or_low,
        midpoint=midpoint,
        range_points=range_points,
        method=method,
        computed_at=computed_at,
        source_bars=source_bars,
    )
