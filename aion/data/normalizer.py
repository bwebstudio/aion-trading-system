"""
aion.data.normalizer
─────────────────────
Convert RawBar → MarketBar.

Responsibilities:
  1. Resolve timezone ambiguity (broker-local → UTC → market timezone).
  2. Enforce OHLC structural validity.
  3. Produce a fully typed, immutable MarketBar ready for all downstream use.

This module does NOT validate data quality (gaps, spikes, staleness).
That is handled by aion.data.validator.

Rules:
- All output timestamps are tz-aware.
- `is_valid=False` is set when OHLC constraints are violated — the bar
  is kept but flagged so consumers can choose to skip it.
- Normalisation is pure / side-effect free.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from aion.core.enums import DataSource, Timeframe
from aion.core.models import InstrumentSpec, MarketBar, RawBar


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def normalize_bar(
    raw: RawBar,
    instrument: InstrumentSpec,
    timeframe: Timeframe,
    *,
    market_tz: ZoneInfo | None = None,
) -> MarketBar:
    """
    Convert a single RawBar to a normalised MarketBar.

    Parameters
    ----------
    raw:
        The source bar, as returned by an adapter.
    instrument:
        Instrument specification that carries timezone names.
    timeframe:
        The timeframe this bar represents.
    market_tz:
        Pre-built ZoneInfo for the market timezone.  If None it is built
        from `instrument.market_timezone`.  Pass it when normalising many
        bars for the same instrument to avoid repeated ZoneInfo lookups.
    """
    if market_tz is None:
        market_tz = ZoneInfo(instrument.market_timezone)

    broker_tz = ZoneInfo(instrument.broker_timezone)

    ts_utc = _to_utc(raw.timestamp, broker_tz)
    ts_market = ts_utc.astimezone(market_tz)
    is_valid = _ohlc_is_valid(raw)

    return MarketBar(
        symbol=raw.symbol,
        timestamp_utc=ts_utc,
        timestamp_market=ts_market,
        timeframe=timeframe,
        open=raw.open,
        high=raw.high,
        low=raw.low,
        close=raw.close,
        tick_volume=raw.tick_volume,
        real_volume=raw.real_volume,
        spread=raw.spread,
        source=raw.source,
        is_valid=is_valid,
    )


def normalize_bars(
    raw_bars: list[RawBar],
    instrument: InstrumentSpec,
    timeframe: Timeframe,
) -> list[MarketBar]:
    """
    Normalise a list of RawBar objects in one call.

    The market ZoneInfo is built once and reused for all bars.
    """
    market_tz = ZoneInfo(instrument.market_timezone)
    return [
        normalize_bar(raw, instrument, timeframe, market_tz=market_tz)
        for raw in raw_bars
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _to_utc(ts: datetime, broker_tz: ZoneInfo) -> datetime:
    """
    Convert a timestamp to UTC.

    - If `ts` is naive, it is first localised to `broker_tz`.
    - If `ts` is already tz-aware, it is converted to UTC directly.
    - The result is always UTC-aware.
    """
    if ts.tzinfo is None:
        # Naive: assume broker timezone.
        # `replace` is correct here because the timestamp is already *in*
        # the broker's local time — we are attaching the label, not shifting.
        ts = ts.replace(tzinfo=broker_tz)

    return ts.astimezone(timezone.utc)


def _ohlc_is_valid(bar: RawBar) -> bool:
    """
    Check that OHLC values satisfy basic structural constraints.

    A bar fails if:
    - Any of OHLC is non-finite (NaN / Inf handled upstream, but guard here)
    - high < low
    - high < open  or  high < close
    - low > open   or  low > close
    - tick_volume < 0
    - spread < 0
    """
    try:
        o, h, l, c = bar.open, bar.high, bar.low, bar.close
    except (TypeError, AttributeError):
        return False

    if not all(_is_finite(v) for v in (o, h, l, c)):
        return False
    if h < l:
        return False
    if h < o or h < c:
        return False
    if l > o or l > c:
        return False
    if bar.tick_volume < 0:
        return False
    if bar.spread < 0:
        return False
    return True


def _is_finite(value: float) -> bool:
    """Return True if value is a finite real number."""
    import math
    try:
        return math.isfinite(value)
    except (TypeError, ValueError):
        return False
