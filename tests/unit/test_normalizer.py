"""
tests/unit/test_normalizer.py
──────────────────────────────
Unit tests for aion.data.normalizer.

Tests verify:
  - Naive timestamps are localised to the broker timezone before UTC conversion
  - UTC-aware timestamps are converted correctly
  - Market timezone conversion produces the expected local time
  - OHLC values are preserved exactly
  - is_valid=False is set when OHLC constraints are violated
  - Spread / tick_volume sign validation
  - normalize_bars processes a list correctly
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import pytest

from aion.core.enums import AssetClass, DataSource, Timeframe
from aion.core.models import InstrumentSpec, RawBar
from aion.data.normalizer import normalize_bar, normalize_bars


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def make_instrument(
    broker_timezone: str = "Etc/UTC",
    market_timezone: str = "Etc/UTC",
) -> InstrumentSpec:
    return InstrumentSpec(
        symbol="EURUSD",
        broker_symbol="EURUSD",
        asset_class=AssetClass.FOREX,
        price_timezone=broker_timezone,
        market_timezone=market_timezone,
        broker_timezone=broker_timezone,
        tick_size=0.00001,
        point_value=10.0,
        contract_size=100_000.0,
        min_lot=0.01,
        lot_step=0.01,
        currency_profit="USD",
        currency_margin="USD",
        session_calendar="forex_standard",
        trading_hours_label="Sun 22:00 - Fri 22:00 UTC",
    )


def make_raw(
    timestamp: datetime,
    open: float = 1.1000,
    high: float = 1.1010,
    low: float = 1.0990,
    close: float = 1.1005,
    tick_volume: float = 100.0,
    real_volume: float = 0.0,
    spread: float = 2.0,
    symbol: str = "EURUSD",
) -> RawBar:
    return RawBar(
        symbol=symbol,
        timestamp=timestamp,
        open=open,
        high=high,
        low=low,
        close=close,
        tick_volume=tick_volume,
        real_volume=real_volume,
        spread=spread,
        source=DataSource.CSV,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Timezone localisation
# ─────────────────────────────────────────────────────────────────────────────


def test_naive_timestamp_localised_to_broker_tz():
    """A naive timestamp is interpreted as broker-local time, then converted to UTC."""
    # Broker is America/New_York (UTC-5 in winter)
    instrument = make_instrument(broker_timezone="America/New_York")
    # 09:30 New_York time = 14:30 UTC (winter, no DST)
    naive_ts = datetime(2024, 1, 15, 9, 30, 0)  # no tzinfo
    raw = make_raw(timestamp=naive_ts)

    bar = normalize_bar(raw, instrument, Timeframe.M1)

    expected_utc = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
    assert bar.timestamp_utc == expected_utc


def test_naive_timestamp_utc_broker_stays_unchanged():
    """If broker is UTC, a naive timestamp should come out as-is in UTC."""
    instrument = make_instrument(broker_timezone="Etc/UTC")
    naive_ts = datetime(2024, 1, 15, 8, 0, 0)
    raw = make_raw(timestamp=naive_ts)

    bar = normalize_bar(raw, instrument, Timeframe.M1)

    expected_utc = datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc)
    assert bar.timestamp_utc == expected_utc


def test_aware_timestamp_converted_to_utc():
    """An already-aware timestamp is correctly converted to UTC regardless of zone."""
    instrument = make_instrument(broker_timezone="Etc/UTC")
    # Timestamp in NY (UTC-5 winter)
    ny_tz = ZoneInfo("America/New_York")
    aware_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=ny_tz)
    raw = make_raw(timestamp=aware_ts)

    bar = normalize_bar(raw, instrument, Timeframe.M1)

    expected_utc = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
    assert bar.timestamp_utc == expected_utc


def test_timestamp_utc_is_always_utc_aware():
    """normalize_bar always returns a UTC-aware timestamp_utc."""
    instrument = make_instrument()
    raw = make_raw(timestamp=datetime(2024, 1, 15, 8, 0, 0))
    bar = normalize_bar(raw, instrument, Timeframe.M1)
    assert bar.timestamp_utc.tzinfo is not None
    assert bar.timestamp_utc.utcoffset().total_seconds() == 0


def test_timestamp_market_is_converted_to_market_tz():
    """timestamp_market is the UTC timestamp expressed in the market timezone."""
    # Market timezone is Europe/London (UTC+1 in summer, UTC+0 in winter)
    instrument = make_instrument(
        broker_timezone="Etc/UTC",
        market_timezone="Europe/London",
    )
    # January = winter → London = UTC
    naive_ts = datetime(2024, 1, 15, 8, 0, 0)
    raw = make_raw(timestamp=naive_ts)
    bar = normalize_bar(raw, instrument, Timeframe.M1)

    london_tz = ZoneInfo("Europe/London")
    expected_market = datetime(2024, 1, 15, 8, 0, 0, tzinfo=london_tz)
    assert bar.timestamp_market == expected_market


def test_timestamp_market_summer_time():
    """In summer (BST = UTC+1), timestamp_market is 1 hour ahead of UTC."""
    instrument = make_instrument(
        broker_timezone="Etc/UTC",
        market_timezone="Europe/London",
    )
    # July: London is BST (UTC+1)
    naive_ts = datetime(2024, 7, 15, 8, 0, 0)
    raw = make_raw(timestamp=naive_ts)
    bar = normalize_bar(raw, instrument, Timeframe.M1)

    # 08:00 UTC → 09:00 BST
    assert bar.timestamp_market.hour == 9
    assert bar.timestamp_market.minute == 0


# ─────────────────────────────────────────────────────────────────────────────
# OHLC preservation
# ─────────────────────────────────────────────────────────────────────────────


def test_ohlc_values_preserved_exactly():
    instrument = make_instrument()
    raw = make_raw(
        timestamp=datetime(2024, 1, 15, 8, 0, 0),
        open=1.2345,
        high=1.2350,
        low=1.2340,
        close=1.2348,
    )
    bar = normalize_bar(raw, instrument, Timeframe.M1)

    assert bar.open == pytest.approx(1.2345)
    assert bar.high == pytest.approx(1.2350)
    assert bar.low == pytest.approx(1.2340)
    assert bar.close == pytest.approx(1.2348)


def test_tick_volume_preserved():
    instrument = make_instrument()
    raw = make_raw(timestamp=datetime(2024, 1, 15, 8, 0, 0), tick_volume=750.0)
    bar = normalize_bar(raw, instrument, Timeframe.M1)
    assert bar.tick_volume == pytest.approx(750.0)


def test_spread_preserved():
    instrument = make_instrument()
    raw = make_raw(timestamp=datetime(2024, 1, 15, 8, 0, 0), spread=3.5)
    bar = normalize_bar(raw, instrument, Timeframe.M1)
    assert bar.spread == pytest.approx(3.5)


def test_symbol_preserved():
    instrument = make_instrument()
    raw = make_raw(timestamp=datetime(2024, 1, 15, 8, 0, 0), symbol="EURUSD")
    bar = normalize_bar(raw, instrument, Timeframe.M1)
    assert bar.symbol == "EURUSD"


def test_timeframe_set_correctly():
    instrument = make_instrument()
    raw = make_raw(timestamp=datetime(2024, 1, 15, 8, 0, 0))
    bar = normalize_bar(raw, instrument, Timeframe.M5)
    assert bar.timeframe == Timeframe.M5


# ─────────────────────────────────────────────────────────────────────────────
# OHLC validity
# ─────────────────────────────────────────────────────────────────────────────


def test_valid_ohlc_sets_is_valid_true():
    instrument = make_instrument()
    raw = make_raw(
        timestamp=datetime(2024, 1, 15, 8, 0, 0),
        open=1.1000,
        high=1.1010,
        low=1.0990,
        close=1.1005,
    )
    bar = normalize_bar(raw, instrument, Timeframe.M1)
    assert bar.is_valid is True


def test_high_below_low_sets_is_valid_false():
    """high < low → structurally invalid."""
    instrument = make_instrument()
    raw = make_raw(
        timestamp=datetime(2024, 1, 15, 8, 0, 0),
        open=1.1000,
        high=1.0980,   # high < low — invalid
        low=1.1010,
        close=1.1005,
    )
    bar = normalize_bar(raw, instrument, Timeframe.M1)
    assert bar.is_valid is False


def test_high_below_open_sets_is_valid_false():
    instrument = make_instrument()
    raw = make_raw(
        timestamp=datetime(2024, 1, 15, 8, 0, 0),
        open=1.1020,
        high=1.1010,   # high < open
        low=1.0990,
        close=1.1005,
    )
    bar = normalize_bar(raw, instrument, Timeframe.M1)
    assert bar.is_valid is False


def test_low_above_close_sets_is_valid_false():
    instrument = make_instrument()
    raw = make_raw(
        timestamp=datetime(2024, 1, 15, 8, 0, 0),
        open=1.1000,
        high=1.1010,
        low=1.1008,   # low > close
        close=1.1005,
    )
    bar = normalize_bar(raw, instrument, Timeframe.M1)
    assert bar.is_valid is False


def test_negative_tick_volume_sets_is_valid_false():
    instrument = make_instrument()
    raw = make_raw(
        timestamp=datetime(2024, 1, 15, 8, 0, 0),
        tick_volume=-10.0,
    )
    bar = normalize_bar(raw, instrument, Timeframe.M1)
    assert bar.is_valid is False


def test_negative_spread_sets_is_valid_false():
    instrument = make_instrument()
    raw = make_raw(
        timestamp=datetime(2024, 1, 15, 8, 0, 0),
        spread=-1.0,
    )
    bar = normalize_bar(raw, instrument, Timeframe.M1)
    assert bar.is_valid is False


def test_doji_bar_is_valid():
    """Open == Close is a valid doji candle."""
    instrument = make_instrument()
    raw = make_raw(
        timestamp=datetime(2024, 1, 15, 8, 0, 0),
        open=1.1000,
        high=1.1005,
        low=1.0995,
        close=1.1000,
    )
    bar = normalize_bar(raw, instrument, Timeframe.M1)
    assert bar.is_valid is True


# ─────────────────────────────────────────────────────────────────────────────
# normalize_bars (batch)
# ─────────────────────────────────────────────────────────────────────────────


def test_normalize_bars_returns_same_count():
    instrument = make_instrument()
    raws = [
        make_raw(datetime(2024, 1, 15, 8, i, 0))
        for i in range(5)
    ]
    bars = normalize_bars(raws, instrument, Timeframe.M1)
    assert len(bars) == 5


def test_normalize_bars_all_utc_aware():
    instrument = make_instrument(broker_timezone="America/New_York")
    raws = [
        make_raw(datetime(2024, 1, 15, 9, i, 0))
        for i in range(3)
    ]
    bars = normalize_bars(raws, instrument, Timeframe.M1)
    assert all(b.timestamp_utc.tzinfo is not None for b in bars)


def test_normalize_bars_market_tz_reused():
    """normalize_bars builds ZoneInfo once and reuses it for all bars."""
    # Not a behavioral test — just verify the batch function returns
    # consistent results matching individual normalize_bar calls.
    instrument = make_instrument(
        broker_timezone="Etc/UTC",
        market_timezone="America/New_York",
    )
    raws = [
        make_raw(datetime(2024, 1, 15, 14, i, 0))
        for i in range(3)
    ]
    bars_batch = normalize_bars(raws, instrument, Timeframe.M1)
    bars_single = [normalize_bar(r, instrument, Timeframe.M1) for r in raws]

    for b_batch, b_single in zip(bars_batch, bars_single):
        assert b_batch.timestamp_utc == b_single.timestamp_utc
        assert b_batch.timestamp_market == b_single.timestamp_market


def test_normalize_bars_empty_input():
    instrument = make_instrument()
    result = normalize_bars([], instrument, Timeframe.M1)
    assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# DST edge cases
# ─────────────────────────────────────────────────────────────────────────────


def test_broker_ny_dst_spring_forward():
    """
    US spring-forward on 2024-03-10: clocks go from 02:00 → 03:00.
    09:30 New_York EDT (summer) = 13:30 UTC (not 14:30).
    """
    instrument = make_instrument(broker_timezone="America/New_York")
    # After US DST spring-forward (March 10, 2024)
    naive_ts = datetime(2024, 3, 11, 9, 30, 0)  # NY summer time
    raw = make_raw(timestamp=naive_ts)
    bar = normalize_bar(raw, instrument, Timeframe.M1)

    expected_utc = datetime(2024, 3, 11, 13, 30, 0, tzinfo=timezone.utc)
    assert bar.timestamp_utc == expected_utc


def test_broker_london_dst_spring_forward():
    """
    UK spring-forward on 2024-03-31: clocks go from 01:00 → 02:00 UTC.
    08:00 BST (summer) = 07:00 UTC (not 08:00).
    """
    instrument = make_instrument(broker_timezone="Europe/London")
    # After UK DST spring-forward (April 1 is already summer)
    naive_ts = datetime(2024, 4, 1, 8, 0, 0)  # BST (UTC+1)
    raw = make_raw(timestamp=naive_ts)
    bar = normalize_bar(raw, instrument, Timeframe.M1)

    expected_utc = datetime(2024, 4, 1, 7, 0, 0, tzinfo=timezone.utc)
    assert bar.timestamp_utc == expected_utc
