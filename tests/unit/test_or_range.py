"""
tests/unit/test_or_range.py
────────────────────────────
Unit tests for aion.strategies.or_range — parametrisable OR computation.

Covers:
  - SINGLE_CANDLE: match by time + timeframe, no match, range validation
  - CANDLE_BLOCK:  aggregate highs/lows, partial blocks, range validation
  - OpeningRangeConfig: frozen, validators
  - OpeningRangeLevel: all fields populated correctly
  - Edge cases: empty bars, no matching timeframe, midnight boundary
"""

from __future__ import annotations

from datetime import datetime, time, timezone

import pytest

from aion.core.enums import DataSource, Timeframe
from aion.core.models import MarketBar
from aion.strategies.or_range import (
    ORMethod,
    OpeningRangeConfig,
    OpeningRangeLevel,
    compute_opening_range,
)

_UTC = timezone.utc


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _bar(
    hour: int,
    minute: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    tf: Timeframe = Timeframe.M5,
    day: int = 15,
) -> MarketBar:
    ts = datetime(2025, 1, day, hour, minute, 0, tzinfo=_UTC)
    return MarketBar(
        symbol="US100.cash",
        timestamp_utc=ts,
        timestamp_market=ts,
        timeframe=tf,
        open=open_,
        high=high,
        low=low,
        close=close,
        tick_volume=100.0,
        real_volume=0.0,
        spread=2.0,
        source=DataSource.CSV,
    )


def _m1_bar(hour: int, minute: int, high: float, low: float) -> MarketBar:
    """Shorthand for M1 bars with open=low, close=high (bullish)."""
    return _bar(hour, minute, open_=low, high=high, low=low, close=high, tf=Timeframe.M1)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE_CANDLE tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSingleCandle:

    def test_finds_matching_bar(self):
        bars = [
            _bar(13, 25, 21100, 21110, 21090, 21105),
            _bar(13, 30, 21105, 21120, 21095, 21115),  # OR bar
            _bar(13, 35, 21115, 21125, 21110, 21120),
        ]
        config = OpeningRangeConfig(
            method=ORMethod.SINGLE_CANDLE,
            reference_time=time(13, 30),
            candle_timeframe=Timeframe.M5,
        )
        level = compute_opening_range(bars, config)

        assert level is not None
        assert level.or_high == 21120.0
        assert level.or_low == 21095.0
        assert level.range_points == 25.0
        assert level.midpoint == pytest.approx((21120 + 21095) / 2)
        assert level.method == ORMethod.SINGLE_CANDLE
        assert level.source_bars == 1

    def test_returns_none_when_no_match(self):
        bars = [_bar(13, 25, 21100, 21110, 21090, 21105)]
        config = OpeningRangeConfig(
            method=ORMethod.SINGLE_CANDLE,
            reference_time=time(13, 30),
            candle_timeframe=Timeframe.M5,
        )
        assert compute_opening_range(bars, config) is None

    def test_returns_none_wrong_timeframe(self):
        """M1 bar at the right time but config expects M5."""
        bars = [_m1_bar(13, 30, 21120, 21095)]
        config = OpeningRangeConfig(
            method=ORMethod.SINGLE_CANDLE,
            reference_time=time(13, 30),
            candle_timeframe=Timeframe.M5,
        )
        assert compute_opening_range(bars, config) is None

    def test_returns_none_when_range_too_small(self):
        bars = [_bar(13, 30, 21100, 21102, 21099, 21101)]  # range = 3
        config = OpeningRangeConfig(
            method=ORMethod.SINGLE_CANDLE,
            reference_time=time(13, 30),
            candle_timeframe=Timeframe.M5,
            min_range_points=5.0,
        )
        assert compute_opening_range(bars, config) is None

    def test_returns_none_when_range_too_large(self):
        bars = [_bar(13, 30, 21100, 21200, 21050, 21150)]  # range = 150
        config = OpeningRangeConfig(
            method=ORMethod.SINGLE_CANDLE,
            reference_time=time(13, 30),
            candle_timeframe=Timeframe.M5,
            max_range_points=100.0,
        )
        assert compute_opening_range(bars, config) is None

    def test_min_range_zero_allows_any(self):
        bars = [_bar(13, 30, 21100, 21100.5, 21100, 21100.3)]  # range = 0.5
        config = OpeningRangeConfig(
            method=ORMethod.SINGLE_CANDLE,
            reference_time=time(13, 30),
            candle_timeframe=Timeframe.M5,
            min_range_points=0.0,
        )
        assert compute_opening_range(bars, config) is not None

    def test_computed_at_is_bar_timestamp(self):
        bars = [_bar(13, 30, 21100, 21120, 21095, 21110)]
        config = OpeningRangeConfig(
            method=ORMethod.SINGLE_CANDLE,
            reference_time=time(13, 30),
            candle_timeframe=Timeframe.M5,
        )
        level = compute_opening_range(bars, config)
        assert level.computed_at == bars[0].timestamp_utc

    def test_selects_first_match_when_multiple_days(self):
        """Two bars at 13:30 on different days — picks the first."""
        bar_day1 = _bar(13, 30, 21100, 21120, 21090, 21110, day=15)
        bar_day2 = _bar(13, 30, 21200, 21220, 21190, 21210, day=16)
        config = OpeningRangeConfig(
            method=ORMethod.SINGLE_CANDLE,
            reference_time=time(13, 30),
            candle_timeframe=Timeframe.M5,
        )
        level = compute_opening_range([bar_day1, bar_day2], config)
        assert level.or_high == 21120.0  # from day 15

    def test_empty_bars_returns_none(self):
        config = OpeningRangeConfig(
            method=ORMethod.SINGLE_CANDLE,
            reference_time=time(13, 30),
            candle_timeframe=Timeframe.M5,
        )
        assert compute_opening_range([], config) is None

    def test_candle_timeframe_none_returns_none(self):
        bars = [_bar(13, 30, 21100, 21120, 21095, 21110)]
        config = OpeningRangeConfig(
            method=ORMethod.SINGLE_CANDLE,
            reference_time=time(13, 30),
            candle_timeframe=None,
        )
        assert compute_opening_range(bars, config) is None


# ─────────────────────────────────────────────────────────────────────────────
# CANDLE_BLOCK tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCandleBlock:

    def _five_m1_bars(self) -> list[MarketBar]:
        """M1 bars 13:30, 13:31, 13:32, 13:33, 13:34."""
        return [
            _m1_bar(13, 30, 21110, 21100),  # H=21110, L=21100
            _m1_bar(13, 31, 21115, 21105),  # H=21115
            _m1_bar(13, 32, 21120, 21095),  # H=21120, L=21095 ← extremes
            _m1_bar(13, 33, 21112, 21098),
            _m1_bar(13, 34, 21108, 21097),
        ]

    def test_aggregates_block_correctly(self):
        bars = self._five_m1_bars()
        config = OpeningRangeConfig(
            method=ORMethod.CANDLE_BLOCK,
            reference_time=time(13, 30),
            block_duration_minutes=5,
        )
        level = compute_opening_range(bars, config)

        assert level is not None
        assert level.or_high == 21120.0
        assert level.or_low == 21095.0
        assert level.range_points == 25.0
        assert level.method == ORMethod.CANDLE_BLOCK
        assert level.source_bars == 5

    def test_partial_block_still_works(self):
        """Only 3 of 5 expected bars — still computes OR from available bars."""
        bars = [
            _m1_bar(13, 30, 21110, 21100),
            _m1_bar(13, 31, 21115, 21098),
            _m1_bar(13, 32, 21108, 21102),
        ]
        config = OpeningRangeConfig(
            method=ORMethod.CANDLE_BLOCK,
            reference_time=time(13, 30),
            block_duration_minutes=5,
        )
        level = compute_opening_range(bars, config)
        assert level is not None
        assert level.source_bars == 3
        assert level.or_high == 21115.0
        assert level.or_low == 21098.0

    def test_excludes_bars_outside_block(self):
        """Bars before and after the block are excluded."""
        bars = [
            _m1_bar(13, 29, 21200, 21050),  # before block
            _m1_bar(13, 30, 21110, 21100),
            _m1_bar(13, 31, 21112, 21098),
            _m1_bar(13, 35, 21300, 21000),  # after block (>= 13:35)
        ]
        config = OpeningRangeConfig(
            method=ORMethod.CANDLE_BLOCK,
            reference_time=time(13, 30),
            block_duration_minutes=5,
        )
        level = compute_opening_range(bars, config)
        assert level is not None
        assert level.source_bars == 2  # only 13:30, 13:31
        assert level.or_high == 21112.0  # NOT 21300 from the bar at 13:35
        assert level.or_low == 21098.0

    def test_returns_none_no_matching_bars(self):
        bars = [_m1_bar(14, 0, 21110, 21100)]
        config = OpeningRangeConfig(
            method=ORMethod.CANDLE_BLOCK,
            reference_time=time(13, 30),
            block_duration_minutes=5,
        )
        assert compute_opening_range(bars, config) is None

    def test_returns_none_range_too_small(self):
        bars = [
            _m1_bar(13, 30, 21101, 21100),  # range = 1
            _m1_bar(13, 31, 21101.5, 21100.5),
        ]
        config = OpeningRangeConfig(
            method=ORMethod.CANDLE_BLOCK,
            reference_time=time(13, 30),
            block_duration_minutes=5,
            min_range_points=5.0,
        )
        assert compute_opening_range(bars, config) is None

    def test_returns_none_range_too_large(self):
        bars = [
            _m1_bar(13, 30, 21200, 21000),  # range = 200
        ]
        config = OpeningRangeConfig(
            method=ORMethod.CANDLE_BLOCK,
            reference_time=time(13, 30),
            block_duration_minutes=5,
            max_range_points=100.0,
        )
        assert compute_opening_range(bars, config) is None

    def test_filters_by_block_timeframe(self):
        """M5 bars at the right time are ignored when block_timeframe=M1."""
        bars = [
            _bar(13, 30, 21100, 21200, 21050, 21150, tf=Timeframe.M5),  # M5 — wrong TF
            _m1_bar(13, 30, 21110, 21100),  # M1 — correct TF
        ]
        config = OpeningRangeConfig(
            method=ORMethod.CANDLE_BLOCK,
            reference_time=time(13, 30),
            block_duration_minutes=5,
            block_timeframe=Timeframe.M1,
        )
        level = compute_opening_range(bars, config)
        assert level is not None
        assert level.source_bars == 1
        assert level.or_high == 21110.0

    def test_block_duration_none_returns_none(self):
        bars = [_m1_bar(13, 30, 21110, 21100)]
        config = OpeningRangeConfig(
            method=ORMethod.CANDLE_BLOCK,
            reference_time=time(13, 30),
            block_duration_minutes=None,
        )
        assert compute_opening_range(bars, config) is None

    def test_computed_at_is_last_bar_in_block(self):
        bars = [
            _m1_bar(13, 30, 21110, 21100),
            _m1_bar(13, 31, 21115, 21098),
            _m1_bar(13, 32, 21108, 21102),
        ]
        config = OpeningRangeConfig(
            method=ORMethod.CANDLE_BLOCK,
            reference_time=time(13, 30),
            block_duration_minutes=5,
        )
        level = compute_opening_range(bars, config)
        assert level.computed_at == bars[-1].timestamp_utc


# ─────────────────────────────────────────────────────────────────────────────
# timezone_source tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTimezoneSource:

    def test_market_timezone_source(self):
        """When timezone_source='market', match against timestamp_market."""
        from zoneinfo import ZoneInfo

        ny_tz = ZoneInfo("America/New_York")
        ts_utc = datetime(2025, 1, 15, 14, 30, 0, tzinfo=_UTC)
        ts_market = ts_utc.astimezone(ny_tz)  # 09:30 ET

        bar = MarketBar(
            symbol="US100.cash",
            timestamp_utc=ts_utc,
            timestamp_market=ts_market,
            timeframe=Timeframe.M5,
            open=21100, high=21120, low=21095, close=21110,
            tick_volume=100, real_volume=0, spread=2,
            source=DataSource.CSV,
        )

        # Match by market time (09:30 ET), not UTC (14:30)
        config = OpeningRangeConfig(
            method=ORMethod.SINGLE_CANDLE,
            reference_time=time(9, 30),
            timezone_source="market",
            candle_timeframe=Timeframe.M5,
        )
        level = compute_opening_range([bar], config)
        assert level is not None
        assert level.or_high == 21120.0

    def test_broker_timezone_source_default(self):
        """Default timezone_source='broker' matches against timestamp_utc."""
        ts_utc = datetime(2025, 1, 15, 13, 30, 0, tzinfo=_UTC)
        bar = MarketBar(
            symbol="US100.cash",
            timestamp_utc=ts_utc,
            timestamp_market=ts_utc,
            timeframe=Timeframe.M5,
            open=21100, high=21120, low=21095, close=21110,
            tick_volume=100, real_volume=0, spread=2,
            source=DataSource.CSV,
        )
        config = OpeningRangeConfig(
            method=ORMethod.SINGLE_CANDLE,
            reference_time=time(13, 30),
            candle_timeframe=Timeframe.M5,
        )
        level = compute_opening_range([bar], config)
        assert level is not None


# ─────────────────────────────────────────────────────────────────────────────
# Config model tests
# ─────────────────────────────────────────────────────────────────────────────


class TestOpeningRangeConfig:

    def test_frozen(self):
        config = OpeningRangeConfig(
            method=ORMethod.SINGLE_CANDLE,
            reference_time=time(13, 30),
            candle_timeframe=Timeframe.M5,
        )
        with pytest.raises(Exception):
            config.method = ORMethod.CANDLE_BLOCK

    def test_block_duration_must_be_positive(self):
        with pytest.raises(Exception):
            OpeningRangeConfig(
                method=ORMethod.CANDLE_BLOCK,
                reference_time=time(13, 30),
                block_duration_minutes=0,
            )

    def test_defaults(self):
        config = OpeningRangeConfig(
            method=ORMethod.SINGLE_CANDLE,
            reference_time=time(13, 30),
        )
        assert config.timezone_source == "broker"
        assert config.min_range_points == 0.0
        assert config.max_range_points is None
        assert config.block_timeframe == Timeframe.M1


# ─────────────────────────────────────────────────────────────────────────────
# OpeningRangeLevel model tests
# ─────────────────────────────────────────────────────────────────────────────


class TestOpeningRangeLevel:

    def test_frozen(self):
        level = OpeningRangeLevel(
            or_high=21120, or_low=21095, midpoint=21107.5,
            range_points=25, method=ORMethod.SINGLE_CANDLE,
            computed_at=datetime(2025, 1, 15, 13, 30, tzinfo=_UTC),
            source_bars=1,
        )
        with pytest.raises(Exception):
            level.or_high = 99999

    def test_all_fields_populated(self):
        level = OpeningRangeLevel(
            or_high=21120, or_low=21095, midpoint=21107.5,
            range_points=25, method=ORMethod.CANDLE_BLOCK,
            computed_at=datetime(2025, 1, 15, 13, 34, tzinfo=_UTC),
            source_bars=5,
        )
        assert level.or_high == 21120
        assert level.or_low == 21095
        assert level.midpoint == 21107.5
        assert level.range_points == 25
        assert level.method == ORMethod.CANDLE_BLOCK
        assert level.source_bars == 5
