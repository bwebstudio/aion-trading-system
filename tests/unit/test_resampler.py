"""
tests/unit/test_resampler.py
─────────────────────────────
Unit tests for aion.data.resampler.

Tests verify:
  - OHLCV aggregation correctness (exact numeric results)
  - Spread policy (mean)
  - Timeframe label on output bars
  - Edge cases: empty input, non-M1 source, unsupported target
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from aion.core.enums import DataSource, Timeframe
from aion.core.models import MarketBar
from aion.data.resampler import ResamplerError, resample_bars
from tests.unit._fixtures import BASE_TS, make_bar


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_5_m1_bars() -> list[MarketBar]:
    """
    Five M1 bars starting at BASE_TS with known OHLCV values.

    bar 0: O=1.1000 H=1.1010 L=1.0990 C=1.1005  vol=100 spread=2.0
    bar 1: O=1.1005 H=1.1015 L=1.1000 C=1.1008  vol=200 spread=3.0
    bar 2: O=1.1008 H=1.1025 L=1.1003 C=1.1015  vol=150 spread=1.0
    bar 3: O=1.1015 H=1.1030 L=1.1008 C=1.1020  vol=120 spread=2.5
    bar 4: O=1.1020 H=1.1035 L=1.1015 C=1.1028  vol=180 spread=2.0

    Expected M5 aggregate:
      open        = 1.1000  (bar 0's open)
      high        = 1.1035  (max of all highs)
      low         = 1.0990  (min of all lows)
      close       = 1.1028  (bar 4's close)
      tick_volume = 750     (100+200+150+120+180)
      real_volume = 0
      spread      = 2.1     (mean: (2+3+1+2.5+2)/5)
    """
    specs = [
        (1.1000, 1.1010, 1.0990, 1.1005, 100.0, 2.0),
        (1.1005, 1.1015, 1.1000, 1.1008, 200.0, 3.0),
        (1.1008, 1.1025, 1.1003, 1.1015, 150.0, 1.0),
        (1.1015, 1.1030, 1.1008, 1.1020, 120.0, 2.5),
        (1.1020, 1.1035, 1.1015, 1.1028, 180.0, 2.0),
    ]
    bars = []
    for i, (o, h, lo, c, vol, sp) in enumerate(specs):
        bars.append(
            make_bar(
                offset_minutes=i,
                open=o,
                high=h,
                low=lo,
                close=c,
                tick_volume=vol,
                spread=sp,
                timeframe=Timeframe.M1,
            )
        )
    return bars


# ─────────────────────────────────────────────────────────────────────────────
# M1 → M5
# ─────────────────────────────────────────────────────────────────────────────


def test_m1_to_m5_produces_one_bar():
    bars = make_5_m1_bars()
    result = resample_bars(bars, Timeframe.M5)
    assert len(result) == 1


def test_m1_to_m5_open_is_first():
    bars = make_5_m1_bars()
    result = resample_bars(bars, Timeframe.M5)
    assert result[0].open == pytest.approx(1.1000)


def test_m1_to_m5_high_is_max():
    bars = make_5_m1_bars()
    result = resample_bars(bars, Timeframe.M5)
    assert result[0].high == pytest.approx(1.1035)


def test_m1_to_m5_low_is_min():
    bars = make_5_m1_bars()
    result = resample_bars(bars, Timeframe.M5)
    assert result[0].low == pytest.approx(1.0990)


def test_m1_to_m5_close_is_last():
    bars = make_5_m1_bars()
    result = resample_bars(bars, Timeframe.M5)
    assert result[0].close == pytest.approx(1.1028)


def test_m1_to_m5_tick_volume_is_sum():
    bars = make_5_m1_bars()
    result = resample_bars(bars, Timeframe.M5)
    assert result[0].tick_volume == pytest.approx(750.0)


def test_m1_to_m5_spread_is_mean():
    """Spread policy: mean of all M1 spreads in the period."""
    bars = make_5_m1_bars()
    result = resample_bars(bars, Timeframe.M5)
    expected_mean = (2.0 + 3.0 + 1.0 + 2.5 + 2.0) / 5
    assert result[0].spread == pytest.approx(expected_mean)


def test_m1_to_m5_timeframe_label():
    bars = make_5_m1_bars()
    result = resample_bars(bars, Timeframe.M5)
    assert result[0].timeframe == Timeframe.M5


def test_m1_to_m5_symbol_preserved():
    bars = make_5_m1_bars()
    result = resample_bars(bars, Timeframe.M5)
    assert result[0].symbol == "EURUSD"


def test_m1_to_m5_source_preserved():
    bars = make_5_m1_bars()
    result = resample_bars(bars, Timeframe.M5)
    assert result[0].source == DataSource.CSV


def test_m1_to_m5_is_valid_true():
    """Resampled bars are always marked valid."""
    bars = make_5_m1_bars()
    result = resample_bars(bars, Timeframe.M5)
    assert all(b.is_valid for b in result)


def test_m1_to_m5_timestamp_utc_is_period_open():
    """The resampled bar timestamp equals the opening timestamp of the period."""
    bars = make_5_m1_bars()
    result = resample_bars(bars, Timeframe.M5)
    # Bar starts at BASE_TS (08:00 UTC) — first M1 bar's timestamp
    assert result[0].timestamp_utc == BASE_TS


# ─────────────────────────────────────────────────────────────────────────────
# M1 → M15
# ─────────────────────────────────────────────────────────────────────────────


def test_m1_to_m15_aggregates_15_bars():
    """15 sequential M1 bars should produce exactly 1 M15 bar."""
    bars = [make_bar(offset_minutes=i) for i in range(15)]
    result = resample_bars(bars, Timeframe.M15)
    assert len(result) == 1
    assert result[0].timeframe == Timeframe.M15


def test_m1_to_m15_partial_window_included():
    """
    12 M1 bars (incomplete M15 period) should still produce 1 bar.
    The resampler includes incomplete last windows — caller discards if needed.
    """
    bars = [make_bar(offset_minutes=i) for i in range(12)]
    result = resample_bars(bars, Timeframe.M15)
    assert len(result) == 1


# ─────────────────────────────────────────────────────────────────────────────
# M1 → H1
# ─────────────────────────────────────────────────────────────────────────────


def test_m1_to_h1_aggregates_60_bars():
    bars = [make_bar(offset_minutes=i) for i in range(60)]
    result = resample_bars(bars, Timeframe.H1)
    assert len(result) == 1
    assert result[0].timeframe == Timeframe.H1


def test_m1_to_h1_two_hours():
    """120 M1 bars spanning two hours should produce 2 H1 bars."""
    bars = [make_bar(offset_minutes=i) for i in range(120)]
    result = resample_bars(bars, Timeframe.H1)
    assert len(result) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_input_returns_empty():
    result = resample_bars([], Timeframe.M5)
    assert result == []


def test_single_m1_bar_resampled_to_m5():
    """A single M1 bar produces one M5 bar (incomplete period — included)."""
    bars = [make_bar()]
    result = resample_bars(bars, Timeframe.M5)
    assert len(result) == 1
    assert result[0].open == bars[0].open
    assert result[0].close == bars[0].close


def test_non_m1_source_raises():
    bars = [make_bar(timeframe=Timeframe.M5)]
    with pytest.raises(ResamplerError, match="M1"):
        resample_bars(bars, Timeframe.M15)


def test_unsupported_target_raises():
    bars = make_5_m1_bars()
    with pytest.raises(ResamplerError, match="Unsupported"):
        resample_bars(bars, Timeframe.D1)


def test_same_timeframe_raises():
    bars = make_5_m1_bars()
    with pytest.raises(ResamplerError):
        resample_bars(bars, Timeframe.M1)


# ─────────────────────────────────────────────────────────────────────────────
# No lookahead guarantee
# ─────────────────────────────────────────────────────────────────────────────


def test_resampled_open_does_not_use_future_bar():
    """
    The open of a resampled bar must be the FIRST M1 bar's open,
    not any future bar's price.  We change the 5th bar to an extreme
    price and verify it does not affect 'open'.
    """
    bars = make_5_m1_bars()
    # Mutate bar 4 (last in period) to have a very different open
    extreme_bar = make_bar(
        offset_minutes=4, open=9.9999, high=9.9999, low=9.9999, close=9.9999
    )
    bars[4] = extreme_bar
    result = resample_bars(bars, Timeframe.M5)
    # open must still be bar[0].open = 1.1000
    assert result[0].open == pytest.approx(1.1000)
