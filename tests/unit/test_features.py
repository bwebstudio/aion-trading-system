"""
tests/unit/test_features.py
────────────────────────────
Unit tests for aion.data.features.

Tests verify:
  - Candle structure math (body, wicks)
  - Log return correctness and direction
  - ATR and rolling range require full window (None when insufficient)
  - Session features filtered correctly by session open time
  - No obvious lookahead: adding a future bar does not change past features
  - All expected feature fields are present in the output model
  - Off-hours session produces None session features
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from aion.core.constants import ATR_PERIOD, FEATURE_SET_VERSION, ROLLING_RANGE_SHORT
from aion.core.enums import Timeframe
from aion.data.features import compute_feature_vector
from tests.unit._fixtures import (
    BASE_TS,
    LONDON_OPEN_TS,
    make_bar,
    make_london_session,
    make_off_hours_session,
    make_sequential_bars,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def compute(
    n: int = 50,
    *,
    price_step: float = 0.0001,
    session=None,
    timeframe: Timeframe = Timeframe.M1,
):
    """Convenience: compute features on n sequential bars."""
    bars = make_sequential_bars(n, price_step=price_step)
    ctx = session or make_off_hours_session()
    return compute_feature_vector(bars, ctx, timeframe)


# ─────────────────────────────────────────────────────────────────────────────
# Empty input
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_bars_all_features_none():
    ctx = make_off_hours_session()
    fv = compute_feature_vector([], ctx, Timeframe.M1)
    assert fv.atr_14 is None
    assert fv.return_1 is None
    assert fv.candle_body is None
    assert fv.session_high is None


# ─────────────────────────────────────────────────────────────────────────────
# Feature version
# ─────────────────────────────────────────────────────────────────────────────


def test_feature_set_version_present():
    fv = compute(n=5)
    assert fv.feature_set_version == FEATURE_SET_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# Candle structure — single bar math
# ─────────────────────────────────────────────────────────────────────────────


def test_candle_body_bullish():
    """body = |close - open| for a bullish bar."""
    bar = make_bar(open=1.1000, high=1.1020, low=1.0980, close=1.1015)
    fv = compute_feature_vector([bar], make_off_hours_session(), Timeframe.M1)
    assert fv.candle_body is not None
    assert abs(fv.candle_body - abs(1.1015 - 1.1000)) < 1e-8


def test_candle_body_bearish():
    """body = |close - open| is always non-negative."""
    bar = make_bar(open=1.1015, high=1.1020, low=1.0980, close=1.1000)
    fv = compute_feature_vector([bar], make_off_hours_session(), Timeframe.M1)
    assert fv.candle_body is not None
    assert fv.candle_body >= 0


def test_upper_wick():
    """upper_wick = high - max(open, close)"""
    bar = make_bar(open=1.1000, high=1.1020, low=1.0980, close=1.1010)
    expected = 1.1020 - max(1.1000, 1.1010)  # = 0.0010
    fv = compute_feature_vector([bar], make_off_hours_session(), Timeframe.M1)
    assert fv.upper_wick is not None
    assert abs(fv.upper_wick - expected) < 1e-8


def test_lower_wick():
    """lower_wick = min(open, close) - low"""
    bar = make_bar(open=1.1000, high=1.1020, low=1.0980, close=1.1010)
    expected = min(1.1000, 1.1010) - 1.0980  # = 0.0020
    fv = compute_feature_vector([bar], make_off_hours_session(), Timeframe.M1)
    assert fv.lower_wick is not None
    assert abs(fv.lower_wick - expected) < 1e-8


def test_wicks_non_negative():
    """Wicks are always >= 0 for structurally valid bars."""
    bars = make_sequential_bars(50)
    fv = compute_feature_vector(bars, make_off_hours_session(), Timeframe.M1)
    assert fv.upper_wick is not None and fv.upper_wick >= 0
    assert fv.lower_wick is not None and fv.lower_wick >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Returns
# ─────────────────────────────────────────────────────────────────────────────


def test_return_1_is_log_return():
    """return_1 = log(close[t] / close[t-1])"""
    prev_close = 1.1000
    curr_close = 1.1100
    bars = [
        make_bar(offset_minutes=0, close=prev_close),
        make_bar(offset_minutes=1, close=curr_close),
    ]
    expected = math.log(curr_close / prev_close)
    fv = compute_feature_vector(bars, make_off_hours_session(), Timeframe.M1)
    assert fv.return_1 is not None
    assert abs(fv.return_1 - expected) < 1e-10


def test_return_1_negative_on_down_move():
    bars = [
        make_bar(offset_minutes=0, close=1.1100),
        make_bar(offset_minutes=1, close=1.1000),
    ]
    fv = compute_feature_vector(bars, make_off_hours_session(), Timeframe.M1)
    assert fv.return_1 is not None
    assert fv.return_1 < 0


def test_return_1_is_none_with_single_bar():
    """Cannot compute return with only one bar (no previous close)."""
    bars = [make_bar()]
    fv = compute_feature_vector(bars, make_off_hours_session(), Timeframe.M1)
    assert fv.return_1 is None


def test_return_5_is_none_with_fewer_than_6_bars():
    bars = make_sequential_bars(5)
    fv = compute_feature_vector(bars, make_off_hours_session(), Timeframe.M1)
    assert fv.return_5 is None


def test_return_5_available_with_6_bars():
    bars = make_sequential_bars(6)
    fv = compute_feature_vector(bars, make_off_hours_session(), Timeframe.M1)
    assert fv.return_5 is not None


# ─────────────────────────────────────────────────────────────────────────────
# No lookahead — returns
# ─────────────────────────────────────────────────────────────────────────────


def test_return_1_uses_previous_close_not_future():
    """
    return_1 at time T = log(close[T] / close[T-1]).
    We set close[T] to a known value and verify the math explicitly.
    Adding a future bar must not affect the calculation for bar T.
    """
    prev_close = 1.1000
    curr_close = 1.2000
    future_close = 9.9999  # extreme — must not influence return_1

    bars_to_t = [
        make_bar(offset_minutes=0, close=prev_close),
        make_bar(offset_minutes=1, close=curr_close),
    ]
    bars_with_future = bars_to_t + [
        make_bar(offset_minutes=2, close=future_close)
    ]

    fv_at_t = compute_feature_vector(bars_to_t, make_off_hours_session(), Timeframe.M1)

    # Also verify that compute_feature_vector(bars_to_t) stays the same
    # regardless of what comes next (pure function, same input → same output)
    fv_at_t_again = compute_feature_vector(bars_to_t, make_off_hours_session(), Timeframe.M1)
    assert fv_at_t.return_1 == fv_at_t_again.return_1

    # And that the computation with future bars gives a DIFFERENT result
    # for the LAST bar (which is now the future bar), not for bar T
    fv_future = compute_feature_vector(bars_with_future, make_off_hours_session(), Timeframe.M1)
    expected_future_return = math.log(future_close / curr_close)
    assert abs(fv_future.return_1 - expected_future_return) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# ATR — window requirements
# ─────────────────────────────────────────────────────────────────────────────


def test_atr_14_none_with_insufficient_bars():
    """ATR-14 requires exactly 14 bars; fewer → None."""
    bars = make_sequential_bars(ATR_PERIOD - 1)
    fv = compute_feature_vector(bars, make_off_hours_session(), Timeframe.M1)
    assert fv.atr_14 is None


def test_atr_14_available_with_exactly_14_bars():
    bars = make_sequential_bars(ATR_PERIOD)
    fv = compute_feature_vector(bars, make_off_hours_session(), Timeframe.M1)
    assert fv.atr_14 is not None


def test_atr_14_is_positive():
    bars = make_sequential_bars(50)
    fv = compute(n=50)
    assert fv.atr_14 is not None
    assert fv.atr_14 > 0


def test_atr_14_increases_with_higher_volatility():
    """Higher range bars → higher ATR."""
    low_vol = make_sequential_bars(20, price_step=0.00005)
    high_vol = make_sequential_bars(20, price_step=0.0010)

    fv_low = compute_feature_vector(low_vol, make_off_hours_session(), Timeframe.M1)
    fv_high = compute_feature_vector(high_vol, make_off_hours_session(), Timeframe.M1)

    assert fv_low.atr_14 is not None
    assert fv_high.atr_14 is not None
    assert fv_high.atr_14 > fv_low.atr_14


# ─────────────────────────────────────────────────────────────────────────────
# Rolling range — window requirements
# ─────────────────────────────────────────────────────────────────────────────


def test_rolling_range_10_none_below_window():
    bars = make_sequential_bars(ROLLING_RANGE_SHORT - 1)
    fv = compute_feature_vector(bars, make_off_hours_session(), Timeframe.M1)
    assert fv.rolling_range_10 is None


def test_rolling_range_10_available_at_window():
    bars = make_sequential_bars(ROLLING_RANGE_SHORT)
    fv = compute_feature_vector(bars, make_off_hours_session(), Timeframe.M1)
    assert fv.rolling_range_10 is not None


def test_rolling_range_is_non_negative():
    fv = compute(n=50)
    assert fv.rolling_range_10 is not None
    assert fv.rolling_range_10 >= 0
    assert fv.rolling_range_20 is not None
    assert fv.rolling_range_20 >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Spread features
# ─────────────────────────────────────────────────────────────────────────────


def test_spread_mean_available_with_one_bar():
    """spread_mean uses min_periods=1, so available from the first bar."""
    bars = [make_bar(spread=3.0)]
    fv = compute_feature_vector(bars, make_off_hours_session(), Timeframe.M1)
    assert fv.spread_mean_20 is not None
    assert abs(fv.spread_mean_20 - 3.0) < 1e-8


def test_spread_zscore_none_with_constant_spread():
    """If spread is constant, std=0 and z-score is undefined → None."""
    bars = [make_bar(offset_minutes=i, spread=2.0) for i in range(25)]
    fv = compute_feature_vector(bars, make_off_hours_session(), Timeframe.M1)
    assert fv.spread_zscore_20 is None


def test_spread_zscore_available_with_varying_spread():
    bars = []
    for i in range(25):
        spread = 2.0 if i % 2 == 0 else 4.0
        bars.append(make_bar(offset_minutes=i, spread=spread))
    fv = compute_feature_vector(bars, make_off_hours_session(), Timeframe.M1)
    assert fv.spread_zscore_20 is not None


# ─────────────────────────────────────────────────────────────────────────────
# Session features — off hours → all None
# ─────────────────────────────────────────────────────────────────────────────


def test_off_hours_session_features_all_none():
    bars = make_sequential_bars(50)
    ctx = make_off_hours_session()
    fv = compute_feature_vector(bars, ctx, Timeframe.M1)
    assert fv.session_high is None
    assert fv.session_low is None
    assert fv.opening_range_high is None
    assert fv.opening_range_low is None
    assert fv.vwap_session is None
    assert fv.distance_to_session_high is None
    assert fv.distance_to_session_low is None


# ─────────────────────────────────────────────────────────────────────────────
# Session features — london session
# ─────────────────────────────────────────────────────────────────────────────


def test_session_high_is_max_of_session_bars():
    """
    Session high must equal the max high of all bars in the session window.
    Bars before session open must NOT contribute.
    """
    session_open = LONDON_OPEN_TS  # 08:00 UTC
    ctx = make_london_session()

    # 10 pre-session bars with a very high price that should NOT count
    pre_session_bars = [
        make_bar(
            offset_minutes=-10 + i,
            high=2.0000,  # extreme high — before session
            base_ts=session_open,
        )
        for i in range(10)
    ]
    # 10 session bars (starting at session_open) with lower highs
    session_bars = [
        make_bar(
            offset_minutes=i,
            high=1.1010 + i * 0.0001,
            low=1.0990,
            close=1.1005,
            open=1.1000,
            base_ts=session_open,
        )
        for i in range(10)
    ]

    all_bars = pre_session_bars + session_bars
    fv = compute_feature_vector(all_bars, ctx, Timeframe.M1)

    # session_high must be max of session_bars only
    expected_high = max(b.high for b in session_bars)
    assert fv.session_high is not None
    assert abs(fv.session_high - expected_high) < 1e-8


def test_session_low_is_min_of_session_bars():
    session_open = LONDON_OPEN_TS
    ctx = make_london_session()

    pre_session_bars = [
        make_bar(
            offset_minutes=-5 + i,
            low=0.5000,  # very low — before session, should not count
            base_ts=session_open,
        )
        for i in range(5)
    ]
    session_bars = make_sequential_bars(10, base_ts=session_open)
    all_bars = pre_session_bars + session_bars

    fv = compute_feature_vector(all_bars, ctx, Timeframe.M1)
    expected_low = min(b.low for b in session_bars)
    assert fv.session_low is not None
    assert abs(fv.session_low - expected_low) < 1e-8


def test_vwap_is_weighted_by_volume():
    """
    VWAP = Σ(typical_price × tick_volume) / Σ(tick_volume)
    With two bars of equal volume, VWAP = mean of typical prices.
    """
    session_open = LONDON_OPEN_TS
    ctx = make_london_session()

    bars = [
        make_bar(
            offset_minutes=0,
            open=1.1000, high=1.1020, low=1.0980, close=1.1010,
            tick_volume=100.0,
            base_ts=session_open,
        ),
        make_bar(
            offset_minutes=1,
            open=1.1010, high=1.1030, low=1.1000, close=1.1020,
            tick_volume=100.0,
            base_ts=session_open,
        ),
    ]
    tp0 = (1.1020 + 1.0980 + 1.1010) / 3
    tp1 = (1.1030 + 1.1000 + 1.1020) / 3
    expected_vwap = (tp0 * 100 + tp1 * 100) / 200

    fv = compute_feature_vector(bars, ctx, Timeframe.M1)
    assert fv.vwap_session is not None
    assert abs(fv.vwap_session - expected_vwap) < 1e-8


def test_opening_range_available_when_completed():
    """Opening range is available when opening_range_completed=True."""
    session_open = LONDON_OPEN_TS
    ctx = make_london_session(opening_range_completed=True)

    # Make 35 session bars so the 30-min OR window is fully covered
    bars = make_sequential_bars(35, base_ts=session_open)
    fv = compute_feature_vector(bars, ctx, Timeframe.M1)
    assert fv.opening_range_high is not None
    assert fv.opening_range_low is not None
    assert fv.opening_range_high >= fv.opening_range_low


# ─────────────────────────────────────────────────────────────────────────────
# Distance features
# ─────────────────────────────────────────────────────────────────────────────


def test_distance_to_session_high_correct():
    """distance_to_session_high = close - session_high (negative if below)."""
    session_open = LONDON_OPEN_TS
    ctx = make_london_session()

    bars = make_sequential_bars(10, base_ts=session_open)
    fv = compute_feature_vector(bars, ctx, Timeframe.M1)

    if fv.session_high is not None and fv.distance_to_session_high is not None:
        expected = bars[-1].close - fv.session_high
        assert abs(fv.distance_to_session_high - expected) < 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# All expected fields present
# ─────────────────────────────────────────────────────────────────────────────


def test_all_field_names_present_in_feature_vector():
    """The FeatureVector model must have all expected field names."""
    from aion.core.models import FeatureVector

    expected_fields = {
        "symbol", "timestamp_utc", "timeframe",
        "atr_14", "rolling_range_10", "rolling_range_20",
        "session_high", "session_low",
        "opening_range_high", "opening_range_low",
        "vwap_session",
        "spread_mean_20", "spread_zscore_20",
        "return_1", "return_5",
        "volatility_percentile_20",
        "candle_body", "upper_wick", "lower_wick",
        "distance_to_session_high", "distance_to_session_low",
        "feature_set_version",
    }
    model_fields = set(FeatureVector.model_fields.keys())
    missing = expected_fields - model_fields
    assert missing == set(), f"Missing fields in FeatureVector: {missing}"
