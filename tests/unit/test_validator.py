"""
tests/unit/test_validator.py
─────────────────────────────
Unit tests for aion.data.validator.

Each test covers a single check type.
The goal is to verify: detection works, counts are correct, warnings are
informative, and quality_score reflects the severity of problems.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from aion.core.constants import MIN_QUALITY_SCORE, STALE_BAR_CONSECUTIVE_THRESHOLD
from aion.core.enums import Timeframe
from aion.data.validator import (
    _count_duplicates,
    _count_invalid_ohlc,
    _count_missing_bars,
    _count_negative_spreads,
    _count_null_bars,
    _count_out_of_order,
    _count_spikes,
    _count_stale_bars,
    validate_bars,
)
from tests.unit._fixtures import BASE_TS, make_bar, make_sequential_bars


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def bars_with_gap(gap_minutes: int = 5) -> list:
    """10 sequential bars with a gap of gap_minutes inserted after bar 4."""
    bars = make_sequential_bars(10)
    # Shift bars 5..9 forward by (gap_minutes - 1) extra minutes
    shifted = []
    for i, bar in enumerate(bars):
        if i >= 5:
            extra = timedelta(minutes=(gap_minutes - 1))
            from aion.core.models import MarketBar

            bar = MarketBar(
                **{
                    **bar.model_dump(),
                    "timestamp_utc": bar.timestamp_utc + extra,
                    "timestamp_market": bar.timestamp_market + extra,
                }
            )
        shifted.append(bar)
    return shifted


# ─────────────────────────────────────────────────────────────────────────────
# validate_bars — empty input
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_bars_returns_zero_quality():
    report = validate_bars([], Timeframe.M1)
    assert report.quality_score == 0.0
    assert report.rows_checked == 0
    assert "No bars" in report.warnings[0]


# ─────────────────────────────────────────────────────────────────────────────
# Perfect data
# ─────────────────────────────────────────────────────────────────────────────


def test_perfect_data_quality_score_is_one():
    bars = make_sequential_bars(50)
    report = validate_bars(bars, Timeframe.M1)

    assert report.quality_score == 1.0
    assert report.duplicate_timestamps == 0
    assert report.out_of_order_rows == 0
    assert report.missing_bars == 0
    assert report.stale_bars == 0
    assert report.spike_bars == 0
    assert report.null_rows == 0
    assert report.warnings == []


def test_perfect_data_above_min_quality_threshold():
    """A clean dataset must be usable for live decisions."""
    bars = make_sequential_bars(50)
    report = validate_bars(bars, Timeframe.M1)
    assert report.quality_score >= MIN_QUALITY_SCORE


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate timestamps
# ─────────────────────────────────────────────────────────────────────────────


def test_duplicate_timestamps_detected():
    bars = make_sequential_bars(10)
    # Insert a copy of bar 3 at position 4
    bars.insert(4, bars[3])
    report = validate_bars(bars, Timeframe.M1)
    assert report.duplicate_timestamps >= 1


def test_duplicate_timestamps_lowers_quality():
    bars = make_sequential_bars(20)
    # Insert 4 duplicates
    for _ in range(4):
        bars.insert(5, bars[4])
    report = validate_bars(bars, Timeframe.M1)
    assert report.quality_score < 1.0


def test_count_duplicates_direct():
    bars = make_sequential_bars(5)
    bars_with_dups = bars + [bars[2]]  # one extra copy of bar 2
    assert _count_duplicates(bars_with_dups) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Out-of-order rows
# ─────────────────────────────────────────────────────────────────────────────


def test_out_of_order_detected():
    bars = make_sequential_bars(10)
    # Swap bars 3 and 7 — creates two out-of-order positions
    bars[3], bars[7] = bars[7], bars[3]
    report = validate_bars(bars, Timeframe.M1)
    assert report.out_of_order_rows >= 1


def test_count_out_of_order_direct():
    bars = make_sequential_bars(5)
    reversed_bars = list(reversed(bars))
    # All consecutive pairs are inverted
    assert _count_out_of_order(reversed_bars) == 4


# ─────────────────────────────────────────────────────────────────────────────
# Missing bars (gaps)
# ─────────────────────────────────────────────────────────────────────────────


def test_missing_bars_detected_on_gap():
    """A 5-minute gap in M1 data = 4 missing bars."""
    bars = bars_with_gap(gap_minutes=5)
    report = validate_bars(bars, Timeframe.M1)
    assert report.missing_bars == 4


def test_no_missing_bars_on_sequential_data():
    bars = make_sequential_bars(20)
    assert _count_missing_bars(bars, Timeframe.M1) == 0


def test_missing_bars_single_bar_series():
    """Cannot detect gaps in a single bar."""
    bars = make_sequential_bars(1)
    assert _count_missing_bars(bars, Timeframe.M1) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Invalid OHLC
# ─────────────────────────────────────────────────────────────────────────────


def test_invalid_ohlc_bar_counted():
    bars = make_sequential_bars(5)
    invalid = make_bar(offset_minutes=99, is_valid=False)
    bars.append(invalid)
    report = validate_bars(bars, Timeframe.M1)
    # The invalid bar should be counted, but the valid bars should not
    assert report.rows_checked == 6
    # quality_score should be less than 1.0 due to the invalid bar
    assert report.quality_score < 1.0


def test_count_invalid_ohlc_uses_is_valid_flag():
    bars = [
        make_bar(offset_minutes=0, is_valid=True),
        make_bar(offset_minutes=1, is_valid=False),
        make_bar(offset_minutes=2, is_valid=False),
    ]
    assert _count_invalid_ohlc(bars) == 2


def test_all_invalid_bars_gives_low_quality():
    bars = [make_bar(offset_minutes=i, is_valid=False) for i in range(10)]
    report = validate_bars(bars, Timeframe.M1)
    assert report.quality_score < 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Negative spreads
# ─────────────────────────────────────────────────────────────────────────────


def test_negative_spread_detected():
    bars = make_sequential_bars(10)
    bad = make_bar(offset_minutes=5, spread=-1.0)
    bars[5] = bad
    report = validate_bars(bars, Timeframe.M1)
    assert any("negative spread" in w.lower() for w in report.warnings)
    assert report.quality_score < 1.0


def test_count_negative_spreads_direct():
    bars = [
        make_bar(offset_minutes=0, spread=2.0),
        make_bar(offset_minutes=1, spread=-1.0),
        make_bar(offset_minutes=2, spread=0.0),  # zero spread is OK
    ]
    assert _count_negative_spreads(bars) == 1


def test_zero_spread_not_flagged():
    """Zero spread is valid (some instruments / demo accounts)."""
    bars = make_sequential_bars(5)
    bars[2] = make_bar(offset_minutes=2, spread=0.0)
    report = validate_bars(bars, Timeframe.M1)
    assert all("negative" not in w.lower() for w in report.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Stale bars
# ─────────────────────────────────────────────────────────────────────────────


def test_stale_bars_detected():
    bars = make_sequential_bars(5)
    # Append THRESHOLD consecutive identical bars
    stale_price = 1.1050
    for i in range(STALE_BAR_CONSECUTIVE_THRESHOLD):
        bars.append(
            make_bar(
                offset_minutes=5 + i,
                open=stale_price,
                high=stale_price,
                low=stale_price,
                close=stale_price,
            )
        )
    report = validate_bars(bars, Timeframe.M1)
    assert report.stale_bars >= STALE_BAR_CONSECUTIVE_THRESHOLD
    assert any("stale" in w.lower() for w in report.warnings)


def test_two_identical_bars_not_stale():
    """A run shorter than THRESHOLD must not be flagged."""
    bars = make_sequential_bars(5)
    bars.append(make_bar(offset_minutes=5, close=bars[-1].close))
    bars.append(make_bar(offset_minutes=6, close=bars[-1].close))
    # Only 2 identical bars — below threshold of 3
    assert _count_stale_bars(bars[-2:]) == 0


def test_count_stale_bars_direct_counts_full_sequence():
    price = 1.1000
    identical = [
        make_bar(
            offset_minutes=i,
            open=price,
            high=price,
            low=price,
            close=price,
        )
        for i in range(5)
    ]
    # 5 identical bars — all should be counted
    count = _count_stale_bars(identical)
    assert count == 5


# ─────────────────────────────────────────────────────────────────────────────
# Spikes
# ─────────────────────────────────────────────────────────────────────────────


def test_spike_detected():
    """
    Create 25 normal bars with small range, then one bar with 15x mean range.
    The spike multiplier is 10 so this should be flagged.
    """
    normal_bars = make_sequential_bars(25)
    # Mean range of normal bars is tiny (price_step * 1.2 ≈ 0.00012)
    # Spike range = 0.0200 (much larger than 10 * 0.00012 = 0.0012)
    spike = make_bar(
        offset_minutes=25,
        open=1.1050,
        high=1.1250,   # range = 0.0200
        low=1.1050,
        close=1.1100,
    )
    bars = normal_bars + [spike]
    report = validate_bars(bars, Timeframe.M1)
    assert report.spike_bars >= 1
    assert any("spike" in w.lower() for w in report.warnings)


def test_no_spike_on_normal_data():
    bars = make_sequential_bars(50)
    assert _count_spikes(bars) == 0


def test_spike_requires_lookback_window():
    """Fewer than LOOKBACK bars → no spike detection (not enough history)."""
    bars = make_sequential_bars(5)
    # Even with a huge range, no spike flagged with < 20 bars of history
    spike = make_bar(
        offset_minutes=5,
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.0,
    )
    bars.append(spike)
    assert _count_spikes(bars) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Null / zero-price bars
# ─────────────────────────────────────────────────────────────────────────────


def test_zero_price_bar_counted_as_null():
    bars = make_sequential_bars(5)
    zero_bar = make_bar(offset_minutes=5, open=0.0, high=0.0, low=0.0, close=0.0)
    bars.append(zero_bar)
    count = _count_null_bars(bars)
    assert count == 1


def test_quality_score_formula_documented_correctly():
    """
    With 1 duplicate inserted in 10 bars (n=11 after insert):
      _count_duplicates  → 1  (extra occurrence of same timestamp)
      _count_out_of_order → 1  (equal consecutive timestamps satisfy >= condition)

      penalty = (0.8 × 1 + 0.5 × 1) / 11 = 1.3 / 11
      score   = 1.0 - 1.3 / 11
    """
    bars = make_sequential_bars(10)
    bars.insert(5, bars[4])  # 1 duplicate → also triggers out-of-order (equal ts)

    report = validate_bars(bars, Timeframe.M1)
    # duplicates=1 (weight 0.8), out_of_order=1 (weight 0.5), n=11
    expected_score = max(0.0, 1.0 - (0.8 * 1 + 0.5 * 1) / 11)
    assert abs(report.quality_score - expected_score) < 1e-9
