"""
tests/unit/test_snapshots.py
─────────────────────────────
Unit tests for aion.data.snapshots.

Tests verify:
  - Snapshot is built successfully from valid inputs
  - latest_bar is the last M1 bar
  - Bar lists are trimmed to window sizes
  - snapshot_id is generated and non-empty
  - is_usable reflects quality_score >= MIN_QUALITY_SCORE
  - bars_for() dispatches correctly
  - SnapshotError raised on empty or mismatched inputs
  - Feature computation uses full history (not just windowed bars)
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from aion.core.constants import (
    MIN_QUALITY_SCORE,
    SNAPSHOT_BARS_M1,
    SNAPSHOT_BARS_M5,
    SNAPSHOT_BARS_M15,
)
from aion.core.enums import Timeframe
from aion.data.resampler import resample_bars
from aion.data.snapshots import SnapshotError, build_snapshot
from tests.unit._fixtures import (
    make_bar,
    make_eurusd_spec,
    make_london_session,
    make_off_hours_session,
    make_sequential_bars,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def build_default_snapshot(n_m1: int = 50):
    """Build a complete snapshot with n_m1 M1 bars."""
    instrument = make_eurusd_spec()
    bars_m1 = make_sequential_bars(n_m1)
    bars_m5 = resample_bars(bars_m1, Timeframe.M5)
    bars_m15 = resample_bars(bars_m1, Timeframe.M15)
    ctx = make_off_hours_session()
    return build_snapshot(instrument, bars_m1, bars_m5, bars_m15, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Basic construction
# ─────────────────────────────────────────────────────────────────────────────


def test_snapshot_builds_without_error():
    snap = build_default_snapshot()
    assert snap is not None


def test_snapshot_id_generated():
    snap = build_default_snapshot()
    assert snap.snapshot_id.startswith("snap_")
    assert len(snap.snapshot_id) > 5


def test_snapshot_ids_are_unique():
    """Each build produces a different snapshot_id."""
    snap1 = build_default_snapshot()
    snap2 = build_default_snapshot()
    assert snap1.snapshot_id != snap2.snapshot_id


def test_snapshot_symbol():
    snap = build_default_snapshot()
    assert snap.symbol == "EURUSD"


def test_snapshot_version():
    from aion.core.constants import SNAPSHOT_VERSION
    snap = build_default_snapshot()
    assert snap.snapshot_version == SNAPSHOT_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# Latest bar
# ─────────────────────────────────────────────────────────────────────────────


def test_latest_bar_is_last_m1_bar():
    bars_m1 = make_sequential_bars(50)
    bars_m5 = resample_bars(bars_m1, Timeframe.M5)
    bars_m15 = resample_bars(bars_m1, Timeframe.M15)
    snap = build_snapshot(
        make_eurusd_spec(), bars_m1, bars_m5, bars_m15, make_off_hours_session()
    )
    assert snap.latest_bar == bars_m1[-1]


def test_timestamp_utc_matches_latest_bar():
    snap = build_default_snapshot()
    assert snap.timestamp_utc == snap.latest_bar.timestamp_utc


# ─────────────────────────────────────────────────────────────────────────────
# Window trimming
# ─────────────────────────────────────────────────────────────────────────────


def test_bars_m1_trimmed_to_window():
    """When n_m1 > SNAPSHOT_BARS_M1, the snapshot stores only the last N."""
    n = SNAPSHOT_BARS_M1 + 20
    bars_m1 = make_sequential_bars(n)
    bars_m5 = resample_bars(bars_m1, Timeframe.M5)
    bars_m15 = resample_bars(bars_m1, Timeframe.M15)
    snap = build_snapshot(
        make_eurusd_spec(), bars_m1, bars_m5, bars_m15, make_off_hours_session()
    )
    assert len(snap.bars_m1) == SNAPSHOT_BARS_M1


def test_bars_m1_not_trimmed_when_within_window():
    """When n < SNAPSHOT_BARS_M1, all bars are kept."""
    n = 30
    assert n < SNAPSHOT_BARS_M1
    snap = build_default_snapshot(n_m1=n)
    assert len(snap.bars_m1) == n


def test_bars_m1_last_bar_is_latest():
    """After trimming, the last bar in bars_m1 must still be the latest bar."""
    snap = build_default_snapshot(n_m1=SNAPSHOT_BARS_M1 + 10)
    assert snap.bars_m1[-1] == snap.latest_bar


def test_custom_window_size_respected():
    bars_m1 = make_sequential_bars(50)
    bars_m5 = resample_bars(bars_m1, Timeframe.M5)
    bars_m15 = resample_bars(bars_m1, Timeframe.M15)
    snap = build_snapshot(
        make_eurusd_spec(),
        bars_m1, bars_m5, bars_m15,
        make_off_hours_session(),
        window_m1=20,
        window_m5=5,
    )
    assert len(snap.bars_m1) == 20


# ─────────────────────────────────────────────────────────────────────────────
# bars_for() dispatch
# ─────────────────────────────────────────────────────────────────────────────


def test_bars_for_m1_returns_bars_m1():
    snap = build_default_snapshot()
    assert snap.bars_for(Timeframe.M1) == snap.bars_m1


def test_bars_for_m5_returns_bars_m5():
    snap = build_default_snapshot()
    assert snap.bars_for(Timeframe.M5) == snap.bars_m5


def test_bars_for_m15_returns_bars_m15():
    snap = build_default_snapshot()
    assert snap.bars_for(Timeframe.M15) == snap.bars_m15


def test_bars_for_unsupported_timeframe_raises():
    snap = build_default_snapshot()
    with pytest.raises(ValueError, match="not available"):
        snap.bars_for(Timeframe.D1)


# ─────────────────────────────────────────────────────────────────────────────
# is_usable
# ─────────────────────────────────────────────────────────────────────────────


def test_is_usable_with_perfect_data():
    snap = build_default_snapshot(n_m1=50)
    assert snap.is_usable is True


def test_quality_report_included():
    snap = build_default_snapshot()
    assert snap.quality_report is not None
    assert snap.quality_report.rows_checked > 0


def test_quality_score_above_threshold_for_clean_data():
    snap = build_default_snapshot()
    assert snap.quality_report.quality_score >= MIN_QUALITY_SCORE


# ─────────────────────────────────────────────────────────────────────────────
# Feature vector
# ─────────────────────────────────────────────────────────────────────────────


def test_feature_vector_included():
    snap = build_default_snapshot()
    assert snap.feature_vector is not None
    assert snap.feature_vector.symbol == "EURUSD"


def test_feature_vector_computed_from_full_history_not_window():
    """
    When we have 120 M1 bars (> window of 100), the feature vector should
    reflect the full 120-bar history (e.g. ATR-14 is not None).

    If features were computed from the windowed bars_m1, and window_m1=10,
    then ATR-14 would still be non-None (10 > ATR_PERIOD? No — 14 > 10).
    So we use a tiny window to force the distinction.
    """
    from aion.core.constants import ATR_PERIOD

    n = ATR_PERIOD + 5  # enough for ATR
    bars_m1 = make_sequential_bars(n)
    bars_m5 = resample_bars(bars_m1, Timeframe.M5)
    bars_m15 = resample_bars(bars_m1, Timeframe.M15)

    # Window smaller than ATR_PERIOD — if features used windowed bars, ATR would be None
    small_window = ATR_PERIOD - 1

    snap = build_snapshot(
        make_eurusd_spec(),
        bars_m1, bars_m5, bars_m15,
        make_off_hours_session(),
        window_m1=small_window,
    )

    # bars_m1 in snapshot is trimmed to small_window
    assert len(snap.bars_m1) == small_window

    # But the feature vector was computed on full history → ATR should be available
    assert snap.feature_vector.atr_14 is not None, (
        "ATR-14 should be computed from full history, not the trimmed window"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Error cases
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_m1_raises_snapshot_error():
    with pytest.raises(SnapshotError, match="empty"):
        build_snapshot(
            make_eurusd_spec(), [], [], [], make_off_hours_session()
        )


def test_symbol_mismatch_in_m1_raises():
    bars_m1 = make_sequential_bars(10)
    wrong_bar = make_bar(offset_minutes=10, symbol="GBPUSD")
    bars_with_mismatch = bars_m1 + [wrong_bar]
    with pytest.raises(SnapshotError, match="mismatch"):
        build_snapshot(
            make_eurusd_spec(),
            bars_with_mismatch, [], [],
            make_off_hours_session()
        )


def test_wrong_timeframe_in_m1_raises():
    """Passing M5 bars as bars_m1 should raise a SnapshotError."""
    bars_m1 = make_sequential_bars(10)
    wrong_tf_bars = [
        make_bar(offset_minutes=i * 5, timeframe=Timeframe.M5) for i in range(3)
    ]
    with pytest.raises(SnapshotError, match="timeframe"):
        build_snapshot(
            make_eurusd_spec(),
            wrong_tf_bars, [], [],
            make_off_hours_session()
        )


# ─────────────────────────────────────────────────────────────────────────────
# Session context flows through
# ─────────────────────────────────────────────────────────────────────────────


def test_session_context_preserved_in_snapshot():
    from aion.core.enums import SessionName
    ctx = make_london_session()
    bars_m1 = make_sequential_bars(20)
    bars_m5 = resample_bars(bars_m1, Timeframe.M5)
    bars_m15 = resample_bars(bars_m1, Timeframe.M15)
    snap = build_snapshot(make_eurusd_spec(), bars_m1, bars_m5, bars_m15, ctx)
    assert snap.session_context.session_name == SessionName.LONDON


def test_instrument_preserved_in_snapshot():
    spec = make_eurusd_spec()
    snap = build_default_snapshot()
    assert snap.instrument.symbol == spec.symbol
    assert snap.instrument.tick_size == spec.tick_size
