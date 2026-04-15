"""
tests/unit/test_sessions.py
────────────────────────────
Unit tests for aion.data.sessions.

Tests verify:
  - Correct session detection at representative UTC times
  - DST transitions (UK March 31, US March 10, 2024)
  - OFF_HOURS detection for mid-session gaps
  - OVERLAP_LONDON_NY detected when both sessions are active
  - session_open_utc / session_close_utc populated correctly
  - Opening range flags (active / completed)
  - build_session_context raises on naive timestamp
  - session_open_utc_for / session_close_utc_for helpers
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from aion.core.constants import OPENING_RANGE_MINUTES
from aion.core.enums import SessionName
from aion.data.sessions import (
    build_session_context,
    session_close_utc_for,
    session_open_utc_for,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

UTC = timezone.utc
# Standard timezone arguments used for most tests
_TZ = dict(
    market_timezone="Etc/UTC",
    broker_timezone="Etc/UTC",
    local_timezone="Etc/UTC",
)


def ctx(ts: datetime):
    """Build a SessionContext for ts using UTC for all timezone args."""
    return build_session_context(ts, **_TZ)


# ─────────────────────────────────────────────────────────────────────────────
# Guard: naive timestamp raises
# ─────────────────────────────────────────────────────────────────────────────


def test_naive_timestamp_raises():
    naive = datetime(2024, 1, 15, 8, 0, 0)  # no tzinfo
    with pytest.raises(ValueError, match="timezone-aware"):
        build_session_context(naive, **_TZ)


# ─────────────────────────────────────────────────────────────────────────────
# Asia session (no DST — Tokyo)
# ─────────────────────────────────────────────────────────────────────────────
# Asia/Tokyo = UTC+9.  Session 09:00–18:00 JST = 00:00–09:00 UTC.


def test_asia_session_active_at_0200_utc():
    ts = datetime(2024, 1, 15, 2, 0, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.is_asia is True
    assert c.session_name == SessionName.ASIA


def test_asia_session_closed_at_0900_utc():
    """Asia closes at 09:00 UTC — the boundary is exclusive (half-open interval)."""
    ts = datetime(2024, 1, 15, 9, 0, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.is_asia is False


def test_asia_session_open_utc_is_midnight_utc():
    """Asia opens at 00:00 UTC (09:00 JST)."""
    ts = datetime(2024, 1, 15, 4, 0, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.session_open_utc is not None
    assert c.session_open_utc.hour == 0
    assert c.session_open_utc.minute == 0


# ─────────────────────────────────────────────────────────────────────────────
# London session — winter (no BST)
# ─────────────────────────────────────────────────────────────────────────────
# In winter (GMT = UTC), London opens at 08:00 UTC, closes at 16:30 UTC.


def test_london_winter_open_at_0800_utc():
    ts = datetime(2024, 1, 15, 8, 0, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.is_london is True


def test_london_winter_session_name():
    ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.session_name == SessionName.LONDON


def test_london_winter_open_utc_is_0800():
    ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.session_open_utc is not None
    assert c.session_open_utc.hour == 8
    assert c.session_open_utc.minute == 0


def test_london_winter_close_utc_is_1630():
    ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.session_close_utc is not None
    assert c.session_close_utc.hour == 16
    assert c.session_close_utc.minute == 30


def test_london_winter_closed_at_0759_utc():
    """One minute before open — still off hours (or Asia)."""
    ts = datetime(2024, 1, 15, 7, 59, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.is_london is False


def test_london_winter_closed_at_1630_utc():
    """Close is exclusive — 16:30 UTC is no longer London."""
    ts = datetime(2024, 1, 15, 16, 30, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.is_london is False


# ─────────────────────────────────────────────────────────────────────────────
# London session — summer (BST = UTC+1)
# ─────────────────────────────────────────────────────────────────────────────
# After UK spring-forward (2024-03-31), London opens at 07:00 UTC.


def test_london_summer_open_at_0700_utc():
    """After UK DST spring-forward (April 1 = summer): London opens at 07:00 UTC."""
    ts = datetime(2024, 4, 1, 7, 0, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.is_london is True


def test_london_summer_open_utc_is_0700():
    ts = datetime(2024, 4, 1, 9, 0, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.session_open_utc is not None
    assert c.session_open_utc.hour == 7
    assert c.session_open_utc.minute == 0


def test_london_summer_not_open_at_0659_utc():
    """Summer: 06:59 UTC is before London open (07:00 BST = 07:00 UTC → 06:00 UTC... wait)

    BST = UTC+1. 08:00 BST = 07:00 UTC. So 06:59 UTC is before the open.
    """
    ts = datetime(2024, 4, 1, 6, 59, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.is_london is False


def test_london_summer_close_utc_is_1530():
    """16:30 BST = 15:30 UTC in summer."""
    ts = datetime(2024, 4, 1, 10, 0, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.session_close_utc is not None
    assert c.session_close_utc.hour == 15
    assert c.session_close_utc.minute == 30


# ─────────────────────────────────────────────────────────────────────────────
# UK DST transition day: 2024-03-31
# ─────────────────────────────────────────────────────────────────────────────
# UK clocks spring forward at 01:00 UTC on 2024-03-31 (01:00 GMT → 02:00 BST).
# On that day, the London session opens at 07:00 UTC (08:00 BST).


def test_london_dst_transition_day_is_summer():
    """On the day of UK DST spring-forward, London opens at 07:00 UTC (summer rule)."""
    ts = datetime(2024, 3, 31, 7, 0, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.is_london is True
    assert c.session_open_utc.hour == 7  # type: ignore[union-attr]


def test_london_dst_transition_day_before_open():
    """Before the new BST open (06:59 UTC) on transition day — London not yet open."""
    ts = datetime(2024, 3, 31, 6, 59, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.is_london is False


# ─────────────────────────────────────────────────────────────────────────────
# New York session — winter (EST = UTC-5)
# ─────────────────────────────────────────────────────────────────────────────
# Winter: NY opens at 14:30 UTC, closes at 22:00 UTC.


def test_ny_winter_open_at_1430_utc():
    ts = datetime(2024, 1, 15, 14, 30, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.is_new_york is True


def test_ny_winter_open_utc_is_1430():
    ts = datetime(2024, 1, 15, 16, 0, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.session_open_utc is not None
    # In OVERLAP, the anchor is NY
    assert c.session_open_utc.hour == 14
    assert c.session_open_utc.minute == 30


def test_ny_winter_not_open_at_1429_utc():
    ts = datetime(2024, 1, 15, 14, 29, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.is_new_york is False


def test_ny_winter_closed_at_2200_utc():
    """22:00 UTC = 17:00 EST — NY should be closed."""
    ts = datetime(2024, 1, 15, 22, 0, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.is_new_york is False


# ─────────────────────────────────────────────────────────────────────────────
# New York session — summer (EDT = UTC-4)
# ─────────────────────────────────────────────────────────────────────────────
# After US spring-forward (2024-03-10), NY opens at 13:30 UTC.


def test_ny_summer_open_at_1330_utc():
    """After US DST spring-forward: NY opens at 13:30 UTC."""
    ts = datetime(2024, 3, 11, 13, 30, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.is_new_york is True


def test_ny_summer_open_utc_is_1330():
    ts = datetime(2024, 3, 11, 15, 0, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.session_open_utc is not None
    # In OVERLAP_LONDON_NY or NEW_YORK, anchor is NY
    assert c.session_open_utc.hour == 13
    assert c.session_open_utc.minute == 30


def test_ny_summer_not_open_at_1329_utc():
    ts = datetime(2024, 3, 11, 13, 29, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.is_new_york is False


# ─────────────────────────────────────────────────────────────────────────────
# US DST transition day: 2024-03-10
# ─────────────────────────────────────────────────────────────────────────────
# US clocks spring forward at 02:00 AM local on 2024-03-10.
# After the transition, 09:30 EDT = 13:30 UTC.


def test_ny_dst_transition_day_summer_open():
    """On US DST day, NY opens at 13:30 UTC (EDT, not EST)."""
    ts = datetime(2024, 3, 10, 13, 30, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.is_new_york is True
    # Verify open_utc reflects summer time
    open_utc = session_open_utc_for(ts, SessionName.NEW_YORK)
    assert open_utc is not None
    assert open_utc.hour == 13
    assert open_utc.minute == 30


def test_ny_dst_transition_day_before_summer_open():
    """On US DST day, 13:29 UTC is before NY open."""
    ts = datetime(2024, 3, 10, 13, 29, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.is_new_york is False


# ─────────────────────────────────────────────────────────────────────────────
# London–NY overlap
# ─────────────────────────────────────────────────────────────────────────────
# Winter: overlap is 14:30–16:30 UTC.


def test_overlap_detected_in_winter():
    """14:30 UTC winter: both London and NY are open."""
    ts = datetime(2024, 1, 15, 14, 30, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.is_london is True
    assert c.is_new_york is True
    assert c.session_name == SessionName.OVERLAP_LONDON_NY


def test_overlap_session_open_is_ny_open():
    """During overlap, the opening range anchor is NY open."""
    ts = datetime(2024, 1, 15, 15, 0, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.session_name == SessionName.OVERLAP_LONDON_NY
    # session_open_utc should be 14:30 (NY open)
    assert c.session_open_utc is not None
    assert c.session_open_utc.hour == 14
    assert c.session_open_utc.minute == 30


# ─────────────────────────────────────────────────────────────────────────────
# OFF_HOURS
# ─────────────────────────────────────────────────────────────────────────────


def test_off_hours_between_sessions():
    """Between NY close and Asia open (22:00–00:00 UTC), it is OFF_HOURS."""
    ts = datetime(2024, 1, 15, 23, 0, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.session_name == SessionName.OFF_HOURS
    assert c.is_session_open_window is False


def test_off_hours_session_open_utc_is_none():
    ts = datetime(2024, 1, 15, 23, 30, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.session_open_utc is None
    assert c.session_close_utc is None


# ─────────────────────────────────────────────────────────────────────────────
# Opening range
# ─────────────────────────────────────────────────────────────────────────────


def test_opening_range_active_during_first_30_minutes():
    """London winter opens at 08:00 UTC. 08:15 should be inside the opening range."""
    ts = datetime(2024, 1, 15, 8, 15, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.session_name == SessionName.LONDON
    assert c.opening_range_active is True
    assert c.opening_range_completed is False


def test_opening_range_completed_after_30_minutes():
    """08:30 UTC = exactly OPENING_RANGE_MINUTES after London winter open."""
    ts = datetime(2024, 1, 15, 8, 30, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.session_name == SessionName.LONDON
    assert c.opening_range_active is False
    assert c.opening_range_completed is True


def test_opening_range_active_at_open_exactly():
    """At exactly 08:00 UTC (London winter open), opening range is active."""
    ts = datetime(2024, 1, 15, 8, 0, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.opening_range_active is True


def test_opening_range_not_active_in_off_hours():
    ts = datetime(2024, 1, 15, 23, 0, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.opening_range_active is False
    assert c.opening_range_completed is False


# ─────────────────────────────────────────────────────────────────────────────
# Derived fields
# ─────────────────────────────────────────────────────────────────────────────


def test_trading_day_is_utc_date():
    ts = datetime(2024, 1, 15, 23, 0, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.trading_day.year == 2024
    assert c.trading_day.month == 1
    assert c.trading_day.day == 15


def test_broker_time_in_correct_zone():
    """broker_time should reflect the broker_timezone, not UTC."""
    ny_tz_str = "America/New_York"
    ts = datetime(2024, 1, 15, 14, 30, 0, tzinfo=UTC)  # 09:30 EST
    c = build_session_context(
        ts,
        market_timezone="Etc/UTC",
        broker_timezone=ny_tz_str,
        local_timezone="Etc/UTC",
    )
    assert c.broker_time.hour == 9
    assert c.broker_time.minute == 30


def test_is_session_open_window_true_during_session():
    ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
    c = ctx(ts)
    assert c.is_session_open_window is True


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────


def test_session_open_utc_for_london_winter():
    ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
    open_utc = session_open_utc_for(ts, SessionName.LONDON)
    assert open_utc is not None
    assert open_utc.hour == 8
    assert open_utc.minute == 0


def test_session_close_utc_for_london_winter():
    ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
    close_utc = session_close_utc_for(ts, SessionName.LONDON)
    assert close_utc is not None
    assert close_utc.hour == 16
    assert close_utc.minute == 30


def test_session_open_utc_for_off_hours_returns_none():
    ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
    assert session_open_utc_for(ts, SessionName.OFF_HOURS) is None


def test_session_open_utc_for_ny_summer():
    """After US DST: NY opens at 13:30 UTC."""
    ts = datetime(2024, 3, 11, 10, 0, 0, tzinfo=UTC)
    open_utc = session_open_utc_for(ts, SessionName.NEW_YORK)
    assert open_utc is not None
    assert open_utc.hour == 13
    assert open_utc.minute == 30
