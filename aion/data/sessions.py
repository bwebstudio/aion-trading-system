"""
aion.data.sessions
──────────────────
Build SessionContext from a UTC timestamp.

Design decisions:
- Sessions are defined in their *own* timezone (LOCAL times) and
  converted to UTC on-the-fly using zoneinfo.  This means DST shifts
  in London and New York are handled automatically — no manual offsets.
- Tokyo has no DST, so Asia session is always 00:00–09:00 UTC.
- The "trading day" is the UTC calendar date.
- "Opening range" tracks the first OPENING_RANGE_MINUTES of the primary
  session.  It is computed from the session's UTC open time.
- build_session_context() is pure / side-effect free.

Session windows (approximate, for reference):
  Asia     09:00–18:00 JST  =  00:00–09:00 UTC  (no DST)
  London   08:00–16:30 BST/GMT  ≈  07:00/08:00–15:30/16:30 UTC
  New York 09:30–17:00 EDT/EST  ≈  13:30/14:30–21:00/22:00 UTC
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

from aion.core.constants import OPENING_RANGE_MINUTES, SESSION_DEFINITIONS
from aion.core.enums import SessionName
from aion.core.models import SessionContext


# ─────────────────────────────────────────────────────────────────────────────
# Internal session definition
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _SessionDef:
    name: SessionName
    tz: ZoneInfo
    open_time: time   # in the session's own timezone
    close_time: time  # in the session's own timezone


def _build_session_defs() -> list[_SessionDef]:
    """Build _SessionDef list from the constants dict, done once at import."""
    defs: list[_SessionDef] = []
    name_map = {
        "ASIA": SessionName.ASIA,
        "LONDON": SessionName.LONDON,
        "NEW_YORK": SessionName.NEW_YORK,
    }
    for key, cfg in SESSION_DEFINITIONS.items():
        h_open, m_open = cfg["open"]
        h_close, m_close = cfg["close"]
        defs.append(
            _SessionDef(
                name=name_map[key],
                tz=ZoneInfo(cfg["timezone"]),
                open_time=time(h_open, m_open),
                close_time=time(h_close, m_close),
            )
        )
    return defs


_SESSIONS: list[_SessionDef] = _build_session_defs()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def build_session_context(
    timestamp_utc: datetime,
    market_timezone: str,
    broker_timezone: str,
    local_timezone: str,
) -> SessionContext:
    """
    Build the full SessionContext for a given UTC timestamp.

    Parameters
    ----------
    timestamp_utc:
        The moment to evaluate.  Must be timezone-aware (UTC).
    market_timezone:
        IANA name for the instrument's market/exchange timezone.
    broker_timezone:
        IANA name for the broker's server timezone.
    local_timezone:
        IANA name for the operator's local timezone (display only).

    Returns
    -------
    SessionContext
        Fully populated, immutable.
    """
    if timestamp_utc.tzinfo is None:
        raise ValueError("timestamp_utc must be timezone-aware")

    mtz = ZoneInfo(market_timezone)
    btz = ZoneInfo(broker_timezone)
    ltz = ZoneInfo(local_timezone)

    broker_time = timestamp_utc.astimezone(btz)
    market_time = timestamp_utc.astimezone(mtz)
    local_time = timestamp_utc.astimezone(ltz)

    trading_day: date = timestamp_utc.date()

    # Determine which sessions are active
    active = _active_sessions(timestamp_utc)
    is_asia = SessionName.ASIA in active
    is_london = SessionName.LONDON in active
    is_new_york = SessionName.NEW_YORK in active

    # Primary session label
    session_name = _primary_session(is_asia, is_london, is_new_york)
    is_open = session_name != SessionName.OFF_HOURS

    # Opening range
    session_open_utc: datetime | None = None
    session_close_utc: datetime | None = None
    opening_range_active = False
    opening_range_completed = False

    if is_open:
        primary_def = _session_def_for(session_name)
        if primary_def is not None:
            session_open_utc, session_close_utc = _session_window_utc(
                timestamp_utc, primary_def
            )
            minutes_since_open = (
                (timestamp_utc - session_open_utc).total_seconds() / 60
            )
            opening_range_active = 0 <= minutes_since_open < OPENING_RANGE_MINUTES
            opening_range_completed = minutes_since_open >= OPENING_RANGE_MINUTES

    return SessionContext(
        trading_day=trading_day,
        broker_time=broker_time,
        market_time=market_time,
        local_time=local_time,
        is_asia=is_asia,
        is_london=is_london,
        is_new_york=is_new_york,
        is_session_open_window=is_open,
        opening_range_active=opening_range_active,
        opening_range_completed=opening_range_completed,
        session_name=session_name,
        session_open_utc=session_open_utc,
        session_close_utc=session_close_utc,
    )


def session_open_utc_for(timestamp_utc: datetime, session: SessionName) -> datetime | None:
    """
    Return the UTC open time of the given session on the same day as timestamp_utc.

    Returns None for OFF_HOURS and OVERLAP_LONDON_NY (use LONDON or NEW_YORK).
    """
    defn = _session_def_for(session)
    if defn is None:
        return None
    open_utc, _ = _session_window_utc(timestamp_utc, defn)
    return open_utc


def session_close_utc_for(timestamp_utc: datetime, session: SessionName) -> datetime | None:
    """Return the UTC close time of the given session on the same day as timestamp_utc."""
    defn = _session_def_for(session)
    if defn is None:
        return None
    _, close_utc = _session_window_utc(timestamp_utc, defn)
    return close_utc


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _active_sessions(ts_utc: datetime) -> set[SessionName]:
    """Return the set of sessions active at ts_utc."""
    active: set[SessionName] = set()
    for defn in _SESSIONS:
        open_utc, close_utc = _session_window_utc(ts_utc, defn)
        if open_utc <= ts_utc < close_utc:
            active.add(defn.name)
    return active


def _session_window_utc(
    ts_utc: datetime, defn: _SessionDef
) -> tuple[datetime, datetime]:
    """
    Compute the UTC open and close datetimes for `defn` on the same
    calendar day as ts_utc when expressed in the session's own timezone.

    This is the key DST-correct step:
      1. Convert ts_utc to the session's local timezone to get the local date.
      2. Build open/close datetimes in the session's local timezone.
      3. Convert back to UTC (zoneinfo handles DST automatically).
    """
    ts_local = ts_utc.astimezone(defn.tz)
    local_date = ts_local.date()

    # Attach the session timezone — zoneinfo resolves DST for us
    open_local = datetime.combine(local_date, defn.open_time, tzinfo=defn.tz)
    close_local = datetime.combine(local_date, defn.close_time, tzinfo=defn.tz)

    return open_local.astimezone(timezone.utc), close_local.astimezone(timezone.utc)


def _primary_session(
    is_asia: bool, is_london: bool, is_new_york: bool
) -> SessionName:
    """Resolve the single primary session label for this moment."""
    if is_london and is_new_york:
        return SessionName.OVERLAP_LONDON_NY
    if is_new_york:
        return SessionName.NEW_YORK
    if is_london:
        return SessionName.LONDON
    if is_asia:
        return SessionName.ASIA
    return SessionName.OFF_HOURS


def _session_def_for(session_name: SessionName) -> _SessionDef | None:
    """
    Return the _SessionDef for the given name.

    OVERLAP_LONDON_NY → uses LONDON's open, NEW_YORK's close is not needed
    here; the opening range for the overlap is measured from NY open.
    For simplicity we return the LONDON def for the overlap (it opened first).
    """
    if session_name == SessionName.OFF_HOURS:
        return None

    target = session_name
    if session_name == SessionName.OVERLAP_LONDON_NY:
        # Opening range of overlap = measured from New York open
        target = SessionName.NEW_YORK

    for defn in _SESSIONS:
        if defn.name == target:
            return defn
    return None
