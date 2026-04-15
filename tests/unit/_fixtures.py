"""
tests/unit/_fixtures.py
────────────────────────
Shared factory helpers for unit tests.

NOT a pytest conftest — kept as a plain module so imports are explicit
and each test file is self-contained and readable.
"""

from __future__ import annotations

from datetime import date, datetime, timezone, timedelta

from aion.core.enums import AssetClass, DataSource, SessionName, Timeframe
from aion.core.models import (
    InstrumentSpec,
    MarketBar,
    SessionContext,
)


# ─────────────────────────────────────────────────────────────────────────────
# Base timestamps
# ─────────────────────────────────────────────────────────────────────────────

BASE_TS = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)
"""A Monday 08:00 UTC — inside both Asia and the pre-London window."""

LONDON_OPEN_TS = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)
"""Approximate London session open in UTC (winter: no DST)."""


# ─────────────────────────────────────────────────────────────────────────────
# MarketBar factory
# ─────────────────────────────────────────────────────────────────────────────


def make_bar(
    offset_minutes: int = 0,
    *,
    symbol: str = "EURUSD",
    open: float = 1.1000,
    high: float = 1.1010,
    low: float = 1.0990,
    close: float = 1.1005,
    tick_volume: float = 100.0,
    real_volume: float = 0.0,
    spread: float = 2.0,
    source: DataSource = DataSource.CSV,
    is_valid: bool = True,
    timeframe: Timeframe = Timeframe.M1,
    base_ts: datetime = BASE_TS,
) -> MarketBar:
    """Factory for a single MarketBar at BASE_TS + offset_minutes."""
    ts = base_ts + timedelta(minutes=offset_minutes)
    return MarketBar(
        symbol=symbol,
        timestamp_utc=ts,
        timestamp_market=ts,
        timeframe=timeframe,
        open=open,
        high=high,
        low=low,
        close=close,
        tick_volume=tick_volume,
        real_volume=real_volume,
        spread=spread,
        source=source,
        is_valid=is_valid,
    )


def make_sequential_bars(
    n: int,
    *,
    base_price: float = 1.1000,
    price_step: float = 0.0001,
    spread: float = 2.0,
    tick_volume: float = 100.0,
    symbol: str = "EURUSD",
    timeframe: Timeframe = Timeframe.M1,
    base_ts: datetime = BASE_TS,
) -> list[MarketBar]:
    """
    Create `n` sequential bars with a gentle price drift.

    Prices: close[i] = base_price + i * price_step
    OHLC is constructed so all constraints hold.
    """
    bars: list[MarketBar] = []
    for i in range(n):
        c = base_price + i * price_step
        o = c - price_step * 0.4
        h = c + abs(price_step) * 0.6
        lo = o - abs(price_step) * 0.6
        bars.append(
            make_bar(
                offset_minutes=i,
                symbol=symbol,
                open=round(o, 5),
                high=round(h, 5),
                low=round(lo, 5),
                close=round(c, 5),
                spread=spread,
                tick_volume=tick_volume,
                timeframe=timeframe,
                base_ts=base_ts,
            )
        )
    return bars


# ─────────────────────────────────────────────────────────────────────────────
# InstrumentSpec factory
# ─────────────────────────────────────────────────────────────────────────────


def make_eurusd_spec() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="EURUSD",
        broker_symbol="EURUSD",
        asset_class=AssetClass.FOREX,
        price_timezone="Etc/UTC",
        market_timezone="Etc/UTC",
        broker_timezone="Etc/UTC",
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


# ─────────────────────────────────────────────────────────────────────────────
# SessionContext factories
# ─────────────────────────────────────────────────────────────────────────────


def make_off_hours_session(ts_utc: datetime | None = None) -> SessionContext:
    """Returns an OFF_HOURS session context (no open/close times)."""
    ts = ts_utc or BASE_TS
    return SessionContext(
        trading_day=ts.date(),
        broker_time=ts,
        market_time=ts,
        local_time=ts,
        is_asia=False,
        is_london=False,
        is_new_york=False,
        is_session_open_window=False,
        opening_range_active=False,
        opening_range_completed=False,
        session_name=SessionName.OFF_HOURS,
        session_open_utc=None,
        session_close_utc=None,
    )


def make_london_session(
    ts_utc: datetime | None = None,
    *,
    opening_range_active: bool = False,
    opening_range_completed: bool = True,
) -> SessionContext:
    """
    Returns a LONDON session context.

    Default: opening range completed, session open at 08:00 UTC.
    """
    ts = ts_utc or LONDON_OPEN_TS + timedelta(hours=2)
    session_open = LONDON_OPEN_TS
    session_close = LONDON_OPEN_TS + timedelta(hours=8, minutes=30)
    return SessionContext(
        trading_day=ts.date(),
        broker_time=ts,
        market_time=ts,
        local_time=ts,
        is_asia=False,
        is_london=True,
        is_new_york=False,
        is_session_open_window=True,
        opening_range_active=opening_range_active,
        opening_range_completed=opening_range_completed,
        session_name=SessionName.LONDON,
        session_open_utc=session_open,
        session_close_utc=session_close,
    )
