"""
aion.data.mt5_adapter
──────────────────────
Load historical OHLCV bars from MetaTrader5.

Status: STUB — ready for implementation.
The interface is final.  The body is filled in once MT5 is available.

MT5 is an optional dependency:
    pip install aion[mt5]

If the `MetaTrader5` package is not installed, importing this module
will NOT raise an error.  Only calling the functions will raise
Mt5AdapterError with a clear message.

Usage (when MT5 is available):
    from aion.data.mt5_adapter import load_mt5_bars, Mt5AdapterError
    bars = load_mt5_bars(
        instrument=spec,
        timeframe=Timeframe.M1,
        date_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
        date_to=datetime(2024, 6, 1, tzinfo=timezone.utc),
        login=12345678,
        password="secret",
        server="ICMarkets-Demo",
    )
"""

from __future__ import annotations

from datetime import datetime

from aion.core.enums import DataSource, Timeframe
from aion.core.models import InstrumentSpec, RawBar

# ─────────────────────────────────────────────────────────────────────────────
# Optional MT5 import
# ─────────────────────────────────────────────────────────────────────────────

try:
    import MetaTrader5 as _mt5  # type: ignore[import]

    _MT5_AVAILABLE = True
except ImportError:
    _mt5 = None  # type: ignore[assignment]
    _MT5_AVAILABLE = False

# MT5 timeframe constants mapped to Aion Timeframe enum
_TIMEFRAME_MAP: dict[Timeframe, int] = {}
if _MT5_AVAILABLE:
    _TIMEFRAME_MAP = {
        Timeframe.M1: _mt5.TIMEFRAME_M1,
        Timeframe.M5: _mt5.TIMEFRAME_M5,
        Timeframe.M15: _mt5.TIMEFRAME_M15,
        Timeframe.M30: _mt5.TIMEFRAME_M30,
        Timeframe.H1: _mt5.TIMEFRAME_H1,
        Timeframe.H4: _mt5.TIMEFRAME_H4,
        Timeframe.D1: _mt5.TIMEFRAME_D1,
    }


class Mt5AdapterError(Exception):
    """Raised for MT5 connection, authentication, or data retrieval problems."""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def load_mt5_bars(
    instrument: InstrumentSpec,
    timeframe: Timeframe,
    date_from: datetime,
    date_to: datetime,
    login: int,
    password: str,
    server: str,
) -> list[RawBar]:
    """
    Download historical bars from MetaTrader5 for the given range.

    Parameters
    ----------
    instrument:
        Full instrument specification.  `broker_symbol` is used for the
        MT5 symbol name.
    timeframe:
        Bar timeframe to request.
    date_from / date_to:
        UTC-aware datetimes that bound the request (inclusive/exclusive
        as per MT5 semantics).
    login / password / server:
        MT5 account credentials.

    Returns
    -------
    list[RawBar]
        Bars in ascending timestamp order.  Timestamps are UTC-aware
        (MT5 returns UTC timestamps for copy_rates_range).

    Raises
    ------
    Mt5AdapterError
        If MT5 is not installed, cannot connect, or returns no data.
    """
    _require_mt5()

    if timeframe not in _TIMEFRAME_MAP:
        raise Mt5AdapterError(f"Unsupported timeframe for MT5: {timeframe}")

    if not _mt5.initialize(login=login, password=password, server=server):
        error = _mt5.last_error()
        raise Mt5AdapterError(f"MT5 initialization failed: {error}")

    try:
        rates = _mt5.copy_rates_range(
            instrument.broker_symbol,
            _TIMEFRAME_MAP[timeframe],
            date_from,
            date_to,
        )
    finally:
        _mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise Mt5AdapterError(
            f"MT5 returned no data for {instrument.broker_symbol} "
            f"{timeframe} {date_from}–{date_to}.  "
            f"Last error: {_mt5.last_error()}"
        )

    return _rates_to_raw_bars(rates, instrument)


def is_mt5_available() -> bool:
    """Return True if the MetaTrader5 package is installed."""
    return _MT5_AVAILABLE


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _require_mt5() -> None:
    if not _MT5_AVAILABLE:
        raise Mt5AdapterError(
            "MetaTrader5 package is not installed.  "
            "Install it with: pip install aion[mt5]"
        )


def _rates_to_raw_bars(rates: object, instrument: InstrumentSpec) -> list[RawBar]:
    """
    Convert a numpy structured array from MT5 to RawBar objects.

    MT5 rate fields: time (POSIX UTC), open, high, low, close,
                     tick_volume, spread, real_volume
    """
    from datetime import timezone as _tz

    bars: list[RawBar] = []
    for rate in rates:  # type: ignore[union-attr]
        ts = datetime.fromtimestamp(int(rate["time"]), tz=_tz.utc)
        bars.append(
            RawBar(
                symbol=instrument.symbol,
                timestamp=ts,
                open=float(rate["open"]),
                high=float(rate["high"]),
                low=float(rate["low"]),
                close=float(rate["close"]),
                tick_volume=float(rate["tick_volume"]),
                real_volume=float(rate["real_volume"]),
                spread=float(rate["spread"]),
                source=DataSource.MT5,
            )
        )
    return bars
