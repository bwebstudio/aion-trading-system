"""
aion.data.csv_adapter
─────────────────────
Load historical OHLCV bars from CSV files and return a list of RawBar.

Supported formats (auto-detected):
  1. MT5 export  — columns like <DATE>, <TIME>, <OPEN>, <HIGH>, ...
  2. Generic     — columns like Date, Time, Open, High, ...
  3. Single-col  — a single 'timestamp' or 'datetime' column

The adapter's only job is to parse and return RawBar objects.
It does NOT validate, normalise, or compute features.

Rules:
- Returns an empty list (not raises) only when the file is present
  but contains zero data rows after parsing.
- Raises CsvAdapterError for unrecoverable problems (missing file,
  unreadable format, missing required columns).
- The timestamp is always stored in the instrument's broker timezone.
  Callers must pass `broker_timezone` so the adapter can attach it.
"""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from aion.core.enums import DataSource
from aion.core.models import RawBar
from aion.data.schema import RawBarCols, assert_raw_bar_schema


class CsvAdapterError(Exception):
    """Raised for unrecoverable CSV loading problems."""


# ─────────────────────────────────────────────────────────────────────────────
# Column name aliases  (maps source column → canonical name)
# ─────────────────────────────────────────────────────────────────────────────

# MT5 export format
_MT5_MAP: dict[str, str] = {
    "<DATE>": "date",
    "<TIME>": "time",
    "<OPEN>": "open",
    "<HIGH>": "high",
    "<LOW>": "low",
    "<CLOSE>": "close",
    "<TICKVOL>": "tick_volume",
    "<VOL>": "real_volume",
    "<SPREAD>": "spread",
}

# Generic / common formats
_GENERIC_MAP: dict[str, str] = {
    "Date": "date",
    "date": "date",
    "Time": "time",
    "time": "time",
    "Datetime": "timestamp",
    "datetime": "timestamp",
    "Timestamp": "timestamp",
    "timestamp": "timestamp",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "tick_volume",
    "Tick Volume": "tick_volume",
    "TickVol": "tick_volume",
    "tick_volume": "tick_volume",
    "Real Volume": "real_volume",
    "real_volume": "real_volume",
    "Spread": "spread",
    "spread": "spread",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def load_csv_bars(
    path: Path,
    symbol: str,
    broker_timezone: str,
    *,
    date_col: str | None = None,
    time_col: str | None = None,
    timestamp_col: str | None = None,
    datetime_format: str | None = None,
) -> list[RawBar]:
    """
    Parse a CSV file and return a list of RawBar objects.

    Parameters
    ----------
    path:
        Path to the CSV file.
    symbol:
        Canonical instrument symbol (e.g. 'EURUSD').
    broker_timezone:
        IANA timezone name for the broker's timestamps
        (e.g. 'Etc/UTC', 'America/New_York').
    date_col:
        Override: name of the date column after column normalisation.
    time_col:
        Override: name of the time column after column normalisation.
    timestamp_col:
        Override: name of a combined datetime column after normalisation.
    datetime_format:
        strptime format string passed to pd.to_datetime.
        If None, pandas auto-detects.

    Returns
    -------
    list[RawBar]
        May be empty if the file has no parseable rows.

    Raises
    ------
    CsvAdapterError
        If the file is missing, empty, unreadable, or lacks required columns.
    """
    df = _read_file(path)
    df = _normalise_columns(df)
    df = _build_timestamp_column(df, date_col, time_col, timestamp_col, datetime_format)
    df = _coerce_numerics(df)
    df = _fill_optional_columns(df)

    assert_raw_bar_schema(df)

    broker_tz = ZoneInfo(broker_timezone)
    return _to_raw_bars(df, symbol, broker_tz)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _read_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise CsvAdapterError(f"File not found: {path}")
    try:
        # sep=None + engine='python' → auto-detect separator
        df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    except Exception as exc:
        raise CsvAdapterError(f"Cannot read CSV '{path}': {exc}") from exc
    if df.empty:
        raise CsvAdapterError(f"CSV file is empty: {path}")
    return df


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename source columns to canonical lowercase names."""
    rename: dict[str, str] = {}
    for col in df.columns:
        if col in _MT5_MAP:
            rename[col] = _MT5_MAP[col]
        elif col in _GENERIC_MAP:
            rename[col] = _GENERIC_MAP[col]
        else:
            # Fallback: lowercase + strip + replace spaces with underscore
            rename[col] = col.strip().lower().replace(" ", "_")
    return df.rename(columns=rename)


def _build_timestamp_column(
    df: pd.DataFrame,
    date_col: str | None,
    time_col: str | None,
    timestamp_col: str | None,
    datetime_format: str | None,
) -> pd.DataFrame:
    """
    Build a single 'timestamp' column from whatever date/time columns exist.

    Auto-detection priority:
    1. Caller-specified timestamp_col or (date_col + time_col)
    2. Single 'timestamp' or 'datetime' column
    3. Separate 'date' + 'time' columns
    4. Single 'date' column
    """
    df = df.copy()

    # --- resolve caller overrides ---
    if timestamp_col is None and date_col is None:
        if "timestamp" in df.columns:
            timestamp_col = "timestamp"
        elif "datetime" in df.columns:
            timestamp_col = "datetime"
        elif "date" in df.columns and "time" in df.columns:
            date_col, time_col = "date", "time"
        elif "date" in df.columns:
            date_col = "date"
        else:
            raise CsvAdapterError(
                "Cannot detect timestamp columns.  "
                "Pass date_col / timestamp_col explicitly."
            )

    # --- parse ---
    if date_col and time_col:
        combined = df[date_col].str.strip() + " " + df[time_col].str.strip()
        df[RawBarCols.TIMESTAMP] = pd.to_datetime(
            combined, format=datetime_format, utc=False
        )
    elif date_col:
        df[RawBarCols.TIMESTAMP] = pd.to_datetime(
            df[date_col], format=datetime_format, utc=False
        )
    elif timestamp_col:
        df[RawBarCols.TIMESTAMP] = pd.to_datetime(
            df[timestamp_col], format=datetime_format, utc=False
        )

    return df


def _coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Convert OHLCV columns to float.  Raise if required columns are missing."""
    for col in (RawBarCols.OPEN, RawBarCols.HIGH, RawBarCols.LOW,
                RawBarCols.CLOSE, RawBarCols.TICK_VOLUME):
        if col not in df.columns:
            raise CsvAdapterError(f"Required column missing after normalisation: '{col}'")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _fill_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure optional columns exist with sensible defaults."""
    if RawBarCols.REAL_VOLUME not in df.columns:
        df[RawBarCols.REAL_VOLUME] = 0.0
    else:
        df[RawBarCols.REAL_VOLUME] = (
            pd.to_numeric(df[RawBarCols.REAL_VOLUME], errors="coerce").fillna(0.0)
        )

    if RawBarCols.SPREAD not in df.columns:
        df[RawBarCols.SPREAD] = 0.0
    else:
        df[RawBarCols.SPREAD] = (
            pd.to_numeric(df[RawBarCols.SPREAD], errors="coerce").fillna(0.0)
        )
    return df


def _to_raw_bars(
    df: pd.DataFrame, symbol: str, broker_tz: ZoneInfo
) -> list[RawBar]:
    """Convert a clean DataFrame to a list of RawBar objects."""
    bars: list[RawBar] = []

    for row in df.itertuples(index=False):
        ts = getattr(row, RawBarCols.TIMESTAMP)
        if pd.isna(ts):
            continue

        # Localise naive timestamps to the broker timezone
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=broker_tz)

        open_ = getattr(row, RawBarCols.OPEN)
        high = getattr(row, RawBarCols.HIGH)
        low = getattr(row, RawBarCols.LOW)
        close = getattr(row, RawBarCols.CLOSE)
        tick_vol = getattr(row, RawBarCols.TICK_VOLUME)
        real_vol = getattr(row, RawBarCols.REAL_VOLUME)
        spread = getattr(row, RawBarCols.SPREAD)

        # Skip rows where OHLC is entirely NaN (pandas coerce artefact)
        if any(
            isinstance(v, float) and pd.isna(v)
            for v in (open_, high, low, close, tick_vol)
        ):
            continue

        bars.append(
            RawBar(
                symbol=symbol,
                timestamp=ts,
                open=float(open_),
                high=float(high),
                low=float(low),
                close=float(close),
                tick_volume=float(tick_vol),
                real_volume=float(real_vol),
                spread=float(spread),
                source=DataSource.CSV,
            )
        )

    return bars
