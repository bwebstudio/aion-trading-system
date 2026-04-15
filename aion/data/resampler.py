"""
aion.data.resampler
────────────────────
Resample M1 bars to higher timeframes (M5, M15, H1).

Design decisions:
  - Source timeframe must be M1.  The resampler is not generic on purpose;
    other source/target combinations introduce ambiguity about alignment.
  - Aggregation rules:
      open         = first bar's open         (entry price of the period)
      high         = max of all highs
      low          = min of all lows
      close        = last bar's close         (exit price of the period)
      tick_volume  = sum                       (total activity)
      real_volume  = sum
      spread       = mean                      (typical spread cost; see note)
  - Spread policy — mean:
      Using max (worst case) inflates the apparent cost of sporadic wide-spread
      events.  Using first (like open) ignores intra-bar spread dynamics.
      Mean best represents the average execution environment during the bar.
  - Bar alignment: label='left', closed='left'.
      A 5-minute bar starting at 08:00 captures M1 bars at
      08:00, 08:01, 08:02, 08:03, 08:04 and is labelled 08:00 (no lookahead).
  - Incomplete last window:
      The last resampled bar may be incomplete if the M1 series ends mid-period
      (e.g. the most recent live bar).  It is included, not dropped.
      Callers that work with only completed bars should discard the last result.
  - is_valid is always True for resampled bars.  Run validate_bars on M1
    data before resampling to catch problems before they propagate.
  - Session boundaries are NOT enforced by the resampler.  A bar may span
    an overnight gap.  Handle session-boundary resampling in the pipeline
    if required.

Rules:
  - Pure function: no side effects.
  - Raises ResamplerError for unsupported configurations.
"""

from __future__ import annotations

from datetime import timezone
from zoneinfo import ZoneInfo

import pandas as pd

from aion.core.enums import DataSource, Timeframe
from aion.core.models import MarketBar


class ResamplerError(Exception):
    """Raised for unsupported or invalid resampling requests."""


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# pandas offset aliases for each target timeframe.
# Uses the pandas 2.x-safe aliases ('min' not 'T', 'h' not 'H').
_RESAMPLE_OFFSETS: dict[Timeframe, str] = {
    Timeframe.M5: "5min",
    Timeframe.M15: "15min",
    Timeframe.H1: "1h",
}

_SUPPORTED_TARGETS: frozenset[Timeframe] = frozenset(_RESAMPLE_OFFSETS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def resample_bars(
    bars: list[MarketBar],
    target_timeframe: Timeframe,
    market_tz: ZoneInfo | None = None,
) -> list[MarketBar]:
    """
    Resample M1 bars to `target_timeframe`.

    Parameters
    ----------
    bars:
        M1 bars, sorted ascending by timestamp_utc.
    target_timeframe:
        Target timeframe.  Must be one of M5, M15, H1.
    market_tz:
        Market timezone for the output bars' timestamp_market field.
        If None, UTC is used.  Pass instrument.market_timezone for correct
        market-time labels.

    Returns
    -------
    list[MarketBar]
        Resampled bars, sorted ascending.  May be empty if `bars` is empty.

    Raises
    ------
    ResamplerError
        If `bars` is not empty and the source timeframe is not M1,
        or if `target_timeframe` is not supported.
    """
    if not bars:
        return []

    _validate_inputs(bars, target_timeframe)

    tz = market_tz or ZoneInfo("UTC")
    symbol = bars[0].symbol
    source = bars[0].source

    df = _bars_to_dataframe(bars)
    resampled_df = _resample_dataframe(df, target_timeframe)

    return _dataframe_to_market_bars(resampled_df, symbol, target_timeframe, source, tz)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _validate_inputs(bars: list[MarketBar], target: Timeframe) -> None:
    source_tf = bars[0].timeframe
    if source_tf != Timeframe.M1:
        raise ResamplerError(
            f"Resampler requires M1 source bars, got {source_tf}.  "
            f"Only M1→higher-TF resampling is supported."
        )
    if target not in _SUPPORTED_TARGETS:
        raise ResamplerError(
            f"Unsupported target timeframe: {target}.  "
            f"Supported: {sorted(t.value for t in _SUPPORTED_TARGETS)}"
        )
    if target == Timeframe.M1:
        raise ResamplerError("Target timeframe M1 is the same as source — nothing to do.")


def _bars_to_dataframe(bars: list[MarketBar]) -> pd.DataFrame:
    """Convert a list of MarketBar to a DataFrame indexed by timestamp_utc."""
    rows = [
        {
            "timestamp_utc": b.timestamp_utc,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "tick_volume": b.tick_volume,
            "real_volume": b.real_volume,
            "spread": b.spread,
        }
        for b in bars
    ]
    df = pd.DataFrame(rows)
    df = df.set_index("timestamp_utc")
    df.index = pd.DatetimeIndex(df.index)
    return df


def _resample_dataframe(df: pd.DataFrame, target: Timeframe) -> pd.DataFrame:
    """
    Aggregate a bar DataFrame to the target timeframe.

    label='left'  → bar labelled with the opening timestamp of the period.
    closed='left' → period is [open, close), no lookahead.
    """
    offset = _RESAMPLE_OFFSETS[target]

    agg = df.resample(offset, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        tick_volume=("tick_volume", "sum"),
        real_volume=("real_volume", "sum"),
        spread=("spread", "mean"),
    )

    # Drop periods with no M1 data (gaps produce all-NaN rows)
    agg = agg.dropna(subset=["open", "close"])

    return agg


def _dataframe_to_market_bars(
    df: pd.DataFrame,
    symbol: str,
    timeframe: Timeframe,
    source: DataSource,
    market_tz: ZoneInfo,
) -> list[MarketBar]:
    """Convert the aggregated DataFrame back to a list of MarketBar."""
    bars: list[MarketBar] = []

    for ts_utc, row in df.iterrows():
        # ts_utc arrives as a pandas Timestamp; convert to stdlib datetime
        ts_utc_dt = ts_utc.to_pydatetime()  # type: ignore[union-attr]
        if ts_utc_dt.tzinfo is None:
            ts_utc_dt = ts_utc_dt.replace(tzinfo=timezone.utc)
        else:
            ts_utc_dt = ts_utc_dt.astimezone(timezone.utc)

        ts_market = ts_utc_dt.astimezone(market_tz)

        bars.append(
            MarketBar(
                symbol=symbol,
                timestamp_utc=ts_utc_dt,
                timestamp_market=ts_market,
                timeframe=timeframe,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                tick_volume=float(row["tick_volume"]),
                real_volume=float(row["real_volume"]),
                spread=float(row["spread"]),
                source=source,
                is_valid=True,
            )
        )

    return bars
