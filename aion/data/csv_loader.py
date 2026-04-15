"""
aion.data.csv_loader
─────────────────────
Load a CSV file and return normalised MarketBar objects ready for the pipeline.

This is a convenience layer over csv_adapter + normalizer.  It exists so
that scripts can load bars in a single call without knowing the internal
two-step process.

Usage:
    from aion.data.csv_loader import load_bars

    bars = load_bars(
        path=Path("data/raw/us100_3months_m1.csv"),
        instrument=us100_spec,
    )

The bars are sorted ascending by timestamp_utc and tagged with Timeframe.M1.
"""

from __future__ import annotations

from pathlib import Path

from aion.core.enums import Timeframe
from aion.core.models import InstrumentSpec, MarketBar
from aion.data.csv_adapter import load_csv_bars
from aion.data.normalizer import normalize_bars


def load_bars(
    path: Path,
    instrument: InstrumentSpec,
    *,
    timeframe: Timeframe = Timeframe.M1,
    drop_last: bool = True,
) -> list[MarketBar]:
    """
    Load a CSV file and return normalised, sorted MarketBar objects.

    Parameters
    ----------
    path:
        Path to the CSV file (MT5 or generic format).
    instrument:
        InstrumentSpec for the target instrument.  Provides symbol,
        broker_timezone, and market_timezone.
    timeframe:
        Timeframe of the source bars.  Default M1.
    drop_last:
        If True (default), drop the last bar.  MT5 CSV exports often
        include an incomplete bar at the end.

    Returns
    -------
    list[MarketBar]
        Sorted ascending by timestamp_utc.
    """
    raw_bars = load_csv_bars(
        path,
        symbol=instrument.symbol,
        broker_timezone=instrument.broker_timezone,
    )

    bars = normalize_bars(raw_bars, instrument, timeframe)
    bars.sort(key=lambda b: b.timestamp_utc)

    if drop_last and len(bars) > 1:
        bars = bars[:-1]

    return bars
