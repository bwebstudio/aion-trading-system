"""
aion.data.snapshots
────────────────────
Assemble a MarketSnapshot from validated, resampled, feature-enriched bars.

Entry point:
    build_snapshot(instrument, bars_m1, bars_m5, bars_m15, session_context)

Internal pipeline:
    1. Validate bars_m1 → DataQualityReport
    2. Compute features from bars_m1 (full history, before windowing)
    3. Trim each bar list to its window size
    4. Assemble MarketSnapshot

Design decisions:
    - Features are computed BEFORE trimming the window.
      Using full history (e.g. 200+ M1 bars) gives accurate rolling stats.
      The snapshot only stores the most recent N bars per timeframe to
      control memory, but the feature vector reflects the full history.
    - The quality report is always run on the full M1 series passed in.
    - bars_m5 / bars_m15 are passed in (already resampled by caller).
      The snapshot builder does not call the resampler internally.
      This keeps responsibilities clear and allows callers to control
      when resampling happens (e.g. not every tick).

Rules:
    - Raises SnapshotError if bars_m1 is empty or inconsistent.
    - Pure function: no file I/O, no side effects.
"""

from __future__ import annotations

from aion.core.constants import (
    SNAPSHOT_BARS_M1,
    SNAPSHOT_BARS_M5,
    SNAPSHOT_BARS_M15,
    SNAPSHOT_VERSION,
)
from aion.core.enums import Timeframe
from aion.core.models import (
    InstrumentSpec,
    MarketBar,
    MarketSnapshot,
    SessionContext,
)
from aion.data.features import compute_feature_vector
from aion.data.validator import validate_bars


class SnapshotError(Exception):
    """Raised when a snapshot cannot be built due to invalid input."""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def build_snapshot(
    instrument: InstrumentSpec,
    bars_m1: list[MarketBar],
    bars_m5: list[MarketBar],
    bars_m15: list[MarketBar],
    session_context: SessionContext,
    *,
    window_m1: int = SNAPSHOT_BARS_M1,
    window_m5: int = SNAPSHOT_BARS_M5,
    window_m15: int = SNAPSHOT_BARS_M15,
) -> MarketSnapshot:
    """
    Build a MarketSnapshot from normalised and resampled bar series.

    Parameters
    ----------
    instrument:
        Full instrument specification.
    bars_m1:
        M1 bars sorted ascending.  Must not be empty.
        Used for: quality validation, feature computation, snapshot window.
    bars_m5:
        M5 bars sorted ascending (pre-resampled by caller).
    bars_m15:
        M15 bars sorted ascending (pre-resampled by caller).
    session_context:
        Session state at the time of the latest bar.
    window_m1 / window_m5 / window_m15:
        Number of most-recent bars to store in the snapshot.
        Features are computed on the FULL series before windowing.

    Returns
    -------
    MarketSnapshot

    Raises
    ------
    SnapshotError
        If bars_m1 is empty or has a mismatched symbol.
    """
    _validate_inputs(instrument, bars_m1, bars_m5, bars_m15)

    latest_bar = bars_m1[-1]

    # Step 1: Validate data quality on full M1 series
    quality_report = validate_bars(bars_m1, Timeframe.M1)

    # Step 2: Compute features from full M1 history
    # (before windowing so rolling stats see as much history as available)
    feature_vector = compute_feature_vector(bars_m1, session_context, Timeframe.M1)

    # Step 3: Trim to snapshot windows
    snapshot_m1 = bars_m1[-window_m1:]
    snapshot_m5 = bars_m5[-window_m5:] if bars_m5 else []
    snapshot_m15 = bars_m15[-window_m15:] if bars_m15 else []

    return MarketSnapshot(
        symbol=instrument.symbol,
        timestamp_utc=latest_bar.timestamp_utc,
        base_timeframe=Timeframe.M1,
        instrument=instrument,
        session_context=session_context,
        latest_bar=latest_bar,
        bars_m1=snapshot_m1,
        bars_m5=snapshot_m5,
        bars_m15=snapshot_m15,
        feature_vector=feature_vector,
        quality_report=quality_report,
        snapshot_version=SNAPSHOT_VERSION,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────


def _validate_inputs(
    instrument: InstrumentSpec,
    bars_m1: list[MarketBar],
    bars_m5: list[MarketBar],
    bars_m15: list[MarketBar],
) -> None:
    if not bars_m1:
        raise SnapshotError(
            f"Cannot build snapshot for {instrument.symbol}: bars_m1 is empty."
        )

    # Verify symbol consistency (catch caller mistakes early)
    for bars, label in [(bars_m1, "M1"), (bars_m5, "M5"), (bars_m15, "M15")]:
        for bar in bars:
            if bar.symbol != instrument.symbol:
                raise SnapshotError(
                    f"Symbol mismatch in {label} bars: "
                    f"expected {instrument.symbol!r}, got {bar.symbol!r}."
                )

    # Verify timeframes
    _check_timeframes(bars_m1, Timeframe.M1, "bars_m1")
    if bars_m5:
        _check_timeframes(bars_m5, Timeframe.M5, "bars_m5")
    if bars_m15:
        _check_timeframes(bars_m15, Timeframe.M15, "bars_m15")


def _check_timeframes(
    bars: list[MarketBar], expected: Timeframe, label: str
) -> None:
    wrong = [b for b in bars if b.timeframe != expected]
    if wrong:
        raise SnapshotError(
            f"{label}: found {len(wrong)} bar(s) with unexpected timeframe "
            f"(expected {expected}, got {wrong[0].timeframe})."
        )
