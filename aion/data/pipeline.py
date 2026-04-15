"""
aion.data.pipeline
───────────────────
Orchestrates the full Market Engine pipeline for a single symbol.

Pipeline stages:
    1.  Load CSV      → list[RawBar]
    2.  Normalise     → list[MarketBar]  (broker tz → UTC → market tz)
    3.  Sort          → ascending by timestamp_utc
    4.  Drop last bar (optional, see below)
    5.  Validate M1   → DataQualityReport
    6.  Resample      → bars_m5, bars_m15
    7.  Feature series → list[FeatureVector]  (O(N) batch pass)
    8.  Build snapshot → MarketSnapshot       (from most recent bars)
    9.  Persist        (optional)

Incomplete-last-bar policy
──────────────────────────
When `drop_incomplete_last_bar=True` (default):
    The final M1 bar is dropped before the snapshot is built.
    Rationale: CSV exports from MT5 often contain a partial last bar
    (the candle is still forming when the export runs).  Including it
    would give the feature engine stale or misleading data.
    Safe default for all historical back-tests and research workflows.

When `drop_incomplete_last_bar=False`:
    The last bar is kept.  Use this only when the caller has verified
    the last bar is complete (e.g. a live feed that always returns
    closed bars).

Missing-bars policy (PROVISIONAL)
──────────────────────────────────
The validator counts missing bars using rounding arithmetic:
    missing = round(delta / expected_delta) - 1
This handles minor clock drift but may over-count gaps during weekends,
public holidays, and market closures.  A calendar-aware gap filter
(using session definitions) will replace this in a future iteration.
The current behaviour is intentionally conservative — it over-reports
rather than under-reports data quality problems.

Session / VWAP consistency
──────────────────────────
Feature vectors are computed via `compute_feature_series`, which uses
incremental session state tracking (O(N) pass).  Session VWAP resets
at each `session_open_utc` change.  The LONDON→OVERLAP_LONDON_NY
transition causes a VWAP reset at NY open (13:30 UTC winter, same in
summer because NY DST shifts independently of London DST).  This is
documented in features.py and is V1 behaviour.

Public API:
    run_historical_pipeline(csv_path, instrument, local_timezone, ...)
        → PipelineResult
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from aion.core.enums import Timeframe
from aion.core.ids import new_pipeline_run_id
from aion.core.models import (
    DataQualityReport,
    FeatureVector,
    InstrumentSpec,
    MarketBar,
    MarketSnapshot,
)
from aion.data.csv_adapter import load_csv_bars
from aion.data.features import compute_feature_series
from aion.data.normalizer import normalize_bars
from aion.data.resampler import resample_bars
from aion.data.sessions import build_session_context
from aion.data.snapshots import build_snapshot
from aion.data.validator import validate_bars


# ─────────────────────────────────────────────────────────────────────────────
# Result object
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PipelineResult:
    """
    Full output of a historical pipeline run.

    Consumers can access individual stages or use the terminal `snapshot`
    directly.  All bar lists are sorted ascending by `timestamp_utc`.
    """

    # Identity
    pipeline_run_id: str

    # Per-stage outputs
    bars_m1: list[MarketBar]
    bars_m5: list[MarketBar]
    bars_m15: list[MarketBar]
    features_m1: list[FeatureVector]
    quality_report: DataQualityReport
    snapshot: MarketSnapshot

    # Pipeline metadata
    bars_loaded: int
    """Number of raw bars loaded from the CSV before any filtering."""

    bars_after_normalise: int
    """After normalisation (failed rows dropped)."""

    bars_dropped_incomplete: int
    """0 or 1 — 1 if `drop_incomplete_last_bar=True` removed the last bar."""

    elapsed_seconds: float
    """Wall-clock time for the full pipeline run."""

    persist_paths: list[Path] = field(default_factory=list)
    """Paths written when `persist=True`.  Empty if persist=False."""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def run_historical_pipeline(
    csv_path: Path,
    instrument: InstrumentSpec,
    local_timezone: str,
    *,
    drop_incomplete_last_bar: bool = True,
    persist: bool = False,
    persist_bar_root: Path | None = None,
    persist_feature_root: Path | None = None,
    persist_snapshot_root: Path | None = None,
    window_m1: int = 100,
    window_m5: int = 100,
    window_m15: int = 100,
) -> PipelineResult:
    """
    Run the full historical pipeline for a single CSV file.

    Parameters
    ----------
    csv_path:
        Path to an MT5 or generic CSV file.
    instrument:
        InstrumentSpec describing the traded instrument.  Must include
        correct `broker_timezone` and `market_timezone`.
    local_timezone:
        IANA timezone name for the operator's local timezone.  Used
        for session context (not for bar timestamps).
    drop_incomplete_last_bar:
        If True (default), the last M1 bar is dropped before building
        the snapshot.  See module docstring for rationale.
    persist:
        If True, write bars and features to Parquet and the snapshot
        to JSON.  Requires at least one of the persist_*_root paths.
    persist_bar_root / persist_feature_root / persist_snapshot_root:
        Root directories for month-partitioned output.  Only used when
        `persist=True`.
    window_m1 / window_m5 / window_m15:
        Number of bars to keep in the snapshot bar windows.

    Returns
    -------
    PipelineResult
        Fully populated result object.
    """
    run_id = new_pipeline_run_id()
    t0 = time.perf_counter()

    # ── Stage 1: Load CSV ─────────────────────────────────────────────────────
    raw_bars = load_csv_bars(
        csv_path,
        symbol=instrument.symbol,
        broker_timezone=instrument.broker_timezone,
    )
    bars_loaded = len(raw_bars)

    # ── Stage 2: Normalise ────────────────────────────────────────────────────
    bars_m1 = normalize_bars(raw_bars, instrument, Timeframe.M1)
    bars_m1.sort(key=lambda b: b.timestamp_utc)
    bars_after_normalise = len(bars_m1)

    # ── Stage 3 (optional): Drop incomplete last bar ──────────────────────────
    bars_dropped_incomplete = 0
    if drop_incomplete_last_bar and len(bars_m1) > 1:
        bars_m1 = bars_m1[:-1]
        bars_dropped_incomplete = 1

    if not bars_m1:
        raise PipelineError(
            f"No M1 bars remain after normalisation "
            f"(loaded={bars_loaded}, drop_incomplete={drop_incomplete_last_bar})."
        )

    # ── Stage 4: Validate ─────────────────────────────────────────────────────
    quality_report = validate_bars(bars_m1, Timeframe.M1)

    # ── Stage 5: Resample ─────────────────────────────────────────────────────
    bars_m5 = resample_bars(bars_m1, Timeframe.M5)
    bars_m15 = resample_bars(bars_m1, Timeframe.M15)

    # ── Stage 6: Batch feature computation ───────────────────────────────────
    features_m1 = compute_feature_series(
        bars_m1,
        timeframe=Timeframe.M1,
        market_timezone=instrument.market_timezone,
        broker_timezone=instrument.broker_timezone,
        local_timezone=local_timezone,
    )

    # ── Stage 7: Build snapshot ───────────────────────────────────────────────
    latest_ts = bars_m1[-1].timestamp_utc
    session_ctx = build_session_context(
        latest_ts,
        market_timezone=instrument.market_timezone,
        broker_timezone=instrument.broker_timezone,
        local_timezone=local_timezone,
    )

    snapshot = build_snapshot(
        instrument,
        bars_m1,
        bars_m5,
        bars_m15,
        session_ctx,
        window_m1=window_m1,
        window_m5=window_m5,
        window_m15=window_m15,
    )

    elapsed = time.perf_counter() - t0

    # ── Stage 8 (optional): Persist ───────────────────────────────────────────
    persist_paths: list[Path] = []
    if persist:
        from aion.data.persistence import (
            save_bars_partitioned,
            save_features_partitioned,
            save_snapshot,
        )

        if persist_bar_root is not None:
            written = save_bars_partitioned(bars_m1, Timeframe.M1, persist_bar_root)
            persist_paths.extend(written)

        if persist_feature_root is not None:
            written = save_features_partitioned(
                features_m1, Timeframe.M1, persist_feature_root
            )
            persist_paths.extend(written)

        if persist_snapshot_root is not None:
            ts_str = latest_ts.strftime("%Y%m%d_%H%M%S")
            snap_path = (
                Path(persist_snapshot_root)
                / instrument.symbol
                / f"{ts_str}_{snapshot.snapshot_id}.json"
            )
            save_snapshot(snapshot, snap_path)
            persist_paths.append(snap_path)

    return PipelineResult(
        pipeline_run_id=run_id,
        bars_m1=bars_m1,
        bars_m5=bars_m5,
        bars_m15=bars_m15,
        features_m1=features_m1,
        quality_report=quality_report,
        snapshot=snapshot,
        bars_loaded=bars_loaded,
        bars_after_normalise=bars_after_normalise,
        bars_dropped_incomplete=bars_dropped_incomplete,
        elapsed_seconds=elapsed,
        persist_paths=persist_paths,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────


class PipelineError(Exception):
    """Raised when the pipeline cannot complete due to data or config issues."""
