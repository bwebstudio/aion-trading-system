"""
aion.data.persistence
──────────────────────
Read and write Parquet files for bars and features, and JSON for snapshots.

Storage layout (month-partitioned):
    data/normalized/{SYMBOL}/{TIMEFRAME}/YYYY-MM.parquet   ← bars
    data/features/{SYMBOL}/{TIMEFRAME}/YYYY-MM.parquet     ← feature vectors
    data/snapshots/{SYMBOL}/YYYYMMDD_HHMMSS_{id}.json      ← snapshots

Design decisions:
  - `timestamp_market` is NOT stored.  It is fully deterministic given
    `timestamp_utc` and `market_timezone`, so re-deriving it on load keeps
    the files smaller and avoids stale tz data.
  - Bars use Parquet (columnar, snappy-compressed) — fast range scans.
  - Snapshots use JSON (human-readable, self-describing).
  - All path helpers are pure functions — no side effects.

Public API:
    save_bars(bars, path)
    load_bars(path, timeframe, market_timezone)
    save_features(features, path)
    load_features(path, timeframe)
    save_snapshot(snapshot, path)
    load_snapshot(path)
    save_bars_partitioned(bars, timeframe, root)
    save_features_partitioned(features, timeframe, root)
    bar_partition_path(root, symbol, timeframe, year, month)
    feature_partition_path(root, symbol, timeframe, year, month)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from aion.core.enums import DataSource, Timeframe
from aion.core.models import FeatureVector, MarketBar, MarketSnapshot


class PersistenceError(Exception):
    """Raised when a persistence operation fails."""


# ─────────────────────────────────────────────────────────────────────────────
# Column names
# ─────────────────────────────────────────────────────────────────────────────

# Columns stored for bars (timestamp_market is excluded — derived on load)
_BAR_COLS = [
    "symbol",
    "timestamp_utc",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "real_volume",
    "spread",
    "source",
    "is_valid",
]

# Columns stored for features
_FEATURE_COLS = [
    "symbol",
    "timestamp_utc",
    "timeframe",
    "atr_14",
    "rolling_range_10",
    "rolling_range_20",
    "volatility_percentile_20",
    "session_high",
    "session_low",
    "opening_range_high",
    "opening_range_low",
    "vwap_session",
    "spread_mean_20",
    "spread_zscore_20",
    "return_1",
    "return_5",
    "candle_body",
    "upper_wick",
    "lower_wick",
    "distance_to_session_high",
    "distance_to_session_low",
    "feature_set_version",
]


# ─────────────────────────────────────────────────────────────────────────────
# Bars
# ─────────────────────────────────────────────────────────────────────────────


def save_bars(bars: list[MarketBar], path: Path) -> None:
    """
    Serialise a list of MarketBar objects to a Parquet file at `path`.

    The parent directory is created if it does not exist.
    `timestamp_market` is NOT stored — it is re-derived on load.
    """
    if not bars:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "symbol": b.symbol,
            "timestamp_utc": b.timestamp_utc,
            "timeframe": b.timeframe.value,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "tick_volume": b.tick_volume,
            "real_volume": b.real_volume,
            "spread": b.spread,
            "source": b.source.value,
            "is_valid": b.is_valid,
        }
        for b in bars
    ]
    df = pd.DataFrame(records)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)


def load_bars(
    path: Path,
    timeframe: Timeframe,
    market_timezone: str | None = None,
) -> list[MarketBar]:
    """
    Load MarketBar objects from a Parquet file.

    Parameters
    ----------
    path:
        Path to a .parquet file produced by `save_bars`.
    timeframe:
        Timeframe of the bars stored in this file.
    market_timezone:
        IANA timezone name used to derive `timestamp_market` from
        `timestamp_utc`.  If None, `timestamp_market` equals `timestamp_utc`.

    Returns
    -------
    list[MarketBar]
        Sorted ascending by `timestamp_utc`.
    """
    path = Path(path)
    if not path.exists():
        raise PersistenceError(f"Bar file not found: {path}")

    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except Exception as exc:
        raise PersistenceError(f"Cannot read bar file '{path}': {exc}") from exc

    mtz = ZoneInfo(market_timezone) if market_timezone else timezone.utc

    bars: list[MarketBar] = []
    for row in df.itertuples(index=False):
        ts_utc = _ensure_utc(row.timestamp_utc)
        ts_market = ts_utc.astimezone(mtz)
        bars.append(
            MarketBar(
                symbol=row.symbol,
                timestamp_utc=ts_utc,
                timestamp_market=ts_market,
                timeframe=Timeframe(row.timeframe),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                tick_volume=float(row.tick_volume),
                real_volume=float(row.real_volume),
                spread=float(row.spread),
                source=DataSource(row.source),
                is_valid=bool(row.is_valid),
            )
        )

    bars.sort(key=lambda b: b.timestamp_utc)
    return bars


# ─────────────────────────────────────────────────────────────────────────────
# Features
# ─────────────────────────────────────────────────────────────────────────────


def save_features(features: list[FeatureVector], path: Path) -> None:
    """
    Serialise a list of FeatureVector objects to a Parquet file at `path`.

    The parent directory is created if it does not exist.
    """
    if not features:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "symbol": fv.symbol,
            "timestamp_utc": fv.timestamp_utc,
            "timeframe": fv.timeframe.value,
            "atr_14": fv.atr_14,
            "rolling_range_10": fv.rolling_range_10,
            "rolling_range_20": fv.rolling_range_20,
            "volatility_percentile_20": fv.volatility_percentile_20,
            "session_high": fv.session_high,
            "session_low": fv.session_low,
            "opening_range_high": fv.opening_range_high,
            "opening_range_low": fv.opening_range_low,
            "vwap_session": fv.vwap_session,
            "spread_mean_20": fv.spread_mean_20,
            "spread_zscore_20": fv.spread_zscore_20,
            "return_1": fv.return_1,
            "return_5": fv.return_5,
            "candle_body": fv.candle_body,
            "upper_wick": fv.upper_wick,
            "lower_wick": fv.lower_wick,
            "distance_to_session_high": fv.distance_to_session_high,
            "distance_to_session_low": fv.distance_to_session_low,
            "feature_set_version": fv.feature_set_version,
        }
        for fv in features
    ]
    df = pd.DataFrame(records)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)


def load_features(
    path: Path,
    timeframe: Timeframe,
) -> list[FeatureVector]:
    """
    Load FeatureVector objects from a Parquet file.

    Returns a list sorted ascending by `timestamp_utc`.
    """
    path = Path(path)
    if not path.exists():
        raise PersistenceError(f"Feature file not found: {path}")

    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except Exception as exc:
        raise PersistenceError(f"Cannot read feature file '{path}': {exc}") from exc

    def _opt(v: object) -> float | None:
        """Convert NaN / None / non-finite to None; keep finite floats."""
        if v is None:
            return None
        try:
            import math
            f = float(v)  # type: ignore[arg-type]
            return f if math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    vectors: list[FeatureVector] = []
    for row in df.itertuples(index=False):
        ts_utc = _ensure_utc(row.timestamp_utc)
        vectors.append(
            FeatureVector(
                symbol=row.symbol,
                timestamp_utc=ts_utc,
                timeframe=Timeframe(row.timeframe),
                atr_14=_opt(row.atr_14),
                rolling_range_10=_opt(row.rolling_range_10),
                rolling_range_20=_opt(row.rolling_range_20),
                volatility_percentile_20=_opt(row.volatility_percentile_20),
                session_high=_opt(row.session_high),
                session_low=_opt(row.session_low),
                opening_range_high=_opt(row.opening_range_high),
                opening_range_low=_opt(row.opening_range_low),
                vwap_session=_opt(row.vwap_session),
                spread_mean_20=_opt(row.spread_mean_20),
                spread_zscore_20=_opt(row.spread_zscore_20),
                return_1=_opt(row.return_1),
                return_5=_opt(row.return_5),
                candle_body=_opt(row.candle_body),
                upper_wick=_opt(row.upper_wick),
                lower_wick=_opt(row.lower_wick),
                distance_to_session_high=_opt(row.distance_to_session_high),
                distance_to_session_low=_opt(row.distance_to_session_low),
                feature_set_version=str(row.feature_set_version),
            )
        )

    vectors.sort(key=lambda fv: fv.timestamp_utc)
    return vectors


# ─────────────────────────────────────────────────────────────────────────────
# Snapshots
# ─────────────────────────────────────────────────────────────────────────────


def save_snapshot(snapshot: MarketSnapshot, path: Path) -> None:
    """
    Serialise a MarketSnapshot to a JSON file at `path`.

    Uses Pydantic's `model_dump_json()` for full fidelity.
    The parent directory is created if it does not exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")


def load_snapshot(path: Path) -> MarketSnapshot:
    """
    Load a MarketSnapshot from a JSON file.

    Raises PersistenceError if the file is missing or cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise PersistenceError(f"Snapshot file not found: {path}")

    try:
        raw = path.read_text(encoding="utf-8")
        return MarketSnapshot.model_validate_json(raw)
    except Exception as exc:
        raise PersistenceError(f"Cannot load snapshot from '{path}': {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Month-partitioned helpers
# ─────────────────────────────────────────────────────────────────────────────


def bar_partition_path(
    root: Path, symbol: str, timeframe: Timeframe, year: int, month: int
) -> Path:
    """
    Return the canonical path for a bar partition.

    Example: root/EURUSD/M1/2024-01.parquet
    """
    return Path(root) / symbol / timeframe.value / f"{year:04d}-{month:02d}.parquet"


def feature_partition_path(
    root: Path, symbol: str, timeframe: Timeframe, year: int, month: int
) -> Path:
    """
    Return the canonical path for a feature partition.

    Example: root/EURUSD/M1/2024-01.parquet
    """
    return Path(root) / symbol / timeframe.value / f"{year:04d}-{month:02d}.parquet"


def save_bars_partitioned(
    bars: list[MarketBar],
    timeframe: Timeframe,
    root: Path,
) -> list[Path]:
    """
    Save bars to month-partitioned Parquet files under `root`.

    Returns the list of paths written.
    Existing files for a given month are overwritten.

    Layout: root/{symbol}/{timeframe}/YYYY-MM.parquet
    """
    if not bars:
        return []

    root = Path(root)
    written: list[Path] = []

    # Group by (symbol, year, month)
    groups: dict[tuple[str, int, int], list[MarketBar]] = {}
    for bar in bars:
        key = (bar.symbol, bar.timestamp_utc.year, bar.timestamp_utc.month)
        groups.setdefault(key, []).append(bar)

    for (symbol, year, month), group in sorted(groups.items()):
        path = bar_partition_path(root, symbol, timeframe, year, month)
        save_bars(group, path)
        written.append(path)

    return written


def save_features_partitioned(
    features: list[FeatureVector],
    timeframe: Timeframe,
    root: Path,
) -> list[Path]:
    """
    Save feature vectors to month-partitioned Parquet files under `root`.

    Returns the list of paths written.
    Existing files for a given month are overwritten.

    Layout: root/{symbol}/{timeframe}/YYYY-MM.parquet
    """
    if not features:
        return []

    root = Path(root)
    written: list[Path] = []

    groups: dict[tuple[str, int, int], list[FeatureVector]] = {}
    for fv in features:
        key = (fv.symbol, fv.timestamp_utc.year, fv.timestamp_utc.month)
        groups.setdefault(key, []).append(fv)

    for (symbol, year, month), group in sorted(groups.items()):
        path = feature_partition_path(root, symbol, timeframe, year, month)
        save_features(group, path)
        written.append(path)

    return written


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_utc(ts: object) -> datetime:
    """
    Coerce a timestamp value (pandas Timestamp or datetime) to a UTC-aware
    Python datetime.  Raises PersistenceError on failure.
    """
    # pandas Timestamp
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()  # type: ignore[union-attr]

    if not isinstance(ts, datetime):
        raise PersistenceError(f"Unexpected timestamp type: {type(ts)}")

    if ts.tzinfo is None:
        # Parquet may strip timezone info — assume UTC
        return ts.replace(tzinfo=timezone.utc)

    return ts.astimezone(timezone.utc)
