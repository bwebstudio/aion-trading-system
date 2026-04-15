"""
aion.core.ids
─────────────
Deterministic and random ID generators for all platform entities.

Rules:
- Snapshot IDs are random (UUID-based) — they identify a moment in time.
- Bar IDs are deterministic — same bar always produces the same ID.
- All IDs are plain strings for easy logging, storage, and debugging.
- No external dependencies.
"""

import uuid
from datetime import datetime


def new_snapshot_id() -> str:
    """Random ID for a MarketSnapshot.  Example: 'snap_3f7a9c12b4e1'"""
    return f"snap_{uuid.uuid4().hex[:12]}"


def new_pipeline_run_id() -> str:
    """Random ID for a pipeline run.  Example: 'run_a1b2c3d4e5f6'"""
    return f"run_{uuid.uuid4().hex[:12]}"


def make_bar_id(symbol: str, timeframe: str, timestamp_utc: datetime) -> str:
    """
    Deterministic bar identifier.

    Same symbol + timeframe + UTC timestamp always yields the same ID.
    Example: 'EURUSD_M1_20240115T083000Z'
    """
    ts = timestamp_utc.strftime("%Y%m%dT%H%M%SZ")
    return f"{symbol}_{timeframe}_{ts}"


def make_feature_vector_id(symbol: str, timeframe: str, timestamp_utc: datetime) -> str:
    """Deterministic ID for a FeatureVector row."""
    ts = timestamp_utc.strftime("%Y%m%dT%H%M%SZ")
    return f"fv_{symbol}_{timeframe}_{ts}"
