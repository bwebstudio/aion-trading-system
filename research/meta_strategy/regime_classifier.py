"""
research.meta_strategy.regime_classifier
─────────────────────────────────────────
Rule-based regime classifier v1.

Classifies each row of a compact feature matrix into one of:

    TREND_UP     — both momentum bins POS
    TREND_DOWN   — both momentum bins NEG
    COMPRESSION  — range_compression_bin == TRUE (and momentum mixed)
    RANGE        — everything else

Priority (first match wins): TREND_UP / TREND_DOWN → COMPRESSION → RANGE.
Trend overrides compression because a trending-but-quiet bar is still
better traded as trend continuation than as compression breakout.

Extension hooks (v2, not yet implemented):
  * distance_to_vwap_bin extremes → RANGE_EXTREME
  * session_bin == ASIA during overlap → LOW_LIQUIDITY
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


# Public regime labels.
TREND_UP = "TREND_UP"
TREND_DOWN = "TREND_DOWN"
RANGE = "RANGE"
COMPRESSION = "COMPRESSION"

ALL_REGIMES: tuple[str, ...] = (TREND_UP, TREND_DOWN, COMPRESSION, RANGE)


def classify_rows(df: "pd.DataFrame") -> np.ndarray:
    """
    Return an object array of regime labels, one per row.

    The df must expose `momentum_3_bin`, `momentum_5_bin`, and
    `range_compression_bin` as categorical columns.  Missing columns
    fall back to RANGE for every row.
    """
    n = len(df)
    out = np.full(n, RANGE, dtype=object)

    m3 = df.get("momentum_3_bin")
    m5 = df.get("momentum_5_bin")
    rc = df.get("range_compression_bin")

    if m3 is None or m5 is None:
        return out

    m3_vals = m3.astype(str).to_numpy()
    m5_vals = m5.astype(str).to_numpy()

    pos_mask = (m3_vals == "POS") & (m5_vals == "POS")
    neg_mask = (m3_vals == "NEG") & (m5_vals == "NEG")
    mixed_mask = ~(pos_mask | neg_mask)

    out[pos_mask] = TREND_UP
    out[neg_mask] = TREND_DOWN

    if rc is not None:
        rc_vals = rc.astype(str).to_numpy()
        comp_mask = mixed_mask & (rc_vals == "TRUE")
        out[comp_mask] = COMPRESSION
    return out


def classify_row(row: dict | "pd.Series") -> str:
    """Single-row classifier (slow path, used for ad-hoc inspection)."""
    m3 = str(row.get("momentum_3_bin", ""))
    m5 = str(row.get("momentum_5_bin", ""))
    rc = str(row.get("range_compression_bin", ""))
    if m3 == "POS" and m5 == "POS":
        return TREND_UP
    if m3 == "NEG" and m5 == "NEG":
        return TREND_DOWN
    if rc == "TRUE":
        return COMPRESSION
    return RANGE


__all__ = [
    "TREND_UP",
    "TREND_DOWN",
    "RANGE",
    "COMPRESSION",
    "ALL_REGIMES",
    "classify_rows",
    "classify_row",
]
