"""
research.pattern_discovery.feature_builder
───────────────────────────────────────────
Build a feature matrix from MarketSnapshots for pattern discovery.

Output features (per row):

    distance_to_vwap
    distance_to_session_high
    distance_to_session_low
    momentum_3
    momentum_5
    range_compression
    session
    time_of_day

Plus rolling-σ normalised (z-score) columns for each continuous feature:

    distance_to_vwap_z           = value / rolling_std_500
    distance_to_session_high_z
    distance_to_session_low_z
    momentum_3_z
    momentum_5_z

Rolling std is computed over a trailing window (default 500 bars) using
only past values — no lookahead.  PatternGenerator uses the `_z` columns
so thresholds are expressed in units of σ, not raw price points.

Design notes
────────────
* Values that cannot be computed (insufficient history, missing VWAP,
  missing session high/low, zero rolling σ) are stored as None.
  Downstream consumers must treat None as "condition did not match".
* The row also carries bookkeeping fields (idx, timestamp_utc, close)
  so the ForwardTester can map matches back to snapshots.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from aion.core.models import MarketSnapshot
from aion.execution.execution_model import detect_session as _detect_session_utc

if TYPE_CHECKING:
    import pandas as pd
    from aion.execution.execution_model import ExecutionModel


CONTINUOUS_FEATURE_NAMES: tuple[str, ...] = (
    "distance_to_vwap",
    "distance_to_session_high",
    "distance_to_session_low",
    "momentum_3",
    "momentum_5",
)


# ─── Bin definitions (shared with pattern_generator) ─────────────────────────

# distance_to_vwap_bin thresholds on the z-score (sigma units).
VWAP_BINS: tuple[tuple[float, float, str], ...] = (
    (float("-inf"), -2.0,        "LT_NEG_2SIG"),
    (-2.0,          -1.5,        "LT_NEG_1P5SIG"),
    (-1.5,          -1.0,        "LT_NEG_1SIG"),
    (-1.0,           1.0,        "MID"),
    ( 1.0,           1.5,        "GT_POS_1SIG"),
    ( 1.5,           2.0,        "GT_POS_1P5SIG"),
    ( 2.0,           float("inf"), "GT_POS_2SIG"),
)

# distance_to_session_high / low → NEAR / MID / FAR (absolute z).
DIST_SESSION_BINS: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.75,           "NEAR"),
    (0.75, 1.5,           "MID"),
    (1.5, float("inf"),   "FAR"),
)

# time_of_day bucket: 4-hour UTC buckets.
TIME_BUCKETS: tuple[tuple[int, int, str], ...] = (
    (0,  4,  "T_00_04"),
    (4,  8,  "T_04_08"),
    (8,  12, "T_08_12"),
    (12, 16, "T_12_16"),
    (16, 20, "T_16_20"),
    (20, 24, "T_20_24"),
)

# Columns the PatternGenerator may condition on (must be categorical/str).
BIN_COLUMNS: tuple[str, ...] = (
    "distance_to_vwap_bin",
    "distance_to_session_high_bin",
    "distance_to_session_low_bin",
    "momentum_3_bin",
    "momentum_5_bin",
    "range_compression_bin",
    "session_bin",
    "time_of_day_bucket",
)


class FeatureBuilder:
    """
    Computes discovery features per snapshot.

    Parameters
    ----------
    compression_lookback:
        Number of prior M1 bars used to compute the mean range for
        `range_compression`.  Default 10.
    sigma_window:
        Trailing window (in rows) used to compute rolling std for the
        z-score normalised columns.  Default 500.
    """

    def __init__(
        self,
        compression_lookback: int = 10,
        sigma_window: int = 500,
    ) -> None:
        self.compression_lookback = compression_lookback
        self.sigma_window = sigma_window

    # ─── Public API ───────────────────────────────────────────────────────────

    def build(self, snapshots: list[MarketSnapshot]) -> list[dict[str, Any]]:
        """Return a list of feature-rows, one per snapshot (same order)."""
        rows = [self._row(i, s) for i, s in enumerate(snapshots)]
        self._add_rolling_zscores(rows)
        return rows

    def build_compact_matrix(
        self,
        snapshots: list[MarketSnapshot],
        *,
        forward_bars: int = 10,
        execution_model: "ExecutionModel | None" = None,
    ):
        """
        Return a compact pandas DataFrame for fast pattern discovery.

        One row per snapshot.  Categorical bin columns are `category`
        dtype (int8 codes internally) to keep memory tight.

        Columns
        ───────
        bookkeeping  : timestamp, symbol, idx
        raw features : distance_to_vwap, distance_to_session_high,
                       distance_to_session_low, momentum_3, momentum_5,
                       range_compression, atr_14
        bins         : distance_to_vwap_bin, distance_to_session_high_bin,
                       distance_to_session_low_bin, momentum_3_bin,
                       momentum_5_bin, range_compression_bin, session_bin,
                       time_of_day_bucket
        forward      : forward_return_10, forward_win_10  (float32 / int8;
                       NaN/0 for the last `forward_bars` rows and rows where
                       the entry bar could not be simulated).

        Parameters
        ----------
        forward_bars:
            Number of snapshots to hold after entry.  Default 10.
        execution_model:
            Used for spread + slippage cost on entry/exit.  If None, the
            forward returns are computed WITHOUT transaction cost
            (useful for quick smoke tests).  In production runs pass a
            configured ExecutionModel.
        """
        import numpy as np
        import pandas as pd

        rows = self.build(snapshots)
        n = len(rows)
        if n == 0:
            return pd.DataFrame()

        close = np.array([r["close"] for r in rows], dtype=np.float64)
        open_ = np.array([r["open"] for r in rows], dtype=np.float64)
        high = np.array([r["high"] for r in rows], dtype=np.float64)
        low = np.array([r["low"] for r in rows], dtype=np.float64)
        atr_14 = np.array(
            [
                snap.feature_vector.atr_14
                if snap.feature_vector.atr_14 is not None
                else np.nan
                for snap in snapshots
            ],
            dtype=np.float64,
        )

        fwd_return, fwd_win = self._precompute_forward(
            snapshots,
            close,
            open_,
            high,
            low,
            atr_14,
            forward_bars=forward_bars,
            execution_model=execution_model,
        )

        df = pd.DataFrame(
            {
                "idx": np.arange(n, dtype=np.int32),
                "timestamp": [r["timestamp_utc"] for r in rows],
                "symbol": [snap.symbol for snap in snapshots],
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "distance_to_vwap": [
                    r["distance_to_vwap"] if r["distance_to_vwap"] is not None else np.nan
                    for r in rows
                ],
                "distance_to_session_high": [
                    r["distance_to_session_high"]
                    if r["distance_to_session_high"] is not None
                    else np.nan
                    for r in rows
                ],
                "distance_to_session_low": [
                    r["distance_to_session_low"]
                    if r["distance_to_session_low"] is not None
                    else np.nan
                    for r in rows
                ],
                "momentum_3": [
                    r["momentum_3"] if r["momentum_3"] is not None else np.nan
                    for r in rows
                ],
                "momentum_5": [
                    r["momentum_5"] if r["momentum_5"] is not None else np.nan
                    for r in rows
                ],
                "range_compression": [
                    r["range_compression"] if r["range_compression"] is not None else False
                    for r in rows
                ],
                "atr_14": atr_14,
                "forward_return_10": fwd_return.astype(np.float32),
                "forward_win_10": fwd_win.astype(np.int8),
            }
        )

        # ─── Derived bins ──────────────────────────────────────────────────────
        z_vwap = np.array(
            [r.get("distance_to_vwap_z") for r in rows], dtype=object
        )
        z_hi = np.array(
            [r.get("distance_to_session_high_z") for r in rows], dtype=object
        )
        z_lo = np.array(
            [r.get("distance_to_session_low_z") for r in rows], dtype=object
        )

        df["distance_to_vwap_bin"] = self._bin_by_edges(
            z_vwap, VWAP_BINS, missing="MID"
        )
        df["distance_to_session_high_bin"] = self._bin_abs(
            z_hi, DIST_SESSION_BINS, missing="MID"
        )
        df["distance_to_session_low_bin"] = self._bin_abs(
            z_lo, DIST_SESSION_BINS, missing="MID"
        )
        df["momentum_3_bin"] = np.where(df["momentum_3"] >= 0, "POS", "NEG")
        df["momentum_5_bin"] = np.where(df["momentum_5"] >= 0, "POS", "NEG")
        df["range_compression_bin"] = np.where(
            df["range_compression"], "TRUE", "FALSE"
        )
        df["session_bin"] = [
            _detect_session_utc(ts) for ts in df["timestamp"]
        ]
        hours = np.array([ts.hour for ts in df["timestamp"]], dtype=np.int16)
        df["time_of_day_bucket"] = self._bucket_hours(hours, TIME_BUCKETS)

        # ─── Memory-tight dtypes ───────────────────────────────────────────────
        for col in BIN_COLUMNS:
            df[col] = df[col].astype("category")
        for col in (
            "open",
            "high",
            "low",
            "close",
            "distance_to_vwap",
            "distance_to_session_high",
            "distance_to_session_low",
            "momentum_3",
            "momentum_5",
            "atr_14",
        ):
            df[col] = df[col].astype(np.float32)
        df["range_compression"] = df["range_compression"].astype(bool)

        return df

    # ─── Forward-return vectorised precomputation ─────────────────────────────

    def _precompute_forward(
        self,
        snapshots: list[MarketSnapshot],
        close,
        open_,
        high,
        low,
        atr_14,
        *,
        forward_bars: int,
        execution_model: "ExecutionModel | None",
    ):
        import numpy as np

        n = len(snapshots)
        fwd_return = np.full(n, np.nan, dtype=np.float64)
        fwd_win = np.zeros(n, dtype=np.int8)

        if n < forward_bars + 2:
            return fwd_return, fwd_win

        # Vectorised spread + deterministic slippage.  We skip the
        # uniform(0.8, 1.2) jitter here: the discovery engine only needs
        # expected cost, and determinism makes the matrix reproducible.
        if execution_model is None:
            spread = np.zeros(n, dtype=np.float64)
            slippage = np.zeros(n, dtype=np.float64)
        else:
            from aion.execution.execution_model import (
                ENTRY_MULTIPLIERS,
                SESSION_MULTIPLIERS,
            )

            # Spread (per symbol, from first snapshot).
            symbol = snapshots[0].symbol
            params = execution_model.params_for(symbol)
            dynamic = 0.02 * np.where(
                np.isfinite(atr_14) & (atr_14 > 0), atr_14, 0.0
            )
            spread = np.maximum(params.min_spread, dynamic)

            # Slippage (retest entry).
            rng = high - low
            momentum = np.where(
                rng > 0, np.abs(close - open_) / np.maximum(rng, 1e-6), 0.0
            )
            base = params.k_vol * rng + params.k_mom * momentum * rng
            session_mult = np.array(
                [
                    SESSION_MULTIPLIERS.get(_detect_session_utc(s.latest_bar.timestamp_utc), 1.0)
                    for s in snapshots
                ],
                dtype=np.float64,
            )
            entry_mult = ENTRY_MULTIPLIERS.get("retest", 1.0)
            slippage = np.maximum(base * session_mult * entry_mult, 0.0)

        # Entry at i+1 open; exit at i+1+N close.
        entry_idx = np.arange(n) + 1
        exit_idx = entry_idx + forward_bars
        mask = exit_idx < n
        ei = entry_idx[mask]
        xi = exit_idx[mask]

        entry_price = open_[ei] + spread[ei] / 2.0 + slippage[ei]
        exit_price = close[xi] - spread[xi] / 2.0 - slippage[xi]
        valid = entry_price > 0

        ret = np.full_like(entry_price, np.nan, dtype=np.float64)
        ret[valid] = (exit_price[valid] - entry_price[valid]) / entry_price[valid]

        idx_valid = np.where(mask)[0]
        fwd_return[idx_valid] = ret
        fwd_win[idx_valid] = (ret > 0).astype(np.int8)
        return fwd_return, fwd_win

    # ─── Binning helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _bin_by_edges(
        values,
        edges: tuple[tuple[float, float, str], ...],
        *,
        missing: str,
    ):
        import numpy as np

        out = np.empty(len(values), dtype=object)
        for i, v in enumerate(values):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                out[i] = missing
                continue
            for lo, hi, label in edges:
                if lo <= v < hi:
                    out[i] = label
                    break
            else:
                out[i] = missing
        return out

    @staticmethod
    def _bin_abs(
        values,
        edges: tuple[tuple[float, float, str], ...],
        *,
        missing: str,
    ):
        import numpy as np

        out = np.empty(len(values), dtype=object)
        for i, v in enumerate(values):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                out[i] = missing
                continue
            a = abs(v)
            for lo, hi, label in edges:
                if lo <= a < hi:
                    out[i] = label
                    break
            else:
                out[i] = missing
        return out

    @staticmethod
    def _bucket_hours(hours, buckets: tuple[tuple[int, int, str], ...]):
        import numpy as np

        out = np.empty(len(hours), dtype=object)
        for i, h in enumerate(hours):
            for lo, hi, label in buckets:
                if lo <= h < hi:
                    out[i] = label
                    break
            else:
                out[i] = buckets[-1][2]
        return out

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _row(self, idx: int, snap: MarketSnapshot) -> dict[str, Any]:
        bar = snap.latest_bar
        fv = snap.feature_vector
        bars_m1 = snap.bars_m1

        close = bar.close
        ts = bar.timestamp_utc
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        # ── distance_to_vwap ────────────────────────────────────────────────
        vwap = fv.vwap_session
        distance_to_vwap = (close - vwap) if vwap is not None else None

        # ── distance_to_session_high / low ──────────────────────────────────
        # Prefer snapshot feature_vector when present; otherwise compute.
        distance_to_session_high = fv.distance_to_session_high
        if distance_to_session_high is None and fv.session_high is not None:
            distance_to_session_high = close - fv.session_high

        distance_to_session_low = fv.distance_to_session_low
        if distance_to_session_low is None and fv.session_low is not None:
            distance_to_session_low = close - fv.session_low

        # ── momentum_3 / momentum_5 ─────────────────────────────────────────
        momentum_3 = self._momentum(bars_m1, 3)
        momentum_5 = self._momentum(bars_m1, 5)

        # ── range_compression ───────────────────────────────────────────────
        range_compression = self._range_compression(bars_m1)

        # ── session + time_of_day ───────────────────────────────────────────
        session = snap.session_context.session_name.value
        time_of_day = ts.hour * 60 + ts.minute

        return {
            "idx": idx,
            "timestamp_utc": ts,
            "close": close,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "session": session,
            "time_of_day": time_of_day,
            "distance_to_vwap": distance_to_vwap,
            "distance_to_session_high": distance_to_session_high,
            "distance_to_session_low": distance_to_session_low,
            "momentum_3": momentum_3,
            "momentum_5": momentum_5,
            "range_compression": range_compression,
        }

    @staticmethod
    def _momentum(bars: list[Any], n: int) -> float | None:
        if len(bars) <= n:
            return None
        return bars[-1].close - bars[-1 - n].close

    def _range_compression(self, bars: list[Any]) -> bool | None:
        """True if current bar's range is smaller than the mean of the prior N."""
        n = self.compression_lookback
        if len(bars) < n + 1:
            return None
        current = bars[-1].high - bars[-1].low
        prior = bars[-1 - n:-1]
        if not prior:
            return None
        mean_range = sum(b.high - b.low for b in prior) / len(prior)
        return current < mean_range

    # ─── Rolling z-score normalisation ────────────────────────────────────────

    def _add_rolling_zscores(self, rows: list[dict[str, Any]]) -> None:
        """
        Populate `{feature}_z` columns using a trailing rolling std window.

        For each row at index i:
            sigma_i = std(values[max(0, i - W + 1):i + 1]) over non-None values
            z_i     = value_i / sigma_i   (None if sigma_i == 0 or missing)

        No lookahead: only values at index ≤ i are used.
        The first rows are skipped (z = None) until the window has at
        least `min_obs` valid samples.
        """
        w = self.sigma_window
        min_obs = max(30, w // 10)

        for feat in CONTINUOUS_FEATURE_NAMES:
            z_col = f"{feat}_z"
            values: list[float | None] = [r.get(feat) for r in rows]

            for i, row in enumerate(rows):
                v = values[i]
                if v is None:
                    row[z_col] = None
                    continue
                window_slice = values[max(0, i - w + 1): i + 1]
                observed = [x for x in window_slice if x is not None]
                if len(observed) < min_obs:
                    row[z_col] = None
                    continue
                mean = sum(observed) / len(observed)
                var = sum((x - mean) ** 2 for x in observed) / len(observed)
                if var <= 0:
                    row[z_col] = None
                    continue
                sigma = math.sqrt(var)
                row[z_col] = v / sigma
