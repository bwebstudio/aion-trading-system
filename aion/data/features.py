"""
aion.data.features
───────────────────
Compute a FeatureVector from a sequence of MarketBar objects.

No-lookahead guarantee:
  Every feature at time T uses only data from bars[0..T] (inclusive).
  Specifically:
  - Rolling statistics use pandas `.rolling()` with no negative shifts.
  - Returns use `.shift(1)` (previous bar), never future bars.
  - Session features filter on `timestamp_utc >= session_open_utc` (past).

Window / min_periods decisions:
  - ATR-14, rolling_range_10, rolling_range_20, volatility_percentile_20:
      min_periods = full window size.
      Rationale: these indicators are only meaningful with a complete window.
      With fewer bars, return None rather than a misleading partial value.
  - spread_mean_20, spread_zscore_20:
      min_periods = 1 for mean, 2 for std.
      Rationale: spread mean is informative even with 1 bar.
  - candle_body, upper_wick, lower_wick:
      No rolling window — computed from the current bar only.
  - return_1, return_5:
      Require 2 and 6 bars respectively (implicit via shift).

ATR note:
  We use a simple rolling mean of True Range (not Wilder's smoothed EMA).
  Rationale: deterministic, reproducible, easier to understand and test.
  The difference is small for window sizes >= 14.  Wilder's EMA can be
  added as an optional variant in the research environment.

VWAP note:
  Typical price = (high + low + close) / 3.
  VWAP = Σ(typical_price × tick_volume) / Σ(tick_volume).
  Reset at each session open.  Returns None if session is OFF_HOURS.

Rules:
  - All functions are pure.
  - Uses pandas/numpy internally, but returns domain models (not DataFrames).
  - All feature values are float | None.  Never return NaN in the model.
"""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import pandas as pd

from aion.core.constants import (
    ATR_PERIOD,
    FEATURE_SET_VERSION,
    OPENING_RANGE_MINUTES,
    RETURN_LONG_BARS,
    RETURN_SHORT_BARS,
    ROLLING_RANGE_LONG,
    ROLLING_RANGE_SHORT,
    SPREAD_LOOKBACK,
    VOLATILITY_PERCENTILE_LOOKBACK,
)
from aion.core.enums import Timeframe
from aion.core.models import FeatureVector, MarketBar, SessionContext


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def compute_feature_vector(
    bars: list[MarketBar],
    session_context: SessionContext,
    timeframe: Timeframe,
) -> FeatureVector:
    """
    Compute a FeatureVector for the most recent bar in `bars`.

    Parameters
    ----------
    bars:
        Sorted ascending by timestamp_utc.  The last bar is the one
        features are computed FOR.  Earlier bars provide context.
        At minimum 1 bar is required; features requiring more bars
        will return None.
    session_context:
        The session context of the current bar.  Used for session-relative
        features (high/low, VWAP, opening range).
    timeframe:
        Timeframe of the provided bars.

    Returns
    -------
    FeatureVector
        All feature values as float | None.  Never raises on empty input.
    """
    if not bars:
        return _empty_feature_vector("UNKNOWN", session_context, timeframe)

    current = bars[-1]
    df = _bars_to_dataframe(bars)

    df = _add_atr(df)
    df = _add_rolling_ranges(df)
    df = _add_returns(df)
    df = _add_spread_features(df)
    df = _add_candle_features(df)
    df = _add_volatility_percentile(df)

    last = df.iloc[-1]

    session_feats = _compute_session_features(bars, session_context)
    s_high = session_feats.get("session_high")
    s_low = session_feats.get("session_low")

    return FeatureVector(
        symbol=current.symbol,
        timestamp_utc=current.timestamp_utc,
        timeframe=timeframe,
        # Volatility
        atr_14=_safe(last.get("atr_14")),
        rolling_range_10=_safe(last.get("rolling_range_10")),
        rolling_range_20=_safe(last.get("rolling_range_20")),
        volatility_percentile_20=_safe(last.get("volatility_percentile_20")),
        # Session
        session_high=s_high,
        session_low=s_low,
        opening_range_high=session_feats.get("opening_range_high"),
        opening_range_low=session_feats.get("opening_range_low"),
        vwap_session=session_feats.get("vwap_session"),
        # Spread
        spread_mean_20=_safe(last.get("spread_mean_20")),
        spread_zscore_20=_safe(last.get("spread_zscore_20")),
        # Returns
        return_1=_safe(last.get("return_1")),
        return_5=_safe(last.get("return_5")),
        # Candle structure
        candle_body=_safe(last.get("candle_body")),
        upper_wick=_safe(last.get("upper_wick")),
        lower_wick=_safe(last.get("lower_wick")),
        # Distance to session extremes
        distance_to_session_high=_distance(current.close, s_high),
        distance_to_session_low=_distance(current.close, s_low),
        feature_set_version=FEATURE_SET_VERSION,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame construction
# ─────────────────────────────────────────────────────────────────────────────


def _bars_to_dataframe(bars: list[MarketBar]) -> pd.DataFrame:
    """Convert a bar list to a numeric DataFrame for rolling calculations."""
    rows = [
        {
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "tick_volume": b.tick_volume,
            "spread": b.spread,
        }
        for b in bars
    ]
    return pd.DataFrame(rows, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation helpers (each returns a new DataFrame)
# ─────────────────────────────────────────────────────────────────────────────


def _add_atr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ATR-14 using simple rolling mean of True Range.

    True Range = max(H-L, |H - prev_C|, |L - prev_C|)
    min_periods = ATR_PERIOD: returns NaN for the first ATR_PERIOD-1 rows.
    """
    df = df.copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr_14"] = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()
    return df


def _add_rolling_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling high-to-low range over the last N bars (including current).

    min_periods = full window: partial windows return NaN.
    No lookahead: the window looks back, never forward.
    """
    df = df.copy()
    df["rolling_range_10"] = (
        df["high"].rolling(ROLLING_RANGE_SHORT, min_periods=ROLLING_RANGE_SHORT).max()
        - df["low"].rolling(ROLLING_RANGE_SHORT, min_periods=ROLLING_RANGE_SHORT).min()
    )
    df["rolling_range_20"] = (
        df["high"].rolling(ROLLING_RANGE_LONG, min_periods=ROLLING_RANGE_LONG).max()
        - df["low"].rolling(ROLLING_RANGE_LONG, min_periods=ROLLING_RANGE_LONG).min()
    )
    return df


def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log returns over 1 and 5 bars.

    return_N = log(close[t] / close[t-N])

    shift(N) accesses PAST bars only → no lookahead.
    NaN for the first N rows (no previous close available).
    """
    df = df.copy()
    df["return_1"] = np.log(df["close"] / df["close"].shift(RETURN_SHORT_BARS))
    df["return_5"] = np.log(df["close"] / df["close"].shift(RETURN_LONG_BARS))
    return df


def _add_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling spread mean and z-score over the last SPREAD_LOOKBACK bars.

    spread_mean: min_periods=1 (useful even with 1 bar).
    spread_zscore: NaN when std is 0 (constant spread) or window < 2.
    """
    df = df.copy()
    mean = df["spread"].rolling(SPREAD_LOOKBACK, min_periods=1).mean()
    std = df["spread"].rolling(SPREAD_LOOKBACK, min_periods=2).std()

    # Avoid division by zero for constant-spread instruments
    std_safe = std.replace(0.0, float("nan"))

    df["spread_mean_20"] = mean
    df["spread_zscore_20"] = (df["spread"] - mean) / std_safe
    return df


def _add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single-bar candle structure features.

    candle_body  = |close - open|
    upper_wick   = high  - max(open, close)
    lower_wick   = min(open, close) - low

    All are non-negative by OHLC structural constraints.
    Computed from the current bar only — no rolling window.
    """
    df = df.copy()
    body_top = df[["open", "close"]].max(axis=1)
    body_bot = df[["open", "close"]].min(axis=1)

    df["candle_body"] = (df["close"] - df["open"]).abs()
    df["upper_wick"] = df["high"] - body_top
    df["lower_wick"] = body_bot - df["low"]
    return df


def _add_volatility_percentile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Percentile rank of the current ATR-14 within the last
    VOLATILITY_PERCENTILE_LOOKBACK values (including current bar).

    Returns a value in [0.0, 1.0]:
      0.0 = current ATR is lowest in the lookback window
      1.0 = current ATR is highest in the lookback window

    min_periods = full window: returns NaN until enough ATR values exist.
    No lookahead: the rank is the current value's position among past values.

    Rolling apply function receives a 1-D numpy array where values[-1]
    is the current bar and values[:-1] are past bars.
    """
    df = df.copy()

    if "atr_14" not in df.columns or df["atr_14"].isna().all():
        df["volatility_percentile_20"] = float("nan")
        return df

    def _rank_current_in_window(window: np.ndarray) -> float:
        """
        Rank the last element of `window` among all other elements.
        Returns NaN if window has fewer than 2 valid values.
        """
        if len(window) < 2:
            return float("nan")
        current = window[-1]
        past = window[:-1]
        valid_past = past[~np.isnan(past)]
        if len(valid_past) == 0:
            return float("nan")
        return float(np.sum(valid_past <= current) / len(valid_past))

    df["volatility_percentile_20"] = (
        df["atr_14"]
        .rolling(VOLATILITY_PERCENTILE_LOOKBACK, min_periods=VOLATILITY_PERCENTILE_LOOKBACK)
        .apply(_rank_current_in_window, raw=True)
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Session-based features
# ─────────────────────────────────────────────────────────────────────────────


def _compute_session_features(
    bars: list[MarketBar],
    session_context: SessionContext,
) -> dict[str, float | None]:
    """
    Compute session-relative features using only bars from the current session.

    All filtering is backwards-looking (bars <= current timestamp).
    Returns all None if the session is OFF_HOURS or no session open time is set.
    """
    null_result: dict[str, float | None] = {
        "session_high": None,
        "session_low": None,
        "opening_range_high": None,
        "opening_range_low": None,
        "vwap_session": None,
    }

    session_open = session_context.session_open_utc
    if session_open is None:
        return null_result

    session_bars = [b for b in bars if b.timestamp_utc >= session_open]
    if not session_bars:
        return null_result

    # Session high / low
    session_high: float = max(b.high for b in session_bars)
    session_low: float = min(b.low for b in session_bars)

    # Session VWAP: Σ(typical_price × tick_volume) / Σ(tick_volume)
    # typical_price = (H + L + C) / 3
    total_pv = sum(
        ((b.high + b.low + b.close) / 3.0) * b.tick_volume for b in session_bars
    )
    total_vol = sum(b.tick_volume for b in session_bars)
    vwap: float | None = total_pv / total_vol if total_vol > 0 else None

    # Opening range: bars from session_open to session_open + OPENING_RANGE_MINUTES
    # Returned for both active and completed states.
    # The session_context flags (opening_range_active, opening_range_completed)
    # tell consumers whether the range is final or still accumulating.
    or_end = session_open + timedelta(minutes=OPENING_RANGE_MINUTES)
    or_bars = [b for b in session_bars if b.timestamp_utc < or_end]

    opening_range_high: float | None = None
    opening_range_low: float | None = None
    if or_bars:
        opening_range_high = max(b.high for b in or_bars)
        opening_range_low = min(b.low for b in or_bars)

    return {
        "session_high": session_high,
        "session_low": session_low,
        "opening_range_high": opening_range_high,
        "opening_range_low": opening_range_low,
        "vwap_session": vwap,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────


def _safe(value: object) -> float | None:
    """Convert a pandas NaN or non-finite float to None; keep finite floats."""
    if value is None:
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _distance(price: float, level: float | None) -> float | None:
    """Signed distance from `price` to `level`.  None if level is None."""
    if level is None:
        return None
    return price - level


# ─────────────────────────────────────────────────────────────────────────────
# Batch computation (historical pipeline)
# ─────────────────────────────────────────────────────────────────────────────


def compute_feature_series(
    bars: list[MarketBar],
    timeframe: Timeframe,
    market_timezone: str,
    broker_timezone: str,
    local_timezone: str,
) -> list[FeatureVector]:
    """
    Compute one FeatureVector per bar in a single efficient pass.

    Use this in the historical pipeline instead of calling
    `compute_feature_vector` once per bar (which would be O(N²)).

    Complexity:
        Rolling features  — O(N) via one pandas DataFrame pass.
        Session features  — O(N) via incremental state tracking.

    Parameters
    ----------
    bars:
        M1 bars sorted ascending.  The FeatureVector at index i is computed
        using bars[0..i] only — no lookahead.
    timeframe:
        Timeframe of the provided bars.
    market_timezone / broker_timezone / local_timezone:
        IANA timezone names.  Used to build SessionContext per bar.

    Returns
    -------
    list[FeatureVector]
        Same length as `bars`.  Empty list if `bars` is empty.

    Session reset policy:
        Session state (VWAP, H/L, opening range) resets when
        `session_open_utc` changes.  LONDON → OVERLAP_LONDON_NY causes a
        reset at NY open (13:30 UTC winter / 13:30 UTC summer) because
        the OVERLAP session anchor is the NY session definition.
        This means VWAP for the LONDON-only portion (08:00–13:30) is
        separate from the OVERLAP/NY portion (13:30–close).
        This is a V1 simplification; a full London-day VWAP can be added
        as an additional feature in a future feature set version.
    """
    if not bars:
        return []

    # ── Step 1: compute all rolling stats in one pandas pass ─────────────────
    df = _bars_to_dataframe(bars)
    df = _add_atr(df)
    df = _add_rolling_ranges(df)
    df = _add_returns(df)
    df = _add_spread_features(df)
    df = _add_candle_features(df)
    df = _add_volatility_percentile(df)

    # ── Step 2: incremental session features ─────────────────────────────────
    session_feats = _compute_session_series_incremental(
        bars, market_timezone, broker_timezone, local_timezone
    )

    # ── Step 3: build one FeatureVector per bar ───────────────────────────────
    result: list[FeatureVector] = []
    for i, bar in enumerate(bars):
        row = df.iloc[i]
        sf = session_feats[i]
        s_high = sf.get("session_high")
        s_low = sf.get("session_low")

        result.append(
            FeatureVector(
                symbol=bar.symbol,
                timestamp_utc=bar.timestamp_utc,
                timeframe=timeframe,
                atr_14=_safe(row.get("atr_14")),
                rolling_range_10=_safe(row.get("rolling_range_10")),
                rolling_range_20=_safe(row.get("rolling_range_20")),
                volatility_percentile_20=_safe(row.get("volatility_percentile_20")),
                session_high=s_high,
                session_low=s_low,
                opening_range_high=sf.get("opening_range_high"),
                opening_range_low=sf.get("opening_range_low"),
                vwap_session=sf.get("vwap_session"),
                spread_mean_20=_safe(row.get("spread_mean_20")),
                spread_zscore_20=_safe(row.get("spread_zscore_20")),
                return_1=_safe(row.get("return_1")),
                return_5=_safe(row.get("return_5")),
                candle_body=_safe(row.get("candle_body")),
                upper_wick=_safe(row.get("upper_wick")),
                lower_wick=_safe(row.get("lower_wick")),
                distance_to_session_high=_distance(bar.close, s_high),
                distance_to_session_low=_distance(bar.close, s_low),
                feature_set_version=FEATURE_SET_VERSION,
            )
        )

    return result


def _compute_session_series_incremental(
    bars: list[MarketBar],
    market_timezone: str,
    broker_timezone: str,
    local_timezone: str,
) -> list[dict[str, float | None]]:
    """
    Compute session-relative feature dicts for all bars in O(N).

    Incremental tracking avoids O(N²) reprocessing.  Session state variables
    are reset whenever `session_open_utc` changes (new session detected).

    A sentinel value is used for the initial comparison so the first bar
    always triggers a proper reset.
    """
    # Import here to avoid making sessions a module-level dependency of features.
    from aion.data.sessions import build_session_context  # noqa: PLC0415

    _SENTINEL = object()  # unique sentinel — never equal to any datetime
    current_session_open: object = _SENTINEL

    # Incremental session accumulators
    s_high = float("-inf")
    s_low = float("inf")
    s_pv_sum = 0.0       # Σ(typical_price × tick_volume)
    s_vol_sum = 0.0      # Σ(tick_volume)
    or_high = float("-inf")
    or_low = float("inf")
    or_finalized = False  # True once we have passed the opening range window

    _null: dict[str, float | None] = {
        "session_high": None,
        "session_low": None,
        "vwap_session": None,
        "opening_range_high": None,
        "opening_range_low": None,
    }

    result: list[dict[str, float | None]] = []

    for bar in bars:
        ctx = build_session_context(
            bar.timestamp_utc, market_timezone, broker_timezone, local_timezone
        )
        new_key = ctx.session_open_utc  # None during OFF_HOURS

        # ── Session reset ─────────────────────────────────────────────────────
        if new_key != current_session_open:
            current_session_open = new_key
            s_high = float("-inf")
            s_low = float("inf")
            s_pv_sum = 0.0
            s_vol_sum = 0.0
            or_high = float("-inf")
            or_low = float("inf")
            or_finalized = False

        if current_session_open is None:
            result.append(dict(_null))
            continue

        # ── Update session accumulators ───────────────────────────────────────
        s_high = max(s_high, bar.high)
        s_low = min(s_low, bar.low)

        tp = (bar.high + bar.low + bar.close) / 3.0
        s_pv_sum += tp * bar.tick_volume
        s_vol_sum += bar.tick_volume
        vwap: float | None = s_pv_sum / s_vol_sum if s_vol_sum > 0 else None

        # ── Opening range accumulation ────────────────────────────────────────
        # current_session_open is a datetime-like here (not None, not sentinel).
        or_end = current_session_open + timedelta(minutes=OPENING_RANGE_MINUTES)

        if not or_finalized:
            if bar.timestamp_utc < or_end:
                or_high = max(or_high, bar.high)
                or_low = min(or_low, bar.low)
            else:
                or_finalized = True  # OR window has closed; values are now fixed

        result.append(
            {
                "session_high": s_high,
                "session_low": s_low,
                "vwap_session": vwap,
                "opening_range_high": or_high if or_high != float("-inf") else None,
                "opening_range_low": or_low if or_low != float("inf") else None,
            }
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────


def _empty_feature_vector(
    symbol: str,
    session_context: SessionContext,
    timeframe: Timeframe,
) -> FeatureVector:
    """Return a FeatureVector with all features set to None."""
    return FeatureVector(
        symbol=symbol,
        timestamp_utc=session_context.broker_time,
        timeframe=timeframe,
        atr_14=None,
        rolling_range_10=None,
        rolling_range_20=None,
        volatility_percentile_20=None,
        session_high=None,
        session_low=None,
        opening_range_high=None,
        opening_range_low=None,
        vwap_session=None,
        spread_mean_20=None,
        spread_zscore_20=None,
        return_1=None,
        return_5=None,
        candle_body=None,
        upper_wick=None,
        lower_wick=None,
        distance_to_session_high=None,
        distance_to_session_low=None,
        feature_set_version=FEATURE_SET_VERSION,
    )
