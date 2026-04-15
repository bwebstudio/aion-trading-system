"""
aion.data.validator
────────────────────
Data quality validation for a sequence of MarketBar objects.

Responsibilities:
  - Detect structural problems in a bar series.
  - Produce a DataQualityReport with an explicit quality_score.
  - Never silently drop bars — report and let the caller decide.

Rules:
  - All functions are pure (no side effects).
  - No bars are modified or removed.
  - Quality score formula is explicit and documented.

Gap counting note:
  All time-based gaps are counted, including expected session boundaries
  (weekends, forex overnight).  High missing_bars counts are normal for
  multi-day datasets.  Apply a session filter before validation if you
  want intra-session gap detection only.
"""

from __future__ import annotations

from datetime import timedelta

from aion.core.constants import (
    SPIKE_ATR_MULTIPLIER,
    STALE_BAR_CONSECUTIVE_THRESHOLD,
    TIMEFRAME_MINUTES,
)
from aion.core.enums import Timeframe
from aion.core.models import DataQualityReport, MarketBar


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def validate_bars(
    bars: list[MarketBar],
    timeframe: Timeframe,
) -> DataQualityReport:
    """
    Run all quality checks on `bars` and return a DataQualityReport.

    Parameters
    ----------
    bars:
        Sequence of MarketBar objects to validate.  May be unsorted.
    timeframe:
        The intended timeframe of the bars (used for gap detection).

    Returns
    -------
    DataQualityReport
        All counts set to zero and quality_score=0.0 if `bars` is empty.
    """
    symbol = bars[0].symbol if bars else "UNKNOWN"
    n = len(bars)
    warnings: list[str] = []

    if n == 0:
        return DataQualityReport(
            symbol=symbol,
            timeframe=timeframe,
            rows_checked=0,
            missing_bars=0,
            duplicate_timestamps=0,
            out_of_order_rows=0,
            stale_bars=0,
            spike_bars=0,
            null_rows=0,
            quality_score=0.0,
            warnings=["No bars provided."],
        )

    duplicates = _count_duplicates(bars)
    out_of_order = _count_out_of_order(bars)
    missing = _count_missing_bars(bars, timeframe)
    invalid_ohlc = _count_invalid_ohlc(bars)
    negative_spreads = _count_negative_spreads(bars)
    stale = _count_stale_bars(bars)
    spikes = _count_spikes(bars)
    null_prices = _count_null_bars(bars)

    if duplicates:
        warnings.append(f"{duplicates} duplicate timestamp(s) detected.")
    if out_of_order:
        warnings.append(f"{out_of_order} out-of-order row(s) detected.")
    if missing:
        warnings.append(
            f"{missing} missing bar gap(s) detected "
            f"(includes expected session boundaries)."
        )
    if invalid_ohlc:
        warnings.append(f"{invalid_ohlc} bar(s) with invalid OHLC structure.")
    if negative_spreads:
        warnings.append(f"{negative_spreads} bar(s) with negative spread.")
    if stale:
        warnings.append(
            f"{stale} bar(s) in stale sequences "
            f"(>= {STALE_BAR_CONSECUTIVE_THRESHOLD} consecutive identical OHLC)."
        )
    if spikes:
        warnings.append(
            f"{spikes} potential spike bar(s) "
            f"(range > {SPIKE_ATR_MULTIPLIER}x rolling mean range)."
        )
    if null_prices:
        warnings.append(f"{null_prices} bar(s) with zero or negative price(s).")

    quality_score = _compute_quality_score(
        n=n,
        duplicates=duplicates,
        out_of_order=out_of_order,
        missing_bars=missing,
        invalid_ohlc=invalid_ohlc,
        negative_spreads=negative_spreads,
        stale_bars=stale,
        spike_bars=spikes,
        null_rows=null_prices,
    )

    return DataQualityReport(
        symbol=symbol,
        timeframe=timeframe,
        rows_checked=n,
        missing_bars=missing,
        duplicate_timestamps=duplicates,
        out_of_order_rows=out_of_order,
        stale_bars=stale,
        spike_bars=spikes,
        null_rows=null_prices,
        quality_score=quality_score,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────


def _count_duplicates(bars: list[MarketBar]) -> int:
    """
    Count bars whose timestamp_utc appears more than once.

    Returns the number of *extra* occurrences (so 2 identical → count 1).
    """
    seen: set = set()
    count = 0
    for bar in bars:
        ts = bar.timestamp_utc
        if ts in seen:
            count += 1
        seen.add(ts)
    return count


def _count_out_of_order(bars: list[MarketBar]) -> int:
    """
    Count positions where bar[i].timestamp_utc >= bar[i+1].timestamp_utc.

    This catches both inverted order and equal consecutive timestamps
    that slipped past the duplicate check (e.g. different sources, same ts).
    """
    count = 0
    for i in range(len(bars) - 1):
        if bars[i].timestamp_utc >= bars[i + 1].timestamp_utc:
            count += 1
    return count


def _count_missing_bars(bars: list[MarketBar], timeframe: Timeframe) -> int:
    """
    Count expected bars that are absent from the series.

    For each consecutive pair, the expected gap is exactly one bar period.
    If the actual gap is N periods, (N-1) bars are missing.

    Note: includes expected gaps at session boundaries (weekends, holidays).
    """
    if len(bars) < 2:
        return 0

    expected_delta = timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    total_missing = 0

    for i in range(1, len(bars)):
        delta = bars[i].timestamp_utc - bars[i - 1].timestamp_utc
        if delta > expected_delta:
            # Round to nearest whole number of periods to absorb sub-minute drift
            n_periods = round(delta / expected_delta)
            total_missing += n_periods - 1

    return total_missing


def _count_invalid_ohlc(bars: list[MarketBar]) -> int:
    """
    Count bars marked is_valid=False by the normaliser.

    The normaliser already checks OHLC structural constraints (H>=L, H>=O/C, etc.).
    We trust that result here rather than re-running the same checks.
    """
    return sum(1 for b in bars if not b.is_valid)


def _count_negative_spreads(bars: list[MarketBar]) -> int:
    """
    Count bars where spread < 0.

    Negative spreads indicate a broker feed problem.
    Note: spread == 0 is legitimate (e.g. crypto, some demo accounts).
    """
    return sum(1 for b in bars if b.spread < 0)


def _count_null_bars(bars: list[MarketBar]) -> int:
    """
    Count bars where any OHLC price is zero or negative.

    Zero prices can slip through the OHLC validity check (0 >= 0 passes
    all structural constraints).  They indicate a dead or absent feed.
    """
    count = 0
    for bar in bars:
        if bar.open <= 0 or bar.high <= 0 or bar.low <= 0 or bar.close <= 0:
            count += 1
    return count


def _count_stale_bars(bars: list[MarketBar]) -> int:
    """
    Count bars that belong to "stale" sequences.

    A stale sequence is a run of STALE_BAR_CONSECUTIVE_THRESHOLD or more
    consecutive bars where open, high, low, close are all identical to the
    previous bar.  This indicates a frozen price feed.

    Returns the total number of bars in all qualifying sequences
    (each run of length N that qualifies contributes N bars to the count).
    """
    if len(bars) < STALE_BAR_CONSECUTIVE_THRESHOLD:
        return 0

    total = 0
    run = 1  # length of the current identical-OHLC run

    for i in range(1, len(bars)):
        prev, curr = bars[i - 1], bars[i]
        if (
            curr.open == prev.open
            and curr.high == prev.high
            and curr.low == prev.low
            and curr.close == prev.close
        ):
            run += 1
        else:
            if run >= STALE_BAR_CONSECUTIVE_THRESHOLD:
                total += run
            run = 1

    if run >= STALE_BAR_CONSECUTIVE_THRESHOLD:
        total += run

    return total


def _count_spikes(bars: list[MarketBar]) -> int:
    """
    Count bars where the high-low range is anomalously large.

    Method: for each bar at index i >= LOOKBACK, compute the rolling mean
    range of the previous LOOKBACK bars (excluding bar i).  If:
        range[i] > SPIKE_ATR_MULTIPLIER * mean_range[i-LOOKBACK : i]
    the bar is flagged as a spike.

    This is a simple heuristic — there will be false positives on genuine
    high-volatility events.  It is intentionally conservative (multiplier=10).

    No lookahead: the rolling window uses only bars BEFORE index i.
    """
    LOOKBACK = 20

    if len(bars) <= LOOKBACK:
        return 0

    count = 0
    for i in range(LOOKBACK, len(bars)):
        window_ranges = [b.high - b.low for b in bars[i - LOOKBACK : i]]
        mean_range = sum(window_ranges) / LOOKBACK
        current_range = bars[i].high - bars[i].low
        if mean_range > 0 and current_range > SPIKE_ATR_MULTIPLIER * mean_range:
            count += 1

    return count


# ─────────────────────────────────────────────────────────────────────────────
# Quality score
# ─────────────────────────────────────────────────────────────────────────────


def _compute_quality_score(
    n: int,
    duplicates: int,
    out_of_order: int,
    missing_bars: int,
    invalid_ohlc: int,
    negative_spreads: int,
    stale_bars: int,
    spike_bars: int,
    null_rows: int,
) -> float:
    """
    Compute a quality score in [0.0, 1.0].

    Formula:
        penalty = Σ (weight_i × count_i) / n
        score   = max(0.0, 1.0 - penalty)

    Severity weights (reflect impact on downstream decisions):
        null_rows         1.0  — critical: prices are meaningless
        duplicates        0.8  — high: corrupts all time-series calculations
        invalid_ohlc      0.8  — high: prices cannot be trusted
        out_of_order      0.5  — medium: sortable but indicates feed issues
        negative_spreads  0.5  — medium: broker feed problem
        missing_bars      0.3  — low-medium: gaps are expected at boundaries
        stale_bars        0.3  — low-medium: frozen feed
        spike_bars        0.2  — low: heuristic, may be genuine volatility

    The denominator is always `n` (number of bars present).
    Missing bars are penalised relative to the bars that ARE present,
    not relative to the total expected series length.  This avoids
    disproportionate penalties for weekend gaps in forex data.
    """
    if n == 0:
        return 0.0

    penalty = (
        1.0 * null_rows
        + 0.8 * duplicates
        + 0.8 * invalid_ohlc
        + 0.5 * out_of_order
        + 0.5 * negative_spreads
        + 0.3 * missing_bars
        + 0.3 * stale_bars
        + 0.2 * spike_bars
    ) / n

    return max(0.0, 1.0 - penalty)
