"""
tests/unit/test_regime_rules.py
─────────────────────────────────
Unit tests for aion.regime.rules.RuleBasedRegimeDetector.

Tests verify:
  - UNKNOWN when volatility_percentile_20 is None
  - COMPRESSION when vp < LOW_PERCENTILE (0.20)
  - RANGE when LOW_PERCENTILE <= vp < HIGH_PERCENTILE (0.20 – 0.75)
  - TREND_UP when vp >= HIGH_PERCENTILE and return_5 > TREND_RETURN_THRESHOLD
  - TREND_DOWN when vp >= HIGH_PERCENTILE and return_5 < -TREND_RETURN_THRESHOLD
  - EXPANSION when vp >= HIGH_PERCENTILE and |return_5| <= threshold
  - EXPANSION when vp >= HIGH_PERCENTILE and return_5 is None
  - Threshold boundary behaviour (exact boundary values)
  - Confidence is in [0.0, 1.0] for all labels
  - RegimeResult is frozen (immutable)
  - detector_id is the expected string
  - model_version is propagated correctly
  - Custom thresholds respected
  - All RegimeLabel values are strings (enum consistency)
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aion.core.constants import FEATURE_SET_VERSION, SNAPSHOT_VERSION
from aion.core.enums import AssetClass, DataSource, RegimeLabel, SessionName, Timeframe
from aion.core.ids import new_snapshot_id
from aion.core.models import (
    DataQualityReport,
    FeatureVector,
    InstrumentSpec,
    MarketBar,
    MarketSnapshot,
    SessionContext,
)
from aion.regime.base import RegimeResult
from aion.regime.rules import (
    HIGH_PERCENTILE,
    LOW_PERCENTILE,
    TREND_RETURN_THRESHOLD,
    RuleBasedRegimeDetector,
)

# ─────────────────────────────────────────────────────────────────────────────
# Snapshot builder helpers (minimal — regime only needs feature_vector)
# ─────────────────────────────────────────────────────────────────────────────

_UTC = timezone.utc
_TS = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_UTC)


def _make_snapshot(
    volatility_percentile_20: float | None = 0.50,
    return_5: float | None = 0.0001,
) -> MarketSnapshot:
    bar = MarketBar(
        symbol="EURUSD",
        timestamp_utc=_TS,
        timestamp_market=_TS,
        timeframe=Timeframe.M1,
        open=1.1000,
        high=1.1010,
        low=1.0990,
        close=1.1005,
        tick_volume=100.0,
        real_volume=0.0,
        spread=2.0,
        source=DataSource.CSV,
    )
    fv = FeatureVector(
        symbol="EURUSD",
        timestamp_utc=_TS,
        timeframe=Timeframe.M1,
        atr_14=0.00015,
        rolling_range_10=0.0010,
        rolling_range_20=0.0012,
        volatility_percentile_20=volatility_percentile_20,
        session_high=1.1060,
        session_low=1.0990,
        opening_range_high=1.1020,
        opening_range_low=1.1000,
        vwap_session=1.1010,
        spread_mean_20=2.0,
        spread_zscore_20=0.0,
        return_1=0.0001,
        return_5=return_5,
        candle_body=0.00005,
        upper_wick=0.00005,
        lower_wick=0.00005,
        distance_to_session_high=-0.0040,
        distance_to_session_low=0.0010,
        feature_set_version=FEATURE_SET_VERSION,
    )
    qr = DataQualityReport(
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        rows_checked=100,
        missing_bars=0,
        duplicate_timestamps=0,
        out_of_order_rows=0,
        stale_bars=0,
        spike_bars=0,
        null_rows=0,
        quality_score=1.0,
        warnings=[],
    )
    instr = InstrumentSpec(
        symbol="EURUSD",
        broker_symbol="EURUSD",
        asset_class=AssetClass.FOREX,
        price_timezone="Etc/UTC",
        market_timezone="Etc/UTC",
        broker_timezone="Etc/UTC",
        tick_size=0.00001,
        point_value=10.0,
        contract_size=100_000.0,
        min_lot=0.01,
        lot_step=0.01,
        currency_profit="USD",
        currency_margin="USD",
        session_calendar="forex_standard",
        trading_hours_label="Sun 22:00 - Fri 22:00 UTC",
    )
    ctx = SessionContext(
        trading_day=_TS.date(),
        broker_time=_TS,
        market_time=_TS,
        local_time=_TS,
        is_asia=False,
        is_london=True,
        is_new_york=False,
        is_session_open_window=True,
        opening_range_active=False,
        opening_range_completed=True,
        session_name=SessionName.LONDON,
        session_open_utc=_TS.replace(hour=8, minute=0, second=0),
        session_close_utc=_TS.replace(hour=16, minute=30, second=0),
    )
    return MarketSnapshot(
        snapshot_id=new_snapshot_id(),
        symbol="EURUSD",
        timestamp_utc=_TS,
        base_timeframe=Timeframe.M1,
        instrument=instr,
        session_context=ctx,
        latest_bar=bar,
        bars_m1=[bar],
        bars_m5=[],
        bars_m15=[],
        feature_vector=fv,
        quality_report=qr,
        snapshot_version=SNAPSHOT_VERSION,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def detector() -> RuleBasedRegimeDetector:
    return RuleBasedRegimeDetector()


# ─────────────────────────────────────────────────────────────────────────────
# Detector identity
# ─────────────────────────────────────────────────────────────────────────────


def test_detector_id(detector):
    assert detector.detector_id == "rule_based_v1"


def test_model_version_is_string(detector):
    result = detector.detect(_make_snapshot())
    assert isinstance(result.model_version, str)
    assert len(result.model_version) > 0


# ─────────────────────────────────────────────────────────────────────────────
# UNKNOWN
# ─────────────────────────────────────────────────────────────────────────────


def test_unknown_when_volatility_percentile_none(detector):
    result = detector.detect(_make_snapshot(volatility_percentile_20=None))
    assert result.label == RegimeLabel.UNKNOWN
    assert result.confidence == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# COMPRESSION
# ─────────────────────────────────────────────────────────────────────────────


def test_compression_on_low_percentile(detector):
    result = detector.detect(_make_snapshot(volatility_percentile_20=0.10))
    assert result.label == RegimeLabel.COMPRESSION


def test_compression_just_below_boundary(detector):
    result = detector.detect(_make_snapshot(volatility_percentile_20=0.19))
    assert result.label == RegimeLabel.COMPRESSION


def test_compression_confidence_is_high_at_zero_percentile(detector):
    """vp=0 → confidence should be 1.0 (maximum compression)."""
    result = detector.detect(_make_snapshot(volatility_percentile_20=0.0))
    assert result.confidence == pytest.approx(1.0)


def test_compression_confidence_decreases_toward_boundary(detector):
    """Closer to LOW_PERCENTILE → lower confidence in COMPRESSION."""
    r_far = detector.detect(_make_snapshot(volatility_percentile_20=0.05))
    r_near = detector.detect(_make_snapshot(volatility_percentile_20=0.18))
    assert r_far.confidence > r_near.confidence


# ─────────────────────────────────────────────────────────────────────────────
# RANGE
# ─────────────────────────────────────────────────────────────────────────────


def test_range_on_mid_percentile(detector):
    result = detector.detect(_make_snapshot(volatility_percentile_20=0.475))
    assert result.label == RegimeLabel.RANGE


def test_range_at_low_percentile_boundary(detector):
    """Exactly at LOW_PERCENTILE (0.20) → RANGE (not COMPRESSION)."""
    result = detector.detect(_make_snapshot(volatility_percentile_20=LOW_PERCENTILE))
    assert result.label == RegimeLabel.RANGE


def test_range_just_below_high_boundary(detector):
    """Just below HIGH_PERCENTILE → RANGE."""
    result = detector.detect(_make_snapshot(volatility_percentile_20=0.74))
    assert result.label == RegimeLabel.RANGE


def test_range_confidence_highest_at_midpoint(detector):
    """Midpoint between LOW and HIGH → highest RANGE confidence."""
    mid = (LOW_PERCENTILE + HIGH_PERCENTILE) / 2.0
    r_mid = detector.detect(_make_snapshot(volatility_percentile_20=mid))
    r_near_low = detector.detect(_make_snapshot(volatility_percentile_20=0.21))
    assert r_mid.confidence > r_near_low.confidence


# ─────────────────────────────────────────────────────────────────────────────
# TREND_UP
# ─────────────────────────────────────────────────────────────────────────────


def test_trend_up_on_high_vol_positive_return(detector):
    result = detector.detect(
        _make_snapshot(volatility_percentile_20=0.80, return_5=0.0005)
    )
    assert result.label == RegimeLabel.TREND_UP


def test_trend_up_at_high_percentile_boundary(detector):
    """Exactly at HIGH_PERCENTILE + positive return → TREND_UP."""
    result = detector.detect(
        _make_snapshot(
            volatility_percentile_20=HIGH_PERCENTILE,
            return_5=TREND_RETURN_THRESHOLD + 0.0001,
        )
    )
    assert result.label == RegimeLabel.TREND_UP


def test_trend_up_confidence_increases_with_larger_return(detector):
    """Larger |return_5| → higher TREND_UP confidence."""
    r_small = detector.detect(
        _make_snapshot(volatility_percentile_20=0.80, return_5=0.0003)
    )
    r_large = detector.detect(
        _make_snapshot(volatility_percentile_20=0.80, return_5=0.0010)
    )
    assert r_large.confidence > r_small.confidence


# ─────────────────────────────────────────────────────────────────────────────
# TREND_DOWN
# ─────────────────────────────────────────────────────────────────────────────


def test_trend_down_on_high_vol_negative_return(detector):
    result = detector.detect(
        _make_snapshot(volatility_percentile_20=0.80, return_5=-0.0005)
    )
    assert result.label == RegimeLabel.TREND_DOWN


def test_trend_down_just_below_threshold_return(detector):
    """return_5 = -(threshold + epsilon) → TREND_DOWN."""
    result = detector.detect(
        _make_snapshot(
            volatility_percentile_20=0.80,
            return_5=-(TREND_RETURN_THRESHOLD + 0.0001),
        )
    )
    assert result.label == RegimeLabel.TREND_DOWN


# ─────────────────────────────────────────────────────────────────────────────
# EXPANSION
# ─────────────────────────────────────────────────────────────────────────────


def test_expansion_on_high_vol_flat_return(detector):
    """High vol + |return_5| <= threshold → EXPANSION."""
    result = detector.detect(
        _make_snapshot(volatility_percentile_20=0.80, return_5=0.0001)
    )
    assert result.label == RegimeLabel.EXPANSION


def test_expansion_when_return5_none_and_high_vol(detector):
    """High vol + return_5=None → EXPANSION (direction unknown)."""
    result = detector.detect(
        _make_snapshot(volatility_percentile_20=0.85, return_5=None)
    )
    assert result.label == RegimeLabel.EXPANSION


def test_expansion_at_high_percentile_boundary_flat_return(detector):
    """Exactly at HIGH_PERCENTILE with small return → EXPANSION."""
    result = detector.detect(
        _make_snapshot(
            volatility_percentile_20=HIGH_PERCENTILE,
            return_5=TREND_RETURN_THRESHOLD - 0.0001,  # just below threshold
        )
    )
    assert result.label == RegimeLabel.EXPANSION


# ─────────────────────────────────────────────────────────────────────────────
# Confidence range
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "vp,r5",
    [
        (None, 0.0),               # UNKNOWN
        (0.05, 0.0),               # COMPRESSION
        (0.0, 0.0),                # COMPRESSION extreme
        (0.475, 0.0001),           # RANGE midpoint
        (0.21, 0.0001),            # RANGE near low boundary
        (0.74, 0.0001),            # RANGE near high boundary
        (0.80, 0.0008),            # TREND_UP
        (0.80, -0.0008),           # TREND_DOWN
        (0.80, 0.0001),            # EXPANSION
        (0.85, None),              # EXPANSION no return
        (1.0, 0.0020),             # TREND_UP extreme
    ],
)
def test_confidence_always_in_unit_interval(vp, r5):
    detector = RuleBasedRegimeDetector()
    result = detector.detect(_make_snapshot(volatility_percentile_20=vp, return_5=r5))
    assert 0.0 <= result.confidence <= 1.0, (
        f"Confidence {result.confidence} out of range for vp={vp}, r5={r5}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# RegimeResult is frozen
# ─────────────────────────────────────────────────────────────────────────────


def test_regime_result_is_frozen(detector):
    result = detector.detect(_make_snapshot())
    with pytest.raises(Exception):
        result.label = RegimeLabel.UNKNOWN  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# Custom thresholds
# ─────────────────────────────────────────────────────────────────────────────


def test_custom_low_threshold():
    """With low_percentile=0.30, vp=0.25 should be COMPRESSION."""
    detector = RuleBasedRegimeDetector(low_percentile=0.30)
    result = detector.detect(_make_snapshot(volatility_percentile_20=0.25))
    assert result.label == RegimeLabel.COMPRESSION


def test_custom_high_threshold():
    """With high_percentile=0.60, vp=0.65 + positive return → TREND_UP."""
    detector = RuleBasedRegimeDetector(high_percentile=0.60)
    result = detector.detect(
        _make_snapshot(volatility_percentile_20=0.65, return_5=0.0005)
    )
    assert result.label == RegimeLabel.TREND_UP


def test_custom_trend_return_threshold():
    """Higher trend threshold: return_5=0.0003 that normally is TREND is now EXPANSION."""
    detector = RuleBasedRegimeDetector(trend_return_threshold=0.0010)
    result = detector.detect(
        _make_snapshot(volatility_percentile_20=0.80, return_5=0.0003)
    )
    assert result.label == RegimeLabel.EXPANSION


# ─────────────────────────────────────────────────────────────────────────────
# Enum consistency
# ─────────────────────────────────────────────────────────────────────────────


def test_regime_label_values_are_strings():
    """All RegimeLabel enum values are strings (JSON-serialisation safe)."""
    for label in RegimeLabel:
        assert isinstance(label.value, str)
