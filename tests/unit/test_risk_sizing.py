"""
tests/unit/test_risk_sizing.py
───────────────────────────────
Unit tests for aion.risk.sizing.

Tests verify:
  compute_risk_amount:
    - correct formula: equity * pct / 100
    - proportional to equity and risk percentage

  compute_position_size:
    - correct formula: risk_amount / (stop * point_value)
    - floors to lot_step (never exceeds risk budget)
    - enforces min_lot when raw result is below minimum
    - respects lot_step precision (no float artifacts)
    - different stops produce different sizes

  _lot_step_precision (internal):
    - correct precision for common lot_step values
"""

from __future__ import annotations

import pytest

from aion.core.enums import AssetClass
from aion.core.models import InstrumentSpec
from aion.risk.models import RiskProfile
from aion.risk.sizing import _lot_step_precision, compute_position_size, compute_risk_amount


# ─────────────────────────────────────────────────────────────────────────────
# Instrument fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _eurusd() -> InstrumentSpec:
    """Standard EURUSD: point_value=10 USD/pip/lot, min_lot=0.01, lot_step=0.01."""
    return InstrumentSpec(
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


def _us100() -> InstrumentSpec:
    """US100 index: point_value=1 USD/point/lot, min_lot=0.1, lot_step=0.1."""
    return InstrumentSpec(
        symbol="US100.cash",
        broker_symbol="US100.cash",
        asset_class=AssetClass.INDICES,
        price_timezone="America/New_York",
        market_timezone="America/New_York",
        broker_timezone="Etc/UTC",
        tick_size=0.01,
        point_value=1.0,
        contract_size=1.0,
        min_lot=0.1,
        lot_step=0.1,
        currency_profit="USD",
        currency_margin="USD",
        session_calendar="us_equity",
        trading_hours_label="Mon-Fri 09:30-16:00 ET",
    )


def _profile(equity: float = 10_000.0, risk_pct: float = 1.0) -> RiskProfile:
    return RiskProfile(account_equity=equity, max_risk_per_trade_pct=risk_pct)


# ─────────────────────────────────────────────────────────────────────────────
# compute_risk_amount
# ─────────────────────────────────────────────────────────────────────────────


def test_risk_amount_formula():
    """$10 000 * 1% = $100."""
    profile = _profile(equity=10_000.0, risk_pct=1.0)
    assert compute_risk_amount(profile) == pytest.approx(100.0)


def test_risk_amount_half_percent():
    """$10 000 * 0.5% = $50."""
    profile = _profile(equity=10_000.0, risk_pct=0.5)
    assert compute_risk_amount(profile) == pytest.approx(50.0)


def test_risk_amount_scales_with_equity():
    r1 = compute_risk_amount(_profile(equity=10_000.0, risk_pct=1.0))
    r2 = compute_risk_amount(_profile(equity=20_000.0, risk_pct=1.0))
    assert r2 == pytest.approx(r1 * 2)


def test_risk_amount_scales_with_pct():
    r1 = compute_risk_amount(_profile(equity=10_000.0, risk_pct=1.0))
    r2 = compute_risk_amount(_profile(equity=10_000.0, risk_pct=2.0))
    assert r2 == pytest.approx(r1 * 2)


# ─────────────────────────────────────────────────────────────────────────────
# compute_position_size — EURUSD
# ─────────────────────────────────────────────────────────────────────────────


def test_position_size_eurusd_10pip_stop():
    """$100 risk / (10 pips * $10/pip/lot) = 1.00 lot."""
    size = compute_position_size(
        risk_amount=100.0,
        stop_distance_points=10.0,
        instrument=_eurusd(),
    )
    assert size == pytest.approx(1.0)


def test_position_size_eurusd_20pip_stop():
    """$100 risk / (20 pips * $10/pip/lot) = 0.50 lot."""
    size = compute_position_size(
        risk_amount=100.0,
        stop_distance_points=20.0,
        instrument=_eurusd(),
    )
    assert size == pytest.approx(0.5)


def test_position_size_eurusd_50pip_stop():
    """$100 risk / (50 pips * $10/pip/lot) = 0.20 lot."""
    size = compute_position_size(
        risk_amount=100.0,
        stop_distance_points=50.0,
        instrument=_eurusd(),
    )
    assert size == pytest.approx(0.2)


def test_position_size_wider_stop_gives_smaller_size():
    """Wider stop -> smaller position (same risk amount)."""
    tight = compute_position_size(100.0, 10.0, _eurusd())
    wide = compute_position_size(100.0, 25.0, _eurusd())
    assert tight > wide


def test_position_size_respects_lot_step():
    """Result is always a multiple of lot_step."""
    size = compute_position_size(100.0, 15.0, _eurusd())
    instrument = _eurusd()
    steps = round(size / instrument.lot_step)
    assert abs(size - steps * instrument.lot_step) < 1e-9


def test_position_size_floors_not_rounds():
    """$100 / (15 pips * $10) = 0.6667 lots -> floored to 0.66, not rounded to 0.67."""
    size = compute_position_size(100.0, 15.0, _eurusd())
    raw = 100.0 / (15.0 * 10.0)
    assert size <= raw + 1e-9  # never exceeds raw lots


def test_position_size_enforces_min_lot():
    """Tiny risk amount -> raw lots < min_lot -> min_lot returned."""
    # $1 risk / (100 pip * $10) = 0.001 lots < min_lot=0.01 -> return 0.01
    size = compute_position_size(1.0, 100.0, _eurusd())
    assert size == pytest.approx(_eurusd().min_lot)


# ─────────────────────────────────────────────────────────────────────────────
# compute_position_size — US100
# ─────────────────────────────────────────────────────────────────────────────


def test_position_size_us100_10point_stop():
    """$100 risk / (10 points * $1/point/lot) = 10.0 lots."""
    size = compute_position_size(
        risk_amount=100.0,
        stop_distance_points=10.0,
        instrument=_us100(),
    )
    assert size == pytest.approx(10.0)


def test_position_size_us100_respects_lot_step():
    size = compute_position_size(100.0, 7.0, _us100())
    instrument = _us100()
    steps = round(size / instrument.lot_step)
    assert abs(size - steps * instrument.lot_step) < 1e-9


def test_position_size_us100_floors_to_lot_step():
    """$100 / (7 * $1) = 14.28... lots -> floored to 14.2 (lot_step=0.1)."""
    size = compute_position_size(100.0, 7.0, _us100())
    assert size == pytest.approx(14.2)


# ─────────────────────────────────────────────────────────────────────────────
# _lot_step_precision
# ─────────────────────────────────────────────────────────────────────────────


def test_lot_step_precision_001():
    assert _lot_step_precision(0.01) == 2


def test_lot_step_precision_01():
    assert _lot_step_precision(0.1) == 1


def test_lot_step_precision_1():
    assert _lot_step_precision(1.0) == 0


def test_lot_step_precision_001_3_decimals():
    assert _lot_step_precision(0.001) == 3
