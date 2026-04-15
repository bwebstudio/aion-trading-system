"""
tests/unit/test_execution_model.py
───────────────────────────────────
Unit tests for aion.execution.execution_model.ExecutionModel.

Covers:
  1. slippage increases with larger bar ranges
  2. slippage increases with momentum
  3. spread respects min_spread per symbol
  4. slippage is always > 0 for normal inputs
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from aion.execution.execution_model import (
    DEFAULT_MIN_SPREAD,
    ExecutionModel,
    SymbolParams,
    detect_session,
)


@dataclass
class FakeBar:
    open: float
    high: float
    low: float
    close: float


def _deterministic_model(
    params: dict[str, SymbolParams] | None = None,
    seed: int = 42,
) -> ExecutionModel:
    return ExecutionModel(params_by_symbol=params, rng=random.Random(seed))


# ─── 1. Slippage increases with larger bar ranges ───────────────────────────


def test_slippage_increases_with_bar_range():
    # Same momentum (body / range = 1.0), larger range -> larger slippage.
    small = FakeBar(open=100.0, high=100.5, low=99.5, close=100.5)  # range=1
    large = FakeBar(open=100.0, high=105.0, low=95.0, close=105.0)  # range=10

    # Fresh RNG per call so both share the same multiplier sequence.
    m1 = _deterministic_model(seed=7)
    s_small = m1.estimate_slippage(small, session="LONDON", entry_type="retest")

    m2 = _deterministic_model(seed=7)
    s_large = m2.estimate_slippage(large, session="LONDON", entry_type="retest")

    assert s_large > s_small
    assert s_large == pytest.approx(s_small * 10.0, rel=1e-6)


# ─── 2. Slippage increases with momentum ────────────────────────────────────


def test_slippage_increases_with_momentum():
    # Same range = 1.0.
    # Low momentum: open=close (body=0, momentum=0).
    low = FakeBar(open=100.0, high=100.5, low=99.5, close=100.0)
    # High momentum: close at top of range (body=1.0, momentum=1.0).
    high = FakeBar(open=99.5, high=100.5, low=99.5, close=100.5)

    m1 = _deterministic_model(seed=11)
    s_low = m1.estimate_slippage(low, session="LONDON", entry_type="retest")

    m2 = _deterministic_model(seed=11)
    s_high = m2.estimate_slippage(high, session="LONDON", entry_type="retest")

    assert s_high > s_low


# ─── 3. Spread respects min_spread per symbol ───────────────────────────────


def test_spread_respects_min_spread():
    model = ExecutionModel()

    # ATR so small that 0.02 * atr < min_spread -> min_spread must win.
    for symbol, min_spread in DEFAULT_MIN_SPREAD.items():
        low_atr_spread = model.estimate_spread(symbol, atr_1m=0.001)
        assert low_atr_spread >= min_spread
        assert low_atr_spread == pytest.approx(min_spread)

    # ATR large enough that 0.02 * atr > min_spread.
    us100_spread = model.estimate_spread("US100.cash", atr_1m=500.0)
    assert us100_spread == pytest.approx(0.02 * 500.0)
    assert us100_spread > DEFAULT_MIN_SPREAD["US100.cash"]


def test_spread_unknown_symbol_defaults_to_zero_floor():
    model = ExecutionModel()
    spread = model.estimate_spread("UNKNOWN", atr_1m=10.0)
    # No min_spread registered -> 0.02 * atr = 0.2.
    assert spread == pytest.approx(0.2)


# ─── 4. Slippage is always > 0 for normal inputs ────────────────────────────


def test_slippage_always_positive():
    model = _deterministic_model(seed=1)
    bar = FakeBar(open=100.0, high=100.8, low=99.4, close=100.6)
    for session in ("ASIA", "LONDON", "NY_OPEN", "NY_MID", "NY_CLOSE"):
        for entry in ("limit", "retest", "breakout"):
            slip = model.estimate_slippage(
                bar, session=session, entry_type=entry, symbol="US100.cash"
            )
            assert slip > 0


# ─── Extra: fallbacks do not crash and return 0 ─────────────────────────────


def test_zero_range_bar_returns_zero_slippage():
    model = _deterministic_model()
    bar = FakeBar(open=100.0, high=100.0, low=100.0, close=100.0)
    assert model.estimate_slippage(bar, session="LONDON", entry_type="retest") == 0.0


def test_session_multipliers_ordering():
    # NY_OPEN (1.4) > NY_CLOSE (1.2) > LONDON (1.0) > ASIA (0.7).
    bar = FakeBar(open=100.0, high=101.0, low=99.0, close=101.0)

    def slip(session: str) -> float:
        m = _deterministic_model(seed=99)
        return m.estimate_slippage(bar, session=session, entry_type="retest")

    assert slip("NY_OPEN") > slip("NY_CLOSE") > slip("LONDON") > slip("ASIA")


def test_entry_multiplier_ordering():
    bar = FakeBar(open=100.0, high=101.0, low=99.0, close=101.0)

    def slip(entry: str) -> float:
        m = _deterministic_model(seed=123)
        return m.estimate_slippage(bar, session="LONDON", entry_type=entry)

    assert slip("breakout") > slip("retest") > slip("limit")


# ─── detect_session boundaries ──────────────────────────────────────────────


def test_detect_session_boundaries():
    def ts(h: int, m: int = 0) -> datetime:
        return datetime(2024, 1, 15, h, m, tzinfo=timezone.utc)

    assert detect_session(ts(0)) == "ASIA"
    assert detect_session(ts(6, 59)) == "ASIA"
    assert detect_session(ts(7)) == "LONDON"
    assert detect_session(ts(13, 0)) == "LONDON"
    assert detect_session(ts(13, 30)) == "NY_OPEN"
    assert detect_session(ts(14, 59)) == "NY_OPEN"
    assert detect_session(ts(15)) == "NY_MID"
    assert detect_session(ts(18, 59)) == "NY_MID"
    assert detect_session(ts(19)) == "NY_CLOSE"
    assert detect_session(ts(20, 59)) == "NY_CLOSE"
    assert detect_session(ts(21)) == "OFF_HOURS"


def test_from_config_missing_file_returns_default_model(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    model = ExecutionModel.from_config(missing)
    # Fallback spread for known symbol still resolves to its default min.
    assert model.estimate_spread("US100.cash", atr_1m=0.0) == pytest.approx(1.5)


def test_from_config_reads_yaml(tmp_path):
    cfg = tmp_path / "exec.yaml"
    cfg.write_text(
        "US100.cash:\n"
        "  min_spread: 2.5\n"
        "  k_vol: 0.1\n"
        "  k_mom: 0.2\n",
        encoding="utf-8",
    )
    model = ExecutionModel.from_config(cfg)
    params = model.params_for("US100.cash")
    assert params.min_spread == pytest.approx(2.5)
    assert params.k_vol == pytest.approx(0.1)
    assert params.k_mom == pytest.approx(0.2)
