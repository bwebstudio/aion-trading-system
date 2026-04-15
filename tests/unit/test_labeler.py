"""
tests/unit/test_labeler.py
────────────────────────────
Unit tests for aion.replay.labeler.label_candidate.

Tests verify:
  - LONG: entry activated → WIN when target reached before stop
  - LONG: entry activated → LOSS when stop reached before target
  - LONG: entry not activated → ENTRY_NOT_ACTIVATED
  - LONG: entry activated, neither stop nor target → TIMEOUT
  - SHORT: WIN, LOSS, ENTRY_NOT_ACTIVATED
  - Same-bar stop + target → LOSS (conservative tie-break)
  - bars_to_entry counting (0-based index of activation bar)
  - bars_to_resolution counting (offset from activation to outcome bar)
  - MFE = max favorable excursion in pips from entry after activation
  - MAE = max adverse excursion in pips from entry after activation
  - mfe_pips / mae_pips / pnl_pips are None when not activated
  - pnl_pips == +target_pips on WIN
  - pnl_pips == -stop_pips on LOSS
  - pnl_pips is None on TIMEOUT
  - Empty future_bars → ENTRY_NOT_ACTIVATED
  - max_bars limits look-ahead
  - Activation on first bar vs. second bar
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from aion.core.enums import DataSource, Timeframe, TradeDirection
from aion.core.models import MarketBar
from aion.replay.labeler import label_candidate
from aion.replay.models import LabelConfig, LabelOutcome
from aion.strategies.models import CandidateSetup

_UTC = timezone.utc
_TS = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_UTC)

# pip_size = 0.00001 * 10 = 0.0001 for EURUSD default config


# ─────────────────────────────────────────────────────────────────────────────
# Factories
# ─────────────────────────────────────────────────────────────────────────────


def make_bar(high: float, low: float, offset_minutes: int = 1) -> MarketBar:
    mid = round((high + low) / 2, 5)
    ts = _TS + timedelta(minutes=offset_minutes)
    return MarketBar(
        symbol="EURUSD",
        timestamp_utc=ts,
        timestamp_market=ts,
        timeframe=Timeframe.M1,
        open=mid,
        high=high,
        low=low,
        close=mid,
        tick_volume=100.0,
        real_volume=0.0,
        spread=2.0,
        source=DataSource.SYNTHETIC,
    )


def make_candidate(
    direction: TradeDirection = TradeDirection.LONG,
    entry_reference: float = 1.1020,
    range_high: float = 1.1020,
    range_low: float = 1.1000,
) -> CandidateSetup:
    return CandidateSetup(
        strategy_id="or_london_v1",
        strategy_version="1.0.0",
        symbol="EURUSD",
        timestamp_utc=_TS,
        direction=direction,
        entry_reference=entry_reference,
        range_high=range_high,
        range_low=range_low,
        range_size_pips=20.0,
        session_name="LONDON",
        quality_score=1.0,
        atr_14=0.00015,
    )


def make_config(
    stop_pips: float = 10.0,
    target_pips: float = 20.0,
    max_bars: int = 20,
) -> LabelConfig:
    return LabelConfig(
        stop_pips=stop_pips,
        target_pips=target_pips,
        max_bars=max_bars,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LONG — WIN
# ─────────────────────────────────────────────────────────────────────────────
# entry=1.1020, stop=1.1010, target=1.1040


def test_long_win_target_reached_before_stop():
    bars = [
        make_bar(high=1.1025, low=1.1015, offset_minutes=1),  # activates entry
        make_bar(high=1.1032, low=1.1022, offset_minutes=2),  # no touch
        make_bar(high=1.1045, low=1.1035, offset_minutes=3),  # target hit
    ]
    result = label_candidate(make_candidate(), bars, make_config())
    assert result.outcome == LabelOutcome.WIN
    assert result.entry_activated is True
    assert result.pnl_pips == pytest.approx(20.0)


def test_long_win_bars_to_entry_is_zero_first_bar():
    bars = [
        make_bar(high=1.1025, low=1.1015, offset_minutes=1),  # activates on bar 0
        make_bar(high=1.1045, low=1.1035, offset_minutes=2),  # target
    ]
    result = label_candidate(make_candidate(), bars, make_config())
    assert result.bars_to_entry == 0


def test_long_win_bars_to_entry_is_one_second_bar():
    bars = [
        make_bar(high=1.1015, low=1.1008, offset_minutes=1),  # not activated
        make_bar(high=1.1025, low=1.1018, offset_minutes=2),  # activates on bar 1
        make_bar(high=1.1045, low=1.1035, offset_minutes=3),  # target
    ]
    result = label_candidate(make_candidate(), bars, make_config())
    assert result.bars_to_entry == 1


def test_long_win_bars_to_resolution():
    """bars_to_resolution: offset from activation bar to target bar."""
    bars = [
        make_bar(high=1.1025, low=1.1015, offset_minutes=1),  # activation (offset 0)
        make_bar(high=1.1032, low=1.1022, offset_minutes=2),  # offset 1
        make_bar(high=1.1045, low=1.1035, offset_minutes=3),  # offset 2 — target
    ]
    result = label_candidate(make_candidate(), bars, make_config())
    assert result.bars_to_resolution == 2


def test_long_win_resolved_on_activation_bar():
    """Target hit on the same bar as entry activation → bars_to_resolution == 0."""
    bars = [
        make_bar(high=1.1045, low=1.1020, offset_minutes=1),  # activates AND hits target
    ]
    result = label_candidate(make_candidate(), bars, make_config())
    assert result.outcome == LabelOutcome.WIN
    assert result.bars_to_resolution == 0


# ─────────────────────────────────────────────────────────────────────────────
# LONG — LOSS
# ─────────────────────────────────────────────────────────────────────────────


def test_long_loss_stop_reached_before_target():
    bars = [
        make_bar(high=1.1025, low=1.1015, offset_minutes=1),  # activates
        make_bar(high=1.1022, low=1.1008, offset_minutes=2),  # stop hit (low <= 1.1010)
    ]
    result = label_candidate(make_candidate(), bars, make_config())
    assert result.outcome == LabelOutcome.LOSS
    assert result.pnl_pips == pytest.approx(-10.0)


def test_long_loss_bars_to_resolution():
    bars = [
        make_bar(high=1.1025, low=1.1015, offset_minutes=1),  # activation (offset 0)
        make_bar(high=1.1022, low=1.1008, offset_minutes=2),  # offset 1 — stop
    ]
    result = label_candidate(make_candidate(), bars, make_config())
    assert result.bars_to_resolution == 1


# ─────────────────────────────────────────────────────────────────────────────
# LONG — TIMEOUT
# ─────────────────────────────────────────────────────────────────────────────


def test_long_timeout_neither_stop_nor_target():
    """Entry activated, but stop and target never reached in max_bars."""
    bars = [
        make_bar(high=1.1025, low=1.1015, offset_minutes=i)
        for i in range(1, 6)
    ]
    # stop=1.1010 (all lows >= 1.1015), target=1.1040 (all highs <= 1.1025)
    result = label_candidate(make_candidate(), bars, make_config(max_bars=5))
    assert result.outcome == LabelOutcome.TIMEOUT
    assert result.bars_to_resolution is None
    assert result.pnl_pips is None


# ─────────────────────────────────────────────────────────────────────────────
# LONG — ENTRY_NOT_ACTIVATED
# ─────────────────────────────────────────────────────────────────────────────


def test_long_entry_not_activated_price_stays_below():
    """All bars have high < entry_reference → no activation."""
    bars = [
        make_bar(high=1.1018, low=1.1010, offset_minutes=i)
        for i in range(1, 4)
    ]
    result = label_candidate(make_candidate(), bars, make_config())
    assert result.outcome == LabelOutcome.ENTRY_NOT_ACTIVATED
    assert result.entry_activated is False
    assert result.bars_to_entry is None
    assert result.mfe_pips is None
    assert result.mae_pips is None
    assert result.pnl_pips is None


def test_empty_future_bars_returns_not_activated():
    result = label_candidate(make_candidate(), [], make_config())
    assert result.outcome == LabelOutcome.ENTRY_NOT_ACTIVATED
    assert result.entry_activated is False


def test_max_bars_limits_look_ahead():
    """Only the first max_bars bars are inspected; bars beyond are ignored."""
    # Entry would activate on bar 5 (index 4), but max_bars=3 stops at index 2
    bars = [
        make_bar(high=1.1015, low=1.1010, offset_minutes=i)  # below entry
        for i in range(1, 6)
    ] + [make_bar(high=1.1025, low=1.1020, offset_minutes=6)]  # would activate if reached

    result = label_candidate(make_candidate(), bars, make_config(max_bars=3))
    assert result.outcome == LabelOutcome.ENTRY_NOT_ACTIVATED


# ─────────────────────────────────────────────────────────────────────────────
# Conservative tie-break: same-bar stop + target
# ─────────────────────────────────────────────────────────────────────────────


def test_same_bar_stop_and_target_returns_loss():
    """Bar simultaneously satisfies stop (low) and target (high) → LOSS."""
    # entry=1.1020, stop=1.1010, target=1.1040
    bars = [
        make_bar(high=1.1020, low=1.1020, offset_minutes=1),  # activation (just touches)
        make_bar(high=1.1045, low=1.1008, offset_minutes=2),  # spans both stop and target
    ]
    result = label_candidate(make_candidate(), bars, make_config())
    assert result.outcome == LabelOutcome.LOSS


# ─────────────────────────────────────────────────────────────────────────────
# MFE and MAE
# ─────────────────────────────────────────────────────────────────────────────


def test_mfe_measures_best_favorable_excursion():
    """MFE = maximum (bar.high - entry) over all post-activation bars."""
    # entry=1.1020, pip_size=0.0001
    bars = [
        make_bar(high=1.1025, low=1.1015, offset_minutes=1),  # favorable: 5 pips
        make_bar(high=1.1035, low=1.1025, offset_minutes=2),  # favorable: 15 pips
        make_bar(high=1.1045, low=1.1035, offset_minutes=3),  # favorable: 25 pips, target hit
    ]
    result = label_candidate(make_candidate(), bars, make_config())
    assert result.outcome == LabelOutcome.WIN
    assert result.mfe_pips == pytest.approx(25.0, abs=0.1)


def test_mae_measures_worst_adverse_excursion():
    """MAE = maximum (entry - bar.low) over all post-activation bars."""
    # entry=1.1020, pip_size=0.0001
    bars = [
        make_bar(high=1.1025, low=1.1015, offset_minutes=1),  # adverse: 5 pips
        make_bar(high=1.1022, low=1.1008, offset_minutes=2),  # adverse: 12 pips, stop hit
    ]
    result = label_candidate(make_candidate(), bars, make_config())
    assert result.outcome == LabelOutcome.LOSS
    # MAE includes the stop-hit bar: 1.1020 - 1.1008 = 0.0012 = 12 pips
    assert result.mae_pips == pytest.approx(12.0, abs=0.1)


def test_mae_is_zero_when_price_only_goes_favorably():
    """If bar.low never drops below entry, MAE should be 0."""
    bars = [
        make_bar(high=1.1025, low=1.1021, offset_minutes=1),  # activated, low > entry
        make_bar(high=1.1045, low=1.1030, offset_minutes=2),  # target hit
    ]
    result = label_candidate(make_candidate(), bars, make_config())
    assert result.outcome == LabelOutcome.WIN
    assert result.mae_pips == pytest.approx(0.0, abs=0.01)


def test_mfe_mae_none_when_not_activated():
    bars = [make_bar(high=1.1010, low=1.1005, offset_minutes=1)]
    result = label_candidate(make_candidate(), bars, make_config())
    assert result.mfe_pips is None
    assert result.mae_pips is None


# ─────────────────────────────────────────────────────────────────────────────
# SHORT direction
# ─────────────────────────────────────────────────────────────────────────────
# entry=1.1000, stop=1.1010, target=1.0980


def test_short_win():
    """SHORT: entry at 1.1000, target at 1.0980."""
    cand = make_candidate(
        direction=TradeDirection.SHORT,
        entry_reference=1.1000,
        range_high=1.1020,
        range_low=1.1000,
    )
    bars = [
        make_bar(high=1.1005, low=1.0998, offset_minutes=1),  # activates (low <= 1.1000)
        make_bar(high=1.0995, low=1.0975, offset_minutes=2),  # target hit (low <= 1.0980)
    ]
    result = label_candidate(cand, bars, make_config(stop_pips=10, target_pips=20))
    assert result.outcome == LabelOutcome.WIN
    assert result.pnl_pips == pytest.approx(20.0)


def test_short_loss():
    """SHORT: entry at 1.1000, stop at 1.1010."""
    cand = make_candidate(
        direction=TradeDirection.SHORT,
        entry_reference=1.1000,
        range_high=1.1020,
        range_low=1.1000,
    )
    bars = [
        make_bar(high=1.1005, low=1.0998, offset_minutes=1),  # activates
        make_bar(high=1.1012, low=1.1005, offset_minutes=2),  # stop hit (high >= 1.1010)
    ]
    result = label_candidate(cand, bars, make_config(stop_pips=10, target_pips=20))
    assert result.outcome == LabelOutcome.LOSS
    assert result.pnl_pips == pytest.approx(-10.0)


def test_short_entry_not_activated():
    """SHORT: price never drops to entry_reference."""
    cand = make_candidate(
        direction=TradeDirection.SHORT,
        entry_reference=1.1000,
        range_high=1.1020,
        range_low=1.1000,
    )
    bars = [
        make_bar(high=1.1015, low=1.1002, offset_minutes=1),  # low > 1.1000, not activated
        make_bar(high=1.1020, low=1.1005, offset_minutes=2),  # low > 1.1000
    ]
    result = label_candidate(cand, bars, make_config())
    assert result.outcome == LabelOutcome.ENTRY_NOT_ACTIVATED


def test_short_stop_and_target_prices_correct():
    """Verify stop is above entry and target is below for SHORT."""
    cand = make_candidate(
        direction=TradeDirection.SHORT,
        entry_reference=1.1000,
        range_high=1.1020,
        range_low=1.1000,
    )
    cfg = make_config(stop_pips=10, target_pips=20)
    bars = [make_bar(high=1.0995, low=1.0990, offset_minutes=1)]  # not activated
    result = label_candidate(cand, bars, cfg)
    # stop above entry, target below entry
    assert result.stop_price == pytest.approx(1.1000 + 10 * 0.0001)
    assert result.target_price == pytest.approx(1.1000 - 20 * 0.0001)


# ─────────────────────────────────────────────────────────────────────────────
# pnl_pips
# ─────────────────────────────────────────────────────────────────────────────


def test_pnl_is_target_pips_on_win():
    bars = [
        make_bar(high=1.1025, low=1.1015, offset_minutes=1),
        make_bar(high=1.1045, low=1.1035, offset_minutes=2),
    ]
    result = label_candidate(make_candidate(), bars, make_config(target_pips=20))
    assert result.pnl_pips == pytest.approx(20.0)


def test_pnl_is_negative_stop_pips_on_loss():
    bars = [
        make_bar(high=1.1025, low=1.1015, offset_minutes=1),
        make_bar(high=1.1018, low=1.1005, offset_minutes=2),
    ]
    result = label_candidate(make_candidate(), bars, make_config(stop_pips=10))
    assert result.pnl_pips == pytest.approx(-10.0)


def test_pnl_is_none_on_timeout():
    bars = [make_bar(high=1.1025, low=1.1015, offset_minutes=i) for i in range(1, 4)]
    result = label_candidate(make_candidate(), bars, make_config(max_bars=3))
    assert result.outcome == LabelOutcome.TIMEOUT
    assert result.pnl_pips is None


def test_pnl_is_none_on_not_activated():
    bars = [make_bar(high=1.1010, low=1.1005, offset_minutes=1)]
    result = label_candidate(make_candidate(), bars, make_config())
    assert result.pnl_pips is None
