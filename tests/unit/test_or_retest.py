"""
tests/unit/test_or_retest.py
──────────────────────────────
Unit tests for OpeningRangeRetestEngine.

Tests the full state machine: OR → break → retest → candidate,
including fake out handling, session reset, and price calculations.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import pytest

from aion.core.constants import FEATURE_SET_VERSION, SNAPSHOT_VERSION
from aion.core.enums import AssetClass, DataSource, SessionName, Timeframe, TradeDirection
from aion.core.ids import new_snapshot_id
from aion.core.models import (
    DataQualityReport,
    FeatureVector,
    InstrumentSpec,
    MarketBar,
    MarketSnapshot,
    SessionContext,
)
from aion.strategies.models import StrategyOutcome
from aion.strategies.or_range import OpeningRangeConfig, ORMethod
from aion.strategies.or_retest import (
    OpeningRangeRetestEngine,
    RetestDefinition,
    RetestPhase,
)

_UTC = timezone.utc

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — US100.cash at NY open (13:30 UTC)
# ─────────────────────────────────────────────────────────────────────────────

_OR_TIME = time(13, 30)
_SESSION_OPEN = datetime(2025, 1, 15, 13, 0, 0, tzinfo=_UTC)


def _instrument() -> InstrumentSpec:
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
        min_lot=0.01,
        lot_step=0.01,
        currency_profit="USD",
        currency_margin="USD",
        session_calendar="us_equity",
        trading_hours_label="Mon-Fri",
    )


def _bar(
    minute_offset: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    tf: Timeframe = Timeframe.M5,
) -> MarketBar:
    """Create a bar at 13:30 + minute_offset UTC."""
    ts = datetime(2025, 1, 15, 13, 30, 0, tzinfo=_UTC) + timedelta(minutes=minute_offset)
    return MarketBar(
        symbol="US100.cash",
        timestamp_utc=ts,
        timestamp_market=ts,
        timeframe=tf,
        open=open_,
        high=high,
        low=low,
        close=close,
        tick_volume=100.0,
        real_volume=0.0,
        spread=2.0,
        source=DataSource.CSV,
    )


def _quality(score: float = 1.0) -> DataQualityReport:
    return DataQualityReport(
        symbol="US100.cash",
        timeframe=Timeframe.M1,
        rows_checked=100,
        missing_bars=0,
        duplicate_timestamps=0,
        out_of_order_rows=0,
        stale_bars=0,
        spike_bars=0,
        null_rows=0,
        quality_score=score,
        warnings=[],
    )


def _fv(ts: datetime) -> FeatureVector:
    return FeatureVector(
        symbol="US100.cash",
        timestamp_utc=ts,
        timeframe=Timeframe.M1,
        atr_14=5.0,
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


def _session(
    ts: datetime,
    name: SessionName = SessionName.NEW_YORK,
    open_utc: datetime | None = _SESSION_OPEN,
) -> SessionContext:
    return SessionContext(
        trading_day=ts.date(),
        broker_time=ts,
        market_time=ts,
        local_time=ts,
        is_asia=False,
        is_london=name in (SessionName.LONDON, SessionName.OVERLAP_LONDON_NY),
        is_new_york=name in (SessionName.NEW_YORK, SessionName.OVERLAP_LONDON_NY),
        is_session_open_window=name != SessionName.OFF_HOURS,
        opening_range_active=False,
        opening_range_completed=True,
        session_name=name,
        session_open_utc=open_utc,
        session_close_utc=ts.replace(hour=21, minute=0) if open_utc else None,
    )


def _snapshot(
    minute_offset: int,
    bars_m5: list[MarketBar] | None = None,
    bars_m1: list[MarketBar] | None = None,
    latest: MarketBar | None = None,
    session_name: SessionName = SessionName.NEW_YORK,
    session_open: datetime | None = _SESSION_OPEN,
    quality_score: float = 1.0,
) -> MarketSnapshot:
    """Build a snapshot for testing. The latest_bar defaults to bars_m5[-1]."""
    ts = datetime(2025, 1, 15, 13, 30, 0, tzinfo=_UTC) + timedelta(minutes=minute_offset)

    if bars_m5 is None:
        bars_m5 = []
    if bars_m1 is None:
        bars_m1 = []

    if latest is None:
        latest = (bars_m5[-1] if bars_m5 else
                  bars_m1[-1] if bars_m1 else
                  _bar(minute_offset, 21100, 21100, 21100, 21100))

    return MarketSnapshot(
        snapshot_id=new_snapshot_id(),
        symbol="US100.cash",
        timestamp_utc=ts,
        base_timeframe=Timeframe.M1,
        instrument=_instrument(),
        session_context=_session(ts, session_name, session_open),
        latest_bar=latest,
        bars_m1=bars_m1,
        bars_m5=bars_m5,
        bars_m15=[],
        feature_vector=_fv(ts),
        quality_report=_quality(quality_score),
        snapshot_version=SNAPSHOT_VERSION,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Engine + config factory
# ─────────────────────────────────────────────────────────────────────────────

# OR = single M5 candle at 13:30 UTC, range 5-200 pts
_OR_CONFIG = OpeningRangeConfig(
    method=ORMethod.SINGLE_CANDLE,
    reference_time=_OR_TIME,
    candle_timeframe=Timeframe.M5,
    min_range_points=5.0,
    max_range_points=200.0,
)


def _defn(**overrides) -> RetestDefinition:
    defaults = dict(
        strategy_id="or_retest_ny_test",
        session_name="NEW_YORK",
        or_config=_OR_CONFIG,
        rr_ratio=2.0,
        allow_fake_out_reversal=True,
    )
    defaults.update(overrides)
    return RetestDefinition(**defaults)


def _engine(**overrides) -> OpeningRangeRetestEngine:
    return OpeningRangeRetestEngine(_defn(**overrides), min_quality_score=0.0)


# Fixed OR bar: H=21120, L=21100 → midpoint=21110, range=20
_OR_BAR = _bar(0, open_=21105, high=21120, low=21100, close=21115)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — OR detection
# ─────────────────────────────────────────────────────────────────────────────


class TestORDetection:

    def test_or_not_ready_without_matching_bar(self):
        engine = _engine()
        snap = _snapshot(5, bars_m5=[_bar(5, 21100, 21110, 21090, 21105)])
        result = engine.evaluate(snap)
        assert result.outcome == StrategyOutcome.NO_TRADE
        assert result.no_trade.reason_code == "OR_NOT_READY"

    def test_or_detected_transitions_to_waiting_break(self):
        engine = _engine()
        # Bar at 13:30 (offset=0) matches OR_TIME
        snap = _snapshot(0, bars_m5=[_OR_BAR], latest=_OR_BAR)
        result = engine.evaluate(snap)
        # OR detected, but the OR bar itself is not a break (close=21115 inside range)
        assert result.outcome == StrategyOutcome.NO_TRADE
        assert engine.phase == RetestPhase.WAITING_BREAK
        assert engine.or_level is not None
        assert engine.or_level.or_high == 21120
        assert engine.or_level.or_low == 21100


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Break detection
# ─────────────────────────────────────────────────────────────────────────────


class TestBreakDetection:

    def _setup_break(self, engine, or_bar=None):
        """Feed the OR bar to move to WAITING_BREAK."""
        if or_bar is None:
            or_bar = _OR_BAR
        snap = _snapshot(0, bars_m5=[or_bar], latest=or_bar)
        engine.evaluate(snap)

    def test_bull_break_detected(self):
        engine = _engine()
        self._setup_break(engine)
        # Bull break: open inside (21118 <= 21120), close outside (21125 > 21120)
        break_bar = _bar(5, open_=21118, high=21126, low=21115, close=21125)
        snap = _snapshot(5, bars_m5=[_OR_BAR, break_bar], latest=break_bar)
        result = engine.evaluate(snap)
        assert result.no_trade.reason_code == "BREAK_DETECTED"
        assert engine.phase == RetestPhase.WAITING_RETEST

    def test_bear_break_detected(self):
        engine = _engine()
        self._setup_break(engine)
        # Bear break: open inside (21102 >= 21100), close outside (21095 < 21100)
        break_bar = _bar(5, open_=21102, high=21105, low=21093, close=21095)
        snap = _snapshot(5, bars_m5=[_OR_BAR, break_bar], latest=break_bar)
        result = engine.evaluate(snap)
        assert result.no_trade.reason_code == "BREAK_DETECTED"
        assert engine.phase == RetestPhase.WAITING_RETEST

    def test_no_break_stays_in_phase(self):
        engine = _engine()
        self._setup_break(engine)
        # Bar stays inside: open=21108, close=21115 — both inside range
        inside_bar = _bar(5, open_=21108, high=21118, low=21105, close=21115)
        snap = _snapshot(5, bars_m5=[_OR_BAR, inside_bar], latest=inside_bar)
        result = engine.evaluate(snap)
        assert result.no_trade.reason_code == "NO_BREAK"
        assert engine.phase == RetestPhase.WAITING_BREAK

    def test_direction_bias_blocks_wrong_direction(self):
        engine = _engine(direction_bias=TradeDirection.SHORT)
        self._setup_break(engine)
        # Bull break — but bias is SHORT
        break_bar = _bar(5, open_=21118, high=21126, low=21115, close=21125)
        snap = _snapshot(5, bars_m5=[_OR_BAR, break_bar], latest=break_bar)
        result = engine.evaluate(snap)
        assert result.no_trade.reason_code == "DIRECTION_BIAS"
        assert engine.phase == RetestPhase.WAITING_BREAK  # stays, not promoted


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Retest confirmation
# ─────────────────────────────────────────────────────────────────────────────


def _setup_to_retest(engine, direction: str = "bull"):
    """Feed OR bar + break bar to reach WAITING_RETEST."""
    or_bar = _OR_BAR
    snap_or = _snapshot(0, bars_m5=[or_bar], latest=or_bar)
    engine.evaluate(snap_or)

    if direction == "bull":
        break_bar = _bar(5, open_=21118, high=21126, low=21115, close=21125)
    else:
        break_bar = _bar(5, open_=21102, high=21105, low=21093, close=21095)
    snap_brk = _snapshot(5, bars_m5=[or_bar, break_bar], latest=break_bar)
    engine.evaluate(snap_brk)
    assert engine.phase == RetestPhase.WAITING_RETEST


class TestRetestConfirmation:

    def test_bull_retest_confirmed(self):
        engine = _engine()
        _setup_to_retest(engine, "bull")
        # Retest: low touches OR high (21119 <= 21120) AND close > OR high (21123 > 21120)
        retest = _bar(10, open_=21122, high=21125, low=21119, close=21123)
        snap = _snapshot(10, bars_m5=[_OR_BAR, retest], latest=retest)
        result = engine.evaluate(snap)
        assert result.outcome == StrategyOutcome.CANDIDATE
        assert result.candidate.direction == TradeDirection.LONG

    def test_bear_retest_confirmed(self):
        engine = _engine()
        _setup_to_retest(engine, "bear")
        # Retest: high touches OR low (21101 >= 21100) AND close < OR low (21097 < 21100)
        retest = _bar(10, open_=21098, high=21101, low=21094, close=21097)
        snap = _snapshot(10, bars_m5=[_OR_BAR, retest], latest=retest)
        result = engine.evaluate(snap)
        assert result.outcome == StrategyOutcome.CANDIDATE
        assert result.candidate.direction == TradeDirection.SHORT

    def test_no_touch_stays_waiting(self):
        engine = _engine()
        _setup_to_retest(engine, "bull")
        # Bar doesn't touch OR high (low=21122 > 21120)
        no_touch = _bar(10, open_=21124, high=21128, low=21122, close=21126)
        snap = _snapshot(10, bars_m5=[_OR_BAR, no_touch], latest=no_touch)
        result = engine.evaluate(snap)
        assert result.outcome == StrategyOutcome.NO_TRADE
        assert result.no_trade.reason_code == "WAITING_RETEST"
        assert engine.phase == RetestPhase.WAITING_RETEST


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Fake out
# ─────────────────────────────────────────────────────────────────────────────


class TestFakeOut:

    def test_fake_out_resets_to_waiting_break(self):
        """Touches OR high but closes inside → reset."""
        engine = _engine(allow_fake_out_reversal=False)
        _setup_to_retest(engine, "bull")
        # Touches (low=21119 <= 21120) but closes inside (21115 <= 21120)
        fake = _bar(10, open_=21122, high=21124, low=21119, close=21115)
        snap = _snapshot(10, bars_m5=[_OR_BAR, fake], latest=fake)
        result = engine.evaluate(snap)
        assert result.no_trade.reason_code == "FAKE_OUT_RESET"
        assert engine.phase == RetestPhase.WAITING_BREAK

    def test_fake_out_reversal_switches_direction(self):
        """Touches OR high but closes inside AND the same bar is a bear break."""
        engine = _engine(allow_fake_out_reversal=True)
        _setup_to_retest(engine, "bull")
        # Touches OR high (low=21119 <= 21120) + closes inside (21095 <= 21120)
        # AND is a bear break: open >= OR_low (21102 >= 21100) + close < OR_low (21095 < 21100)
        reversal = _bar(10, open_=21102, high=21121, low=21093, close=21095)
        snap = _snapshot(10, bars_m5=[_OR_BAR, reversal], latest=reversal)
        result = engine.evaluate(snap)
        assert result.no_trade.reason_code == "FAKE_OUT_REVERSAL"
        assert engine.phase == RetestPhase.WAITING_RETEST
        # Now waiting for BEAR retest
        # Feed a bear retest bar
        bear_retest = _bar(15, open_=21098, high=21101, low=21094, close=21097)
        snap2 = _snapshot(15, bars_m5=[_OR_BAR, bear_retest], latest=bear_retest)
        result2 = engine.evaluate(snap2)
        assert result2.outcome == StrategyOutcome.CANDIDATE
        assert result2.candidate.direction == TradeDirection.SHORT

    def test_fake_out_no_reversal_when_disabled(self):
        """Even if bar is inverse break, reversal is off → plain reset."""
        engine = _engine(allow_fake_out_reversal=False)
        _setup_to_retest(engine, "bull")
        reversal = _bar(10, open_=21102, high=21121, low=21093, close=21095)
        snap = _snapshot(10, bars_m5=[_OR_BAR, reversal], latest=reversal)
        result = engine.evaluate(snap)
        assert result.no_trade.reason_code == "FAKE_OUT_RESET"
        assert engine.phase == RetestPhase.WAITING_BREAK


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Single trade per session
# ─────────────────────────────────────────────────────────────────────────────


class TestSingleTradePerSession:

    def test_no_second_signal_after_candidate(self):
        engine = _engine()
        _setup_to_retest(engine, "bull")
        retest = _bar(10, open_=21122, high=21125, low=21119, close=21123)
        snap = _snapshot(10, bars_m5=[_OR_BAR, retest], latest=retest)
        r1 = engine.evaluate(snap)
        assert r1.outcome == StrategyOutcome.CANDIDATE

        # Feed another bar — should be SESSION_DONE
        another = _bar(15, open_=21122, high=21130, low=21118, close=21128)
        snap2 = _snapshot(15, bars_m5=[_OR_BAR, another], latest=another)
        r2 = engine.evaluate(snap2)
        assert r2.outcome == StrategyOutcome.NO_TRADE
        assert r2.no_trade.reason_code == "SESSION_DONE"


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Session reset
# ─────────────────────────────────────────────────────────────────────────────


class TestSessionReset:

    def test_new_session_resets_state(self):
        engine = _engine()
        _setup_to_retest(engine, "bull")
        assert engine.phase == RetestPhase.WAITING_RETEST

        # New session (different session_open_utc)
        new_open = datetime(2025, 1, 16, 13, 0, 0, tzinfo=_UTC)
        new_or = _bar(0, open_=21200, high=21220, low=21195, close=21210)
        # Override the bar timestamp to be on the new day
        ts_new = datetime(2025, 1, 16, 13, 30, 0, tzinfo=_UTC)
        new_or_bar = MarketBar(
            symbol="US100.cash",
            timestamp_utc=ts_new,
            timestamp_market=ts_new,
            timeframe=Timeframe.M5,
            open=21200, high=21220, low=21195, close=21210,
            tick_volume=100, real_volume=0, spread=2,
            source=DataSource.CSV,
        )
        snap = MarketSnapshot(
            snapshot_id=new_snapshot_id(),
            symbol="US100.cash",
            timestamp_utc=ts_new,
            base_timeframe=Timeframe.M1,
            instrument=_instrument(),
            session_context=_session(ts_new, SessionName.NEW_YORK, new_open),
            latest_bar=new_or_bar,
            bars_m1=[],
            bars_m5=[new_or_bar],
            bars_m15=[],
            feature_vector=_fv(ts_new),
            quality_report=_quality(),
            snapshot_version=SNAPSHOT_VERSION,
        )

        result = engine.evaluate(snap)
        # Should have reset and tried to compute OR
        assert engine.phase in (RetestPhase.WAITING_OR, RetestPhase.WAITING_BREAK)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Price calculations (SL / TP)
# ─────────────────────────────────────────────────────────────────────────────


class TestPriceCalculations:
    """
    OR: H=21120, L=21100, midpoint=21110, half_range=10
    """

    def test_long_sl_is_midpoint(self):
        engine = _engine()
        _setup_to_retest(engine, "bull")
        retest = _bar(10, open_=21122, high=21125, low=21119, close=21123)
        snap = _snapshot(10, bars_m5=[_OR_BAR, retest], latest=retest)
        result = engine.evaluate(snap)
        detail = result.candidate.strategy_detail

        assert detail["stop_price"] == 21110.0  # midpoint

    def test_long_entry_is_or_high(self):
        engine = _engine()
        _setup_to_retest(engine, "bull")
        retest = _bar(10, open_=21122, high=21125, low=21119, close=21123)
        snap = _snapshot(10, bars_m5=[_OR_BAR, retest], latest=retest)
        result = engine.evaluate(snap)

        assert result.candidate.entry_reference == 21120.0  # or_high

    def test_long_tp_is_2r(self):
        engine = _engine(rr_ratio=2.0)
        _setup_to_retest(engine, "bull")
        retest = _bar(10, open_=21122, high=21125, low=21119, close=21123)
        snap = _snapshot(10, bars_m5=[_OR_BAR, retest], latest=retest)
        result = engine.evaluate(snap)
        detail = result.candidate.strategy_detail

        # entry=21120, sl=21110, risk=10, tp=21120+10*2=21140
        assert detail["target_price"] == pytest.approx(21140.0)

    def test_short_sl_is_midpoint(self):
        engine = _engine()
        _setup_to_retest(engine, "bear")
        retest = _bar(10, open_=21098, high=21101, low=21094, close=21097)
        snap = _snapshot(10, bars_m5=[_OR_BAR, retest], latest=retest)
        result = engine.evaluate(snap)
        detail = result.candidate.strategy_detail

        assert detail["stop_price"] == 21110.0  # midpoint

    def test_short_entry_is_or_low(self):
        engine = _engine()
        _setup_to_retest(engine, "bear")
        retest = _bar(10, open_=21098, high=21101, low=21094, close=21097)
        snap = _snapshot(10, bars_m5=[_OR_BAR, retest], latest=retest)
        result = engine.evaluate(snap)

        assert result.candidate.entry_reference == 21100.0  # or_low

    def test_short_tp_is_2r(self):
        engine = _engine(rr_ratio=2.0)
        _setup_to_retest(engine, "bear")
        retest = _bar(10, open_=21098, high=21101, low=21094, close=21097)
        snap = _snapshot(10, bars_m5=[_OR_BAR, retest], latest=retest)
        result = engine.evaluate(snap)
        detail = result.candidate.strategy_detail

        # entry=21100, sl=21110, risk=10, tp=21100-10*2=21080
        assert detail["target_price"] == pytest.approx(21080.0)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — strategy_detail audit
# ─────────────────────────────────────────────────────────────────────────────


class TestStrategyDetail:

    def test_detail_contains_all_audit_fields(self):
        engine = _engine()
        _setup_to_retest(engine, "bull")
        retest = _bar(10, open_=21122, high=21125, low=21119, close=21123)
        snap = _snapshot(10, bars_m5=[_OR_BAR, retest], latest=retest)
        result = engine.evaluate(snap)
        d = result.candidate.strategy_detail

        assert d["engine"] == "or_retest"
        assert d["break_direction"] == "LONG"
        assert d["break_bar_timestamp"] is not None
        assert d["retest_bar_timestamp"] is not None
        assert d["entry_price"] == 21120.0
        assert d["stop_price"] == 21110.0
        assert d["target_price"] == 21140.0
        assert d["rr_ratio"] == 2.0
        assert d["or_high"] == 21120.0
        assert d["or_low"] == 21100.0
        assert d["midpoint"] == 21110.0
        assert d["or_method"] == "SINGLE_CANDLE"
        assert d["or_source_bars"] == 1

    def test_detail_empty_dict_by_default_in_other_engines(self):
        """CandidateSetup.strategy_detail defaults to {} — backwards compatible."""
        from aion.strategies.models import CandidateSetup

        c = CandidateSetup(
            strategy_id="test",
            strategy_version="1.0",
            symbol="X",
            timestamp_utc=datetime(2025, 1, 1, tzinfo=_UTC),
            direction=TradeDirection.LONG,
            entry_reference=100,
            range_high=110,
            range_low=90,
            range_size_pips=20,
            session_name="TEST",
            quality_score=1.0,
            atr_14=None,
        )
        assert c.strategy_detail == {}


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Session guard
# ─────────────────────────────────────────────────────────────────────────────


class TestSessionGuard:

    def test_wrong_session_returns_no_trade(self):
        engine = _engine(session_name="NEW_YORK")
        snap = _snapshot(0, bars_m5=[_OR_BAR], latest=_OR_BAR,
                         session_name=SessionName.ASIA)
        result = engine.evaluate(snap)
        assert result.no_trade.reason_code == "NOT_IN_TARGET_SESSION"

    def test_overlap_counts_for_ny(self):
        engine = _engine(session_name="NEW_YORK")
        snap = _snapshot(0, bars_m5=[_OR_BAR], latest=_OR_BAR,
                         session_name=SessionName.OVERLAP_LONDON_NY)
        result = engine.evaluate(snap)
        # Should proceed (overlap counts as NY), not reject
        assert result.no_trade.reason_code != "NOT_IN_TARGET_SESSION"


# ─────────────────────────────────────────────────────────────────────────────
# Tests — No break on OR bar
# ─────────────────────────────────────────────────────────────────────────────


class TestNoBreakOnORBar:

    def test_or_bar_itself_cannot_be_break(self):
        """Even if the OR bar opens inside and closes outside, it must not
        be treated as a break candle.  Break requires a subsequent bar."""
        engine = _engine()
        # OR bar at minute 0: open=21105 (inside), close=21125 (> or_high=21120)
        # This technically satisfies break conditions, but is the OR bar itself.
        or_bar = _bar(0, open_=21105, high=21126, low=21098, close=21125)
        snap = _snapshot(0, bars_m5=[or_bar], latest=or_bar)
        result = engine.evaluate(snap)
        # OR is set, but break is NOT detected on the same bar
        assert engine.phase == RetestPhase.WAITING_BREAK
        assert result.no_trade.reason_code == "OR_SET"

    def test_break_allowed_on_next_bar(self):
        """After OR is set, a break on the very next bar IS allowed."""
        engine = _engine()
        or_bar = _OR_BAR  # at minute 0
        snap_or = _snapshot(0, bars_m5=[or_bar], latest=or_bar)
        engine.evaluate(snap_or)
        assert engine.phase == RetestPhase.WAITING_BREAK

        # Next bar at minute 5 — valid break
        break_bar = _bar(5, open_=21118, high=21126, low=21115, close=21125)
        snap_brk = _snapshot(5, bars_m5=[or_bar, break_bar], latest=break_bar)
        result = engine.evaluate(snap_brk)
        assert result.no_trade.reason_code == "BREAK_DETECTED"
        assert engine.phase == RetestPhase.WAITING_RETEST

    def test_or_bar_skip_reason_code(self):
        """If we're in WAITING_BREAK but the latest bar is the OR bar,
        reason_code should be OR_BAR_SKIP."""
        engine = _engine()
        or_bar = _OR_BAR
        snap = _snapshot(0, bars_m5=[or_bar], latest=or_bar)
        engine.evaluate(snap)

        # Feed the same bar again (same timestamp)
        snap2 = _snapshot(0, bars_m5=[or_bar], latest=or_bar)
        result = engine.evaluate(snap2)
        assert result.no_trade.reason_code == "OR_BAR_SKIP"
