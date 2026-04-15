"""
tests/unit/test_slippage_execution.py
--------------------------------------
Verify that slippage is applied correctly in PaperExecutionEngine.fill_order().
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aion.core.enums import AssetClass, DataSource, SessionName, Timeframe, TradeDirection
from aion.core.models import InstrumentSpec, MarketBar
from aion.execution.models import ExecutionOrder
from aion.execution.paper import PaperExecutionEngine

_UTC = timezone.utc
_TS = datetime(2025, 1, 15, 14, 35, 0, tzinfo=_UTC)


def _bar(open_: float = 21100.0) -> MarketBar:
    return MarketBar(
        symbol="US100.cash",
        timestamp_utc=_TS,
        timestamp_market=_TS,
        timeframe=Timeframe.M1,
        open=open_,
        high=open_ + 10,
        low=open_ - 10,
        close=open_ + 5,
        tick_volume=100,
        real_volume=0,
        spread=2,
        source=DataSource.CSV,
    )


def _order(direction: TradeDirection) -> ExecutionOrder:
    if direction == TradeDirection.LONG:
        return ExecutionOrder(
            setup_id="s1",
            strategy_id="test",
            symbol="US100.cash",
            direction=TradeDirection.LONG,
            entry_price=21100.0,
            stop_price=21090.0,
            target_price=21120.0,
            position_size=1.0,
            risk_amount=100.0,
            stop_distance_points=10.0,
            target_distance_points=20.0,
            created_at=_TS,
        )
    return ExecutionOrder(
        setup_id="s2",
        strategy_id="test",
        symbol="US100.cash",
        direction=TradeDirection.SHORT,
        entry_price=21100.0,
        stop_price=21110.0,
        target_price=21080.0,
        position_size=1.0,
        risk_amount=100.0,
        stop_distance_points=10.0,
        target_distance_points=20.0,
        created_at=_TS,
    )


class TestSlippageExecution:

    def test_long_fill_adds_slippage(self):
        engine = PaperExecutionEngine()
        fill, pos = engine.fill_order(_order(TradeDirection.LONG), _bar(21100.0), slippage_points=4.0)
        # bar.open=21100, LONG slippage = +4
        assert fill.fill_price == pytest.approx(21104.0)
        assert fill.slippage_points == 4.0

    def test_short_fill_subtracts_slippage(self):
        engine = PaperExecutionEngine()
        fill, pos = engine.fill_order(_order(TradeDirection.SHORT), _bar(21100.0), slippage_points=4.0)
        # bar.open=21100, SHORT slippage = -4
        assert fill.fill_price == pytest.approx(21096.0)
        assert fill.slippage_points == 4.0

    def test_zero_slippage_preserves_bar_open(self):
        engine = PaperExecutionEngine()
        fill, pos = engine.fill_order(_order(TradeDirection.LONG), _bar(21100.0), slippage_points=0.0)
        assert fill.fill_price == pytest.approx(21100.0)
        assert fill.slippage_points == 0.0

    def test_default_slippage_is_zero(self):
        engine = PaperExecutionEngine()
        fill, pos = engine.fill_order(_order(TradeDirection.LONG), _bar(21100.0))
        assert fill.fill_price == pytest.approx(21100.0)
        assert fill.slippage_points == 0.0

    def test_slippage_affects_position_fill_price(self):
        engine = PaperExecutionEngine()
        fill, pos = engine.fill_order(_order(TradeDirection.LONG), _bar(21100.0), slippage_points=3.0)
        assert pos.fill.fill_price == pytest.approx(21103.0)

    def test_slippage_propagates_to_r_multiple(self):
        """With slippage, stop distance shrinks, so -1R loss is reached sooner."""
        engine = PaperExecutionEngine()
        order = _order(TradeDirection.LONG)
        # order: entry=21100, stop=21090 (10pt distance)
        # With 4pt slippage: fill=21104, stop still at 21090 (14pt distance from fill!)
        # But the ExecutionOrder.stop_price is fixed. The R-multiple uses:
        #   r = (close - fill) / |stop - fill|
        fill, pos = engine.fill_order(order, _bar(21100.0), slippage_points=4.0)
        assert fill.fill_price == pytest.approx(21104.0)

        # Evaluate at stop: close at stop=21090
        stop_bar = MarketBar(
            symbol="US100.cash",
            timestamp_utc=datetime(2025, 1, 15, 14, 36, tzinfo=_UTC),
            timestamp_market=datetime(2025, 1, 15, 14, 36, tzinfo=_UTC),
            timeframe=Timeframe.M1,
            open=21095, high=21095, low=21089, close=21090,
            tick_volume=100, real_volume=0, spread=2,
            source=DataSource.CSV,
        )
        closed = engine.evaluate_bar(pos, stop_bar, bar_index=1)
        assert closed is not None
        # r = (21090 - 21104) / |21090 - 21104| = -14/14 = -1.0
        assert closed.r_multiple == pytest.approx(-1.0)
        # pnl = -1.0 * 100 = -100
        assert closed.pnl_amount == pytest.approx(-100.0)
