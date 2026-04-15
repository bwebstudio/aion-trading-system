"""
scripts/run_compare_strategies.py
───────────────────────────────────
Run OpeningRange and VWAP Fade on their respective synthetic datasets
and print a side-by-side metrics comparison.

Usage:
  python scripts/run_compare_strategies.py
  python scripts/run_compare_strategies.py --snapshots 150

Datasets
────────
OR-data   : Breakout scenario.  Price extends above OR after the range closes.
            high=1.1045, low=1.1035. OR LONG entry=1.1020 hits target=1.1040.

VWAP-data : Reversion scenario.  Price extends 15 pips above VWAP, then reverts.
            high=1.1030, low=1.0995. VWAP Fade SHORT entry=1.1025 hits target=1.1005.

Cross-run note
──────────────
The two strategies have opposite expectations:
  - OR breakout expects continuation -> wins on OR-data, loses on VWAP-data.
  - VWAP Fade expects reversion    -> wins on VWAP-data, times out on OR-data.
Both strategies are shown on their designed dataset only; cross-data results
are explained briefly without running them.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aion.analytics.replay_metrics import compute_metrics
from aion.core.constants import FEATURE_SET_VERSION, MIN_QUALITY_SCORE, SNAPSHOT_VERSION
from aion.core.enums import AssetClass, DataSource, SessionName, Timeframe
from aion.core.ids import new_snapshot_id
from aion.core.models import (
    DataQualityReport,
    FeatureVector,
    InstrumentSpec,
    MarketBar,
    MarketSnapshot,
    SessionContext,
)
from aion.replay.models import LabelConfig
from aion.replay.runner import run_replay
from aion.strategies.models import OpeningRangeDefinition
from aion.strategies.opening_range import OpeningRangeEngine
from aion.strategies.vwap_fade import VWAPFadeDefinition, VWAPFadeEngine

_UTC = timezone.utc
_BASE_TS = datetime(2024, 1, 15, 8, 0, 0, tzinfo=_UTC)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _instrument() -> InstrumentSpec:
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


def _session(ts: datetime, or_active: bool) -> SessionContext:
    session_open = ts.replace(hour=8, minute=0, second=0)
    session_close = ts.replace(hour=16, minute=30, second=0)
    return SessionContext(
        trading_day=ts.date(),
        broker_time=ts,
        market_time=ts,
        local_time=ts,
        is_asia=False,
        is_london=True,
        is_new_york=False,
        is_session_open_window=True,
        opening_range_active=or_active,
        opening_range_completed=not or_active,
        session_name=SessionName.LONDON,
        session_open_utc=session_open,
        session_close_utc=session_close,
    )


def _quality() -> DataQualityReport:
    return DataQualityReport(
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


def _fv(ts: datetime, *, vwap: float, or_high: float, or_low: float) -> FeatureVector:
    return FeatureVector(
        symbol="EURUSD",
        timestamp_utc=ts,
        timeframe=Timeframe.M1,
        atr_14=0.00015,
        rolling_range_10=0.0010,
        rolling_range_20=0.0012,
        volatility_percentile_20=0.50,
        session_high=1.1060,
        session_low=1.0990,
        opening_range_high=or_high,
        opening_range_low=or_low,
        vwap_session=vwap,
        spread_mean_20=2.0,
        spread_zscore_20=0.0,
        return_1=0.0001,
        return_5=0.0003,
        candle_body=0.00005,
        upper_wick=0.00005,
        lower_wick=0.00005,
        distance_to_session_high=-0.0040,
        distance_to_session_low=0.0010,
        feature_set_version=FEATURE_SET_VERSION,
    )


def _snap(
    i: int,
    *,
    bar_close: float,
    bar_high: float,
    bar_low: float,
    or_active: bool,
    vwap: float,
    or_high: float,
    or_low: float,
) -> MarketSnapshot:
    ts = _BASE_TS + timedelta(minutes=i)
    mid = round((bar_high + bar_low) / 2, 5)
    bar = MarketBar(
        symbol="EURUSD",
        timestamp_utc=ts,
        timestamp_market=ts,
        timeframe=Timeframe.M1,
        open=bar_close,
        high=bar_high,
        low=bar_low,
        close=bar_close,
        tick_volume=100.0,
        real_volume=0.0,
        spread=2.0,
        source=DataSource.SYNTHETIC,
    )
    return MarketSnapshot(
        snapshot_id=new_snapshot_id(),
        symbol="EURUSD",
        timestamp_utc=ts,
        base_timeframe=Timeframe.M1,
        instrument=_instrument(),
        session_context=_session(ts, or_active=or_active),
        latest_bar=bar,
        bars_m1=[bar],
        bars_m5=[],
        bars_m15=[],
        feature_vector=_fv(ts, vwap=vwap, or_high=or_high, or_low=or_low),
        quality_report=_quality(),
        snapshot_version=SNAPSHOT_VERSION,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builders
# ─────────────────────────────────────────────────────────────────────────────


def build_or_data(n: int = 100) -> list[MarketSnapshot]:
    """
    OR breakout dataset:
      - Snapshots 0-29  : OR window active -> no signal
      - Snapshots 30-99 : OR completed, price breaking above OR high=1.1020
      - Bar: high=1.1045, low=1.1035, close=1.1040
      - VWAP=1.1010 (30 pips below close — but OR strategy ignores VWAP)

    OR LONG labeling (stop=10, target=20 pips):
      entry=1.1020, stop=1.1010, target=1.1040
      high=1.1045 >= entry=1.1020 -> activated
      low=1.1035 > stop=1.1010  -> not stopped
      high=1.1045 >= target=1.1040 -> WIN
    """
    or_cutoff = 30
    snaps = []
    for i in range(n):
        snaps.append(
            _snap(
                i,
                bar_close=1.1040,
                bar_high=1.1045,
                bar_low=1.1035,
                or_active=(i < or_cutoff),
                vwap=1.1010,
                or_high=1.1020,
                or_low=1.1000,
            )
        )
    return snaps


def build_vwap_data(n: int = 100) -> list[MarketSnapshot]:
    """
    VWAP reversion dataset:
      - Snapshots 0-29  : close 3 pips above VWAP -> EXTENSION_TOO_SMALL
      - Snapshots 30-99 : close 15 pips above VWAP -> SHORT candidate
      - VWAP=1.1010, OR range=1.1000-1.1020 (valid for OR too)

    VWAP Fade SHORT labeling (stop=10, target=20 pips):
      entry=1.1025 (close), stop=1.1035, target=1.1005
      Future bars: high=1.1030, low=1.0995
      low=1.0995 <= entry=1.1025 -> activated
      high=1.1030 < stop=1.1035 -> not stopped
      low=1.0995 <= target=1.1005 -> WIN
    """
    or_cutoff = 30
    snaps = []
    for i in range(n):
        is_extended = i >= or_cutoff
        close = 1.1025 if is_extended else 1.1013
        high = 1.1030 if is_extended else 1.1015
        low = 1.0995 if is_extended else 1.1011
        snaps.append(
            _snap(
                i,
                bar_close=close,
                bar_high=high,
                bar_low=low,
                or_active=(i < or_cutoff),
                vwap=1.1010,
                or_high=1.1020,
                or_low=1.1000,
            )
        )
    return snaps


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────


def _section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _pct(v: float | None) -> str:
    return f"{v * 100:.1f}%" if v is not None else "N/A"


def _fp(v: float | None, decimals: int = 1) -> str:
    return f"{v:.{decimals}f}" if v is not None else "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OR and VWAP Fade strategies.")
    parser.add_argument("--snapshots", type=int, default=100)
    args = parser.parse_args()
    n = args.snapshots

    # ── Build datasets ────────────────────────────────────────────────────────
    _section("Building synthetic datasets")
    or_data = build_or_data(n)
    vwap_data = build_vwap_data(n)
    print(f"  OR-data   : {n} snapshots | breakout scenario | bar high=1.1045 low=1.1035")
    print(f"  VWAP-data : {n} snapshots | reversion scenario | bar high=1.1030 low=1.0995")

    # ── Configure strategies ──────────────────────────────────────────────────
    or_engine = OpeningRangeEngine(
        OpeningRangeDefinition(
            strategy_id="or_london_v1",
            session_name="LONDON",
            min_range_pips=5.0,
            max_range_pips=40.0,
        ),
        min_quality_score=MIN_QUALITY_SCORE,
    )
    vwap_engine = VWAPFadeEngine(
        VWAPFadeDefinition(
            strategy_id="vwap_fade_london_v1",
            session_name="LONDON",
            min_distance_to_vwap_pips=10.0,
            max_distance_to_vwap_pips=50.0,
        ),
        min_quality_score=MIN_QUALITY_SCORE,
    )
    or_label_cfg = LabelConfig(stop_pips=10.0, target_pips=20.0, max_bars=30)
    vf_label_cfg = LabelConfig(stop_pips=10.0, target_pips=20.0, max_bars=30)

    # ── Run replays ───────────────────────────────────────────────────────────
    _section("Running replays")
    t0 = time.perf_counter()
    or_result = run_replay(or_data, or_engine, label_config=or_label_cfg)
    t1 = time.perf_counter()
    vf_result = run_replay(vwap_data, vwap_engine, label_config=vf_label_cfg)
    t2 = time.perf_counter()
    print(f"  OR replay   : {t1 - t0:.3f}s  run_id={or_result.summary.run_id}")
    print(f"  VWAP replay : {t2 - t1:.3f}s  run_id={vf_result.summary.run_id}")

    # ── Compute metrics ───────────────────────────────────────────────────────
    or_m = compute_metrics(or_result.records, or_result.labeled_outcomes)
    vf_m = compute_metrics(vf_result.records, vf_result.labeled_outcomes)

    # ── Print comparison ──────────────────────────────────────────────────────
    _section("Strategy Comparison")
    col_w = 18
    metric_w = 26

    header = f"  {'Metric':<{metric_w}}  {'OR (breakout)':<{col_w}}  {'VWAP Fade (reversion)':<{col_w}}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = [
        ("Strategy ID",        or_engine.strategy_id,                        vwap_engine.strategy_id),
        ("Dataset",             "OR-optimized",                               "VWAP-optimized"),
        ("Signal type",         "Breakout",                                   "Mean reversion"),
        ("Direction",           "LONG",                                       "SHORT"),
        ("Snapshots",           str(or_m.total_records),                      str(vf_m.total_records)),
        ("Candidates",          str(or_m.candidate_count),                    str(vf_m.candidate_count)),
        ("No-trade",            str(or_m.no_trade_count),                     str(vf_m.no_trade_count)),
        ("Total labeled",       str(or_m.total_labeled),                      str(vf_m.total_labeled)),
        ("Entry activated",     str(or_m.entry_activated_count),              str(vf_m.entry_activated_count)),
        ("Activation rate",     _pct(or_m.activation_rate),                   _pct(vf_m.activation_rate)),
        ("Wins",                str(or_m.win_count),                          str(vf_m.win_count)),
        ("Losses",              str(or_m.loss_count),                         str(vf_m.loss_count)),
        ("Timeouts",            str(or_m.timeout_count),                      str(vf_m.timeout_count)),
        ("Not activated",       str(or_m.entry_not_activated_count),          str(vf_m.entry_not_activated_count)),
        ("Win rate (activated)", _pct(or_m.win_rate_on_activated),            _pct(vf_m.win_rate_on_activated)),
        ("Avg MFE (pips)",       _fp(or_m.avg_mfe),                           _fp(vf_m.avg_mfe)),
        ("Avg MAE (pips)",       _fp(or_m.avg_mae),                           _fp(vf_m.avg_mae)),
        ("Avg bars to entry",    _fp(or_m.avg_bars_to_entry),                 _fp(vf_m.avg_bars_to_entry)),
        ("Avg bars to resolve",  _fp(or_m.avg_bars_to_resolution),            _fp(vf_m.avg_bars_to_resolution)),
    ]

    for label, or_val, vf_val in rows:
        print(f"  {label:<{metric_w}}  {or_val:<{col_w}}  {vf_val:<{col_w}}")

    _section("Observations")
    print("  Both strategies achieve high win rates on their designed synthetic datasets.")
    print("  They produce opposite signals from the same market state:")
    print("    OR breakout : price above OR high -> LONG (expects continuation)")
    print("    VWAP Fade   : price above VWAP    -> SHORT (expects reversion)")
    print()
    print("  Cross-dataset behaviour (not run here):")
    print("    OR on VWAP-data  -> LONG entry activated; bar.low=1.0995 hits stop -> LOSS")
    print("    VWAP on OR-data  -> SHORT candidate; bar.low=1.1035 misses target  -> TIMEOUT")
    print()
    print("  Both strategies are complementary; they favour different market regimes.")
    print("  Calibrate on real data to validate distance thresholds and RR ratios.")
    print()


if __name__ == "__main__":
    main()
