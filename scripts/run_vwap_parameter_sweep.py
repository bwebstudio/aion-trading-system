"""
scripts/run_vwap_parameter_sweep.py
─────────────────────────────────────
Real calibration grid-search for VWAPFadeEngine.

Usage:
  python scripts/run_vwap_parameter_sweep.py
  python scripts/run_vwap_parameter_sweep.py --snapshots-per-group 30

What this script does:
  1. Builds synthetic MarketSnapshots with five VWAP distance groups
     (5, 8, 10, 12, 15 pips from VWAP) to make min_distance meaningful.
  2. Runs a full grid sweep:
       min_distance_to_vwap_pips ∈ [5, 8, 10, 12]
       require_rejection          ∈ [True, False]
       session_name               ∈ ["LONDON", "OVERLAP_LONDON_NY", "ALL"]
  3. Prints results sorted by composite score (win_rate × activation_rate).
  4. Selects the best configuration and prints its StrategyBaselineProfile.

Synthetic data notes:
  - Five distance groups: 5, 8, 10, 12, 15 pips above VWAP=1.1010.
  - min_distance=5 admits all groups; min_distance=12 admits only 12 and 15-pip.
  - Half of each group uses LONDON session, half uses OVERLAP_LONDON_NY.
  - Bars are SHORT-fade setups: bar.low well below entry_reference so
    entry activates immediately and target=10 pips is reachable.
  - require_rejection=True is satisfied because open > close (bearish bars).

Replace _build_snapshots() with your live data loader for real calibration.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aion.analytics.baseline_selection import (
    rank_sweep_configs,
    select_best_vwap_fade_config,
)
from aion.analytics.parameter_sweeps import run_vwap_parameter_sweep
from aion.analytics.replay_models import VWAPSweepPoint
from aion.core.constants import FEATURE_SET_VERSION, SNAPSHOT_VERSION
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

_UTC = timezone.utc
_BASE_TS = datetime(2024, 1, 15, 8, 0, 0, tzinfo=_UTC)

_VWAP = 1.1010
_PIP = 0.00001 * 10  # 5-decimal forex, 10 ticks per pip

# Distance groups: pips above VWAP → SHORT fade candidates
_DISTANCE_GROUPS = [5, 8, 10, 12, 15]


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot builder
# ─────────────────────────────────────────────────────────────────────────────


def _build_snapshots(n_per_group: int = 20) -> list[MarketSnapshot]:
    """Build synthetic VWAP sweep snapshots.

    Structure: n_per_group snapshots × 5 distance groups × 2 sessions
    (LONDON, OVERLAP_LONDON_NY).  Total = n_per_group × 5 × 2.

    Each bar is a bearish SHORT-fade setup:
      open  = close + 1 pip   (bearish bar → require_rejection passes)
      high  = close + 2 pips  (well below stop = close + 10 pips)
      low   = close - 11 pips (below entry, below target = close - 10 pips)

    So for every snapshot: entry activates on the immediate next bar and
    the target is reached on the same bar → outcome = WIN.
    """
    instrument = _instrument()
    snaps: list[MarketSnapshot] = []
    ts_cursor = _BASE_TS

    for dist_pips in _DISTANCE_GROUPS:
        close = round(_VWAP + dist_pips * _PIP, 5)
        bar_open = round(close + 1 * _PIP, 5)
        bar_high = round(close + 2 * _PIP, 5)
        bar_low = round(close - 11 * _PIP, 5)

        for snap_idx in range(n_per_group):
            # Alternate LONDON / OVERLAP_LONDON_NY within each group
            if snap_idx < n_per_group // 2:
                session_name = SessionName.LONDON
                is_london = True
                is_new_york = False
            else:
                session_name = SessionName.OVERLAP_LONDON_NY
                is_london = True
                is_new_york = True

            ts = ts_cursor
            ts_cursor = ts + timedelta(minutes=1)

            bar = MarketBar(
                symbol="EURUSD",
                timestamp_utc=ts,
                timestamp_market=ts,
                timeframe=Timeframe.M1,
                open=bar_open,
                high=bar_high,
                low=bar_low,
                close=close,
                tick_volume=100.0,
                real_volume=0.0,
                spread=2.0,
                source=DataSource.SYNTHETIC,
            )
            session_open = ts.replace(hour=8, minute=0, second=0)
            session_close_t = ts.replace(hour=16, minute=30, second=0)
            session = SessionContext(
                trading_day=ts.date(),
                broker_time=ts,
                market_time=ts,
                local_time=ts,
                is_asia=False,
                is_london=is_london,
                is_new_york=is_new_york,
                is_session_open_window=True,
                opening_range_active=False,
                opening_range_completed=True,
                session_name=session_name,
                session_open_utc=session_open,
                session_close_utc=session_close_t,
            )
            fv = FeatureVector(
                symbol="EURUSD",
                timestamp_utc=ts,
                timeframe=Timeframe.M1,
                atr_14=0.00015,
                rolling_range_10=0.0010,
                rolling_range_20=0.0012,
                volatility_percentile_20=0.50,
                session_high=1.1060,
                session_low=1.0980,
                opening_range_high=1.1020,
                opening_range_low=1.1000,
                vwap_session=_VWAP,
                spread_mean_20=2.0,
                spread_zscore_20=0.0,
                return_1=0.0001,
                return_5=0.0003,
                candle_body=abs(close - bar_open),
                upper_wick=bar_high - max(bar_open, close),
                lower_wick=min(bar_open, close) - bar_low,
                distance_to_session_high=1.1060 - close,
                distance_to_session_low=close - 1.0980,
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
            snaps.append(
                MarketSnapshot(
                    snapshot_id=new_snapshot_id(),
                    symbol="EURUSD",
                    timestamp_utc=ts,
                    base_timeframe=Timeframe.M1,
                    instrument=instrument,
                    session_context=session,
                    latest_bar=bar,
                    bars_m1=[bar],
                    bars_m5=[],
                    bars_m15=[],
                    feature_vector=fv,
                    quality_report=qr,
                    snapshot_version=SNAPSHOT_VERSION,
                )
            )

    return snaps


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


# ─────────────────────────────────────────────────────────────────────────────
# Sweep point grid builder
# ─────────────────────────────────────────────────────────────────────────────


def _build_sweep_points() -> list[VWAPSweepPoint]:
    """Build the 4 × 2 × 3 = 24-point parameter grid."""
    min_distances = [5.0, 8.0, 10.0, 12.0]
    require_rejection_values = [False, True]
    session_names = ["LONDON", "OVERLAP_LONDON_NY", "ALL"]

    points: list[VWAPSweepPoint] = []
    for min_d, req_rej, sess in itertools.product(
        min_distances, require_rejection_values, session_names
    ):
        rej_label = "rej" if req_rej else "norej"
        label = f"d{int(min_d)}_{rej_label}_{sess}"
        points.append(
            VWAPSweepPoint(
                label=label,
                min_distance_to_vwap_pips=min_d,
                max_distance_to_vwap_pips=50.0,
                require_rejection=req_rej,
                session_name=sess,
                stop_pips=10.0,
                target_pips=10.0,
                max_label_bars=5,
            )
        )
    return points


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────


def _section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _pct(v: float | None) -> str:
    return f"{v * 100:.1f}%" if v is not None else "  N/A "


def _fp(v: float | None, d: int = 1) -> str:
    return f"{v:.{d}f}" if v is not None else "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="VWAP Fade parameter sweep.")
    parser.add_argument("--snapshots-per-group", type=int, default=20)
    args = parser.parse_args()

    n_per_group = args.snapshots_per_group
    total = n_per_group * len(_DISTANCE_GROUPS) * 2

    _section("Building synthetic snapshots")
    snaps = _build_snapshots(n_per_group)
    print(f"  Total snapshots  : {len(snaps)} ({n_per_group} per group × 5 groups × 2 sessions)")
    print(f"  VWAP             : {_VWAP}")
    print(f"  Distance groups  : {_DISTANCE_GROUPS} pips above VWAP (SHORT fade)")
    print(f"  Session split    : half LONDON, half OVERLAP_LONDON_NY per group")
    print(f"  Bar setup        : bearish (open > close), low = entry - 11 pips  => WIN")

    points = _build_sweep_points()
    _section(f"Running sweep — {len(points)} configurations (4×2×3 grid)")
    t0 = time.perf_counter()
    cmp = run_vwap_parameter_sweep(snaps, points)
    t1 = time.perf_counter()
    print(f"  Elapsed : {t1 - t0:.3f}s")

    # ── Full ranked table ─────────────────────────────────────────────────────
    _section("Results (sorted by composite score)")
    pairs = [(r.sweep_point.label, r.metrics) for r in cmp.results]
    ranked = rank_sweep_configs(pairs, min_candidates=1)

    hdr = (
        f"  {'Label':<28}"
        f"  {'Score':>7}"
        f"  {'Cands':>6}"
        f"  {'Win%':>7}"
        f"  {'Act%':>7}"
        f"  {'MFE':>6}"
        f"  {'MAE':>6}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for score, label, m in ranked:
        print(
            f"  {label:<28}"
            f"  {score:>7.4f}"
            f"  {m.candidate_count:>6}"
            f"  {_pct(m.win_rate_on_activated):>7}"
            f"  {_pct(m.activation_rate):>7}"
            f"  {_fp(m.avg_mfe):>6}"
            f"  {_fp(m.avg_mae):>6}"
        )

    filtered_out = [r for r in cmp.results if r.metrics.candidate_count == 0]
    if filtered_out:
        _section(f"Filtered out (0 candidates) — {len(filtered_out)} configs")
        for r in filtered_out:
            print(f"  {r.sweep_point.label}")

    # ── Best configuration ────────────────────────────────────────────────────
    _section("Best configuration")
    result = select_best_vwap_fade_config(cmp, min_candidates=1)
    if result is None:
        print("  No valid configuration found.")
        return

    best_point, profile = result
    print(f"  Label                    : {best_point.label}")
    print(f"  min_distance_to_vwap_pips: {best_point.min_distance_to_vwap_pips}")
    print(f"  max_distance_to_vwap_pips: {best_point.max_distance_to_vwap_pips}")
    print(f"  require_rejection        : {best_point.require_rejection}")
    print(f"  session_name             : {best_point.session_name}")
    print(f"  stop_pips                : {best_point.stop_pips}")
    print(f"  target_pips              : {best_point.target_pips}")

    _section("StrategyBaselineProfile (Risk Allocation v1 input)")
    print(f"  strategy_id              : {profile.strategy_id}")
    print(f"  session                  : {profile.session}")
    print(f"  activation_rate          : {_pct(profile.activation_rate)}")
    print(f"  win_rate                 : {_pct(profile.win_rate)}")
    print(f"  avg_mfe                  : {_fp(profile.avg_mfe)} pips")
    print(f"  avg_mae                  : {_fp(profile.avg_mae)} pips")
    print(f"  expected_resolution_bars : {_fp(profile.expected_resolution_bars)}")
    print()


if __name__ == "__main__":
    main()
