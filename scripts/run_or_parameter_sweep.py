"""
scripts/run_or_parameter_sweep.py
───────────────────────────────────
Real calibration grid-search for OpeningRangeEngine.

Usage:
  python scripts/run_or_parameter_sweep.py
  python scripts/run_or_parameter_sweep.py --snapshots 200

What this script does:
  1. Builds synthetic MarketSnapshots with two OR range groups (15-pip and
     25-pip) to make the min_range_pips dimension meaningful.
  2. Runs a full grid sweep:
       min_range_pips  ∈ [10, 15, 20]
       max_range_pips  ∈ [40, 50, 60]
       session         ∈ ["LONDON", "OVERLAP_LONDON_NY", "ALL"]
  3. Prints results sorted by composite score (win_rate × activation_rate).
  4. Selects the best configuration and prints its StrategyBaselineProfile.

Synthetic data notes:
  - Two OR groups: 15-pip range (OR high=1.1015, low=1.1000) and
    25-pip range (OR high=1.1025, low=1.1000), alternating by snapshot index.
  - min_range=10 accepts both; min_range=20 accepts only the 25-pip group.
  - Bars: high = OR_high + 5 pips, low = OR_high - 5 pips for deterministic
    entry activation and WIN outcomes within stop=10, target=10 pips.
  - All snapshots are LONDON session to demonstrate session filter effects.
    With real data, OVERLAP_LONDON_NY configs produce distinct results.

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
    StrategyBaselineProfile,
    rank_sweep_configs,
    select_best_opening_range_config,
)
from aion.analytics.parameter_sweeps import run_parameter_sweep
from aion.analytics.replay_models import SweepPoint
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

# OR range definitions (high, low) → range size in pips
_OR_GROUPS = [
    (1.1015, 1.1000),   # 15 pips — blocked by min_range_pips=20
    (1.1025, 1.1000),   # 25 pips — passes all min_range thresholds
]


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot builder
# ─────────────────────────────────────────────────────────────────────────────


def _build_snapshots(n: int = 200) -> list[MarketSnapshot]:
    """Build synthetic OR sweep snapshots.

    First n//4 snapshots have OR active (skipped by engine).
    Remaining snapshots alternate between the two OR range groups.
    Bars are sized so entry activates and target=10 pips is reachable.
    """
    instrument = _instrument()
    or_cutoff = n // 4
    snaps: list[MarketSnapshot] = []

    for i in range(n):
        ts = _BASE_TS + timedelta(minutes=i)
        or_active = i < or_cutoff
        or_completed = not or_active

        # Alternate OR range groups for post-cutoff snapshots.
        group_idx = i % 2 if or_completed else 0
        or_high, or_low = _OR_GROUPS[group_idx]

        # Bar positioned to activate entry at OR high and hit target=10 pips.
        bar_high = round(or_high + 0.0005, 5)   # +5 pips above OR high
        bar_low = round(or_high - 0.0005, 5)    # -5 pips below OR high
        bar_close = round((bar_high + bar_low) / 2, 5)
        bar_open = bar_close

        bar = MarketBar(
            symbol="EURUSD",
            timestamp_utc=ts,
            timestamp_market=ts,
            timeframe=Timeframe.M1,
            open=bar_open,
            high=bar_high,
            low=bar_low,
            close=bar_close,
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
            is_london=True,
            is_new_york=False,
            is_session_open_window=True,
            opening_range_active=or_active,
            opening_range_completed=or_completed,
            session_name=SessionName.LONDON,
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
            opening_range_high=or_high if or_completed else None,
            opening_range_low=or_low if or_completed else None,
            vwap_session=1.1010,
            spread_mean_20=2.0,
            spread_zscore_20=0.0,
            return_1=0.0001,
            return_5=0.0003,
            candle_body=0.00003,
            upper_wick=0.00002,
            lower_wick=0.00002,
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


def _build_sweep_points() -> list[SweepPoint]:
    """Build the 3 × 3 × 3 = 27-point parameter grid."""
    min_range_values = [10.0, 15.0, 20.0]
    max_range_values = [40.0, 50.0, 60.0]
    # session → allowed_sessions mapping
    session_configs: dict[str, frozenset[str] | None] = {
        "LONDON": frozenset({"LONDON", "OVERLAP_LONDON_NY"}),
        "OVERLAP_LONDON_NY": frozenset({"OVERLAP_LONDON_NY"}),
        "ALL": None,
    }

    points: list[SweepPoint] = []
    for min_r, max_r, (sess_name, allowed) in itertools.product(
        min_range_values, max_range_values, session_configs.items()
    ):
        label = f"min{int(min_r)}_max{int(max_r)}_{sess_name}"
        points.append(
            SweepPoint(
                label=label,
                min_range_pips=min_r,
                max_range_pips=max_r,
                allowed_sessions=allowed,
                stop_pips=10.0,
                target_pips=10.0,
                max_label_bars=20,
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


def _score_str(label: str, metrics) -> str:
    from aion.analytics.baseline_selection import _composite_score  # noqa: PLC0415
    s = _composite_score(metrics)
    return f"{s:.4f}" if s is not None else "  N/A"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Opening Range parameter sweep.")
    parser.add_argument("--snapshots", type=int, default=200)
    args = parser.parse_args()

    _section("Building synthetic snapshots")
    snaps = _build_snapshots(args.snapshots)
    or_cutoff = args.snapshots // 4
    print(f"  Total     : {len(snaps)} snapshots")
    print(f"  OR active : first {or_cutoff} (excluded by engine)")
    print(f"  Group A   : ~{(args.snapshots - or_cutoff) // 2} snaps, OR range = 15 pips")
    print(f"  Group B   : ~{(args.snapshots - or_cutoff) // 2} snaps, OR range = 25 pips")
    print(f"  Session   : all LONDON (OVERLAP_LONDON_NY configs will show 0 candidates)")

    points = _build_sweep_points()
    _section(f"Running sweep — {len(points)} configurations (3×3×3 grid)")
    t0 = time.perf_counter()
    cmp = run_parameter_sweep(snaps, points)
    t1 = time.perf_counter()
    print(f"  Elapsed : {t1 - t0:.3f}s")

    # ── Full ranked table ─────────────────────────────────────────────────────
    _section("Results (sorted by composite score)")
    pairs = [(r.sweep_point.label, r.metrics) for r in cmp.results]
    ranked = rank_sweep_configs(pairs, min_candidates=1)

    hdr = (
        f"  {'Label':<34}"
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
            f"  {label:<34}"
            f"  {score:>7.4f}"
            f"  {m.candidate_count:>6}"
            f"  {_pct(m.win_rate_on_activated):>7}"
            f"  {_pct(m.activation_rate):>7}"
            f"  {_fp(m.avg_mfe):>6}"
            f"  {_fp(m.avg_mae):>6}"
        )

    # Show configs that were filtered out entirely
    filtered_out = [r for r in cmp.results if r.metrics.candidate_count == 0]
    if filtered_out:
        _section(f"Filtered out (0 candidates) — {len(filtered_out)} configs")
        for r in filtered_out:
            print(f"  {r.sweep_point.label}")

    # ── Best configuration ────────────────────────────────────────────────────
    _section("Best configuration")
    result = select_best_opening_range_config(cmp, min_candidates=1)
    if result is None:
        print("  No valid configuration found.")
        return

    best_point, profile = result
    print(f"  Label              : {best_point.label}")
    print(f"  min_range_pips     : {best_point.min_range_pips}")
    print(f"  max_range_pips     : {best_point.max_range_pips}")
    allowed = best_point.allowed_sessions
    print(f"  allowed_sessions   : {sorted(allowed) if allowed else 'ALL'}")
    print(f"  stop_pips          : {best_point.stop_pips}")
    print(f"  target_pips        : {best_point.target_pips}")

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
