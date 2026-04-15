"""
scripts/run_strategy_replay.py
────────────────────────────────
Run a historical Opening Range strategy replay on synthetic snapshots.

Usage:
  python scripts/run_strategy_replay.py
  python scripts/run_strategy_replay.py --output-dir results/replay/

What this script does:
  1. Builds 100 synthetic MarketSnapshots simulating a London session day.
  2. Evaluates OpeningRangeEngine on each snapshot (no-lookahead).
  3. Labels each candidate setup using forward bars from subsequent snapshots.
  4. Prints a detailed run summary.
  5. Saves the journal to --output-dir if provided.

Snapshot sequence:
  - Snapshots 0-29  : London session, OR still accumulating -> NO_TRADE
  - Snapshots 30-99 : London session, OR completed, 20-pip range -> CANDIDATE
    Future bars have high=1.1045, low=1.1035:
      - LONG entry=1.1020 activated (high >= 1.1020)
      - Target=1.1040 hit immediately (high >= 1.1040) -> WIN
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure the project root is on the path when run directly.
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

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
from aion.regime.rules import RuleBasedRegimeDetector
from aion.replay.journal import CandidateJournal
from aion.replay.models import LabelConfig
from aion.replay.runner import run_replay
from aion.strategies.models import OpeningRangeDefinition
from aion.strategies.opening_range import OpeningRangeEngine

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic snapshot builder
# ─────────────────────────────────────────────────────────────────────────────

_UTC = timezone.utc
_BASE_TS = datetime(2024, 1, 15, 8, 0, 0, tzinfo=_UTC)  # Monday London open


def _make_instrument() -> InstrumentSpec:
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


def _make_bar(ts: datetime, *, high: float, low: float) -> MarketBar:
    mid = round((high + low) / 2, 5)
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


def _make_session(ts: datetime, *, or_active: bool, or_completed: bool) -> SessionContext:
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
        opening_range_completed=or_completed,
        session_name=SessionName.LONDON,
        session_open_utc=session_open,
        session_close_utc=session_close,
    )


def _make_fv(ts: datetime) -> FeatureVector:
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
        opening_range_high=1.1020,
        opening_range_low=1.1000,
        vwap_session=1.1010,
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


def build_synthetic_snapshots(n: int = 100) -> list[MarketSnapshot]:
    """
    Build n synthetic snapshots.

    Snapshots 0-29  : OR still active  -> engine will reject (OR_NOT_COMPLETED)
    Snapshots 30-99 : OR completed, valid 20-pip range -> CANDIDATE (LONG)

    All bars have high=1.1045, low=1.1035 so that:
    - Future labeling finds entry (1.1020) activated immediately
    - Target (1.1040) hit on the same bar -> WIN
    """
    instrument = _make_instrument()
    snapshots = []
    or_cutoff = 30  # first 30 bars are the opening range window

    for i in range(n):
        ts = _BASE_TS + timedelta(minutes=i)
        or_active = i < or_cutoff
        or_completed = not or_active

        bar = _make_bar(ts, high=1.1045, low=1.1035)
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
        snap = MarketSnapshot(
            snapshot_id=new_snapshot_id(),
            symbol="EURUSD",
            timestamp_utc=ts,
            base_timeframe=Timeframe.M1,
            instrument=instrument,
            session_context=_make_session(ts, or_active=or_active, or_completed=or_completed),
            latest_bar=bar,
            bars_m1=[bar],
            bars_m5=[],
            bars_m15=[],
            feature_vector=_make_fv(ts),
            quality_report=qr,
            snapshot_version=SNAPSHOT_VERSION,
        )
        snapshots.append(snap)

    return snapshots


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def _print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Opening Range strategy replay on synthetic data."
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        default=100,
        help="Number of synthetic snapshots to generate (default: 100).",
    )
    parser.add_argument(
        "--stop-pips",
        type=float,
        default=10.0,
        help="Stop distance in pips for labeling (default: 10).",
    )
    parser.add_argument(
        "--target-pips",
        type=float,
        default=20.0,
        help="Target distance in pips for labeling (default: 20).",
    )
    parser.add_argument(
        "--max-label-bars",
        type=int,
        default=30,
        help="Max future bars to inspect for labeling (default: 30).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save journal JSONL files (optional).",
    )
    args = parser.parse_args()

    # ── Build snapshots ───────────────────────────────────────────────────────
    _print_section("Building synthetic snapshots")
    t0 = time.perf_counter()
    snapshots = build_synthetic_snapshots(args.snapshots)
    print(f"  Generated : {len(snapshots)} snapshots")
    print(f"  Symbol    : EURUSD  |  OR range: 20 pips (1.1000 - 1.1020)")
    print(f"  Session   : LONDON  |  OR window: first 30 bars")

    # ── Build engine and config ───────────────────────────────────────────────
    engine = OpeningRangeEngine(
        OpeningRangeDefinition(
            strategy_id="or_london_v1",
            session_name="LONDON",
            min_range_pips=5.0,
            max_range_pips=40.0,
        ),
        min_quality_score=MIN_QUALITY_SCORE,
    )
    label_cfg = LabelConfig(
        stop_pips=args.stop_pips,
        target_pips=args.target_pips,
        max_bars=args.max_label_bars,
    )
    detector = RuleBasedRegimeDetector()

    # ── Run replay ────────────────────────────────────────────────────────────
    _print_section("Running replay")
    result = run_replay(
        snapshots,
        engine,
        regime_detector=detector,
        label_config=label_cfg,
    )
    t1 = time.perf_counter()
    print(f"  Run ID    : {result.summary.run_id}")
    print(f"  Elapsed   : {t1 - t0:.3f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    s = result.summary
    _print_section("Run Summary")
    print(f"  Snapshots evaluated : {s.total_snapshots}")
    print(f"  Candidates found    : {s.total_candidates}")
    print(f"  No-trade decisions  : {s.total_no_trade}")
    print(f"  Insufficient data   : {s.total_insufficient_data}")

    if s.total_labeled > 0:
        win_pct = s.label_wins / s.total_labeled * 100
        _print_section("Label Summary")
        print(f"  Stop / Target       : {args.stop_pips:.0f} / {args.target_pips:.0f} pips")
        print(f"  Max look-ahead      : {args.max_label_bars} bars")
        print(f"  Labeled outcomes    : {s.total_labeled}")
        print(f"    Wins              : {s.label_wins} ({win_pct:.1f}%)")
        print(f"    Losses            : {s.label_losses}")
        print(f"    Timeouts          : {s.label_timeouts}")
        print(f"    Not activated     : {s.label_not_activated}")

    # ── Sample candidates ─────────────────────────────────────────────────────
    candidates = [r for r in result.records if r.evaluation_result.outcome.value == "CANDIDATE"]
    if candidates:
        _print_section("Sample Candidates (first 3)")
        for rec in candidates[:3]:
            c = rec.evaluation_result.candidate
            regime = rec.regime_label.value if rec.regime_label else "N/A"
            print(
                f"  [{rec.bar_index:3d}] {rec.timestamp_utc.strftime('%H:%M')} UTC"
                f"  dir={c.direction.value:<5}"
                f"  entry={c.entry_reference:.5f}"
                f"  range={c.range_size_pips:.1f}pips"
                f"  regime={regime}"
            )

    if result.labeled_outcomes:
        _print_section("Sample Labels (first 3)")
        for lbl in result.labeled_outcomes[:3]:
            print(
                f"  {lbl.outcome.value:<22}"
                f"  entry={lbl.entry_reference:.5f}"
                f"  stop={lbl.stop_price:.5f}"
                f"  target={lbl.target_price:.5f}"
                f"  mfe={lbl.mfe_pips or 0:.1f}pips"
                f"  mae={lbl.mae_pips or 0:.1f}pips"
            )

    # ── Persist journal ───────────────────────────────────────────────────────
    if args.output_dir:
        out = Path(args.output_dir)
        journal = CandidateJournal()
        for rec in result.records:
            journal.add_record(rec)
        for lbl in result.labeled_outcomes:
            journal.add_label(lbl)

        rec_path = out / "records.jsonl"
        lbl_path = out / "labels.jsonl"
        journal.save_records_jsonl(rec_path)
        journal.save_labels_jsonl(lbl_path)

        _print_section("Journal Saved")
        print(f"  Records : {rec_path}")
        print(f"  Labels  : {lbl_path}")

    print()


if __name__ == "__main__":
    main()
