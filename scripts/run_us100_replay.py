"""
scripts/run_us100_replay.py
────────────────────────────
First real-data replay of the AION system on US100.cash M1 data.

Pipeline:
  CSV → MarketBar → FeatureVector → SessionContext → MarketSnapshot
  → Strategies (OR + VWAP) → Risk Allocation → Paper Trading → Summary

Usage:
  python scripts/run_us100_replay.py

The script loads data/raw/us100_3months_m1.csv and runs the full pipeline.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aion.app import PaperTradingConfig, PaperTradingResult, format_summary, run_paper_loop
from aion.core.constants import SNAPSHOT_VERSION
from aion.core.enums import AssetClass, SessionName, Timeframe
from aion.core.ids import new_snapshot_id
from aion.core.models import (
    DataQualityReport,
    InstrumentSpec,
    MarketBar,
    MarketSnapshot,
)
from aion.data.csv_loader import load_bars
from aion.data.features import compute_feature_series
from aion.data.resampler import resample_bars
from aion.data.sessions import build_session_context
from aion.data.validator import validate_bars
from aion.risk.models import RiskProfile
from aion.strategies.models import OpeningRangeDefinition
from aion.strategies.opening_range import OpeningRangeEngine
from aion.strategies.vwap_fade import VWAPFadeDefinition, VWAPFadeEngine


# ─────────────────────────────────────────────────────────────────────────────
# Instrument specification
# ─────────────────────────────────────────────────────────────────────────────

def _us100() -> InstrumentSpec:
    """US100.cash (NAS100) CFD — typical MT5 broker spec."""
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
        trading_hours_label="Mon-Fri, nearly 24h (broker dependent)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Strategy engines
# ─────────────────────────────────────────────────────────────────────────────

# For US100: tick_size=0.01, pip_multiplier=100 → 1 pip = 1.0 point
_PIP_MULTIPLIER = 100.0
_TICK_SIZE = 0.01
_PIP_SIZE = _TICK_SIZE * _PIP_MULTIPLIER  # 1.0 point


def _or_engine() -> OpeningRangeEngine:
    return OpeningRangeEngine(
        OpeningRangeDefinition(
            strategy_id="or_london_us100",
            session_name="LONDON",
            min_range_pips=5.0,
            max_range_pips=50.0,
            pip_multiplier=_PIP_MULTIPLIER,
            max_retest_penetration_points=10.0,
        )
    )


def _vwap_engine() -> VWAPFadeEngine:
    return VWAPFadeEngine(
        VWAPFadeDefinition(
            strategy_id="vwap_fade_overlap_us100",
            session_name="OVERLAP_LONDON_NY",
            min_distance_to_vwap_pips=10.0,
            max_distance_to_vwap_pips=50.0,
            require_rejection=False,
            pip_multiplier=_PIP_MULTIPLIER,
            tick_size=_TICK_SIZE,
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def _config() -> PaperTradingConfig:
    return PaperTradingConfig(
        risk_profile=RiskProfile(
            account_equity=100_000.0,
            max_risk_per_trade_pct=1.0,     # 1% per trade
            max_daily_risk_pct=2.0,         # 2% daily cap
            max_concurrent_positions=3,
            max_positions_per_strategy=2,
        ),
        instrument=_us100(),
        stop_distance_points=10.0,          # 10 points
        target_distance_points=20.0,        # 20 points (2R)
        pip_size=_PIP_SIZE,                 # 1.0 point
        max_bars_open=60,                   # 60 M1 bars = 1 hour timeout
    )


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot builder (efficient batch)
# ─────────────────────────────────────────────────────────────────────────────

_ACTIVE_SESSIONS = {
    SessionName.LONDON,
    SessionName.NEW_YORK,
    SessionName.OVERLAP_LONDON_NY,
}

# Minimum M1 bars before we start building snapshots (for warm-up of features)
_WARMUP_BARS = 30


def _compute_daily_quality(
    bars: list[MarketBar],
    sessions: list,
) -> dict[date, DataQualityReport]:
    """
    Validate bars grouped by trading day.

    This avoids the penalty from weekend/holiday gaps that drag the
    whole-series quality score below MIN_QUALITY_SCORE.  Each day's
    bars are contiguous within their session, so quality is accurate.
    """
    # Group bar indices by trading day
    day_bars: dict[date, list[MarketBar]] = defaultdict(list)
    for bar, ctx in zip(bars, sessions):
        day_bars[ctx.trading_day].append(bar)

    result: dict[date, DataQualityReport] = {}
    for day, day_bar_list in day_bars.items():
        if len(day_bar_list) >= 2:
            result[day] = validate_bars(day_bar_list, Timeframe.M1)

    return result


def _build_snapshots(
    bars_m1: list[MarketBar],
    instrument: InstrumentSpec,
) -> list[MarketSnapshot]:
    """
    Build MarketSnapshots efficiently using batch feature computation.

    Only generates snapshots during active trading sessions (London, NY,
    Overlap) to avoid wasting memory on off-hours bars.

    Approach:
      1. Compute all features in one O(N) pass.
      2. Build session context per bar (also done inside feature series,
         but we need the SessionContext objects for snapshots).
      3. Resample M5/M15 once.
      4. Assemble snapshots only for active-session bars after warm-up.
    """
    n = len(bars_m1)
    print(f"  Computing features for {n:,} bars (batch O(N))...")
    t0 = time.perf_counter()

    features = compute_feature_series(
        bars_m1,
        timeframe=Timeframe.M1,
        market_timezone=instrument.market_timezone,
        broker_timezone=instrument.broker_timezone,
        local_timezone="Etc/UTC",
    )
    t_feat = time.perf_counter() - t0
    print(f"  Features computed in {t_feat:.1f}s")

    print("  Resampling to M5 and M15...")
    bars_m5 = resample_bars(bars_m1, Timeframe.M5)
    bars_m15 = resample_bars(bars_m1, Timeframe.M15)

    print("  Building session contexts...")
    sessions = []
    for bar in bars_m1:
        ctx = build_session_context(
            bar.timestamp_utc,
            market_timezone=instrument.market_timezone,
            broker_timezone=instrument.broker_timezone,
            local_timezone="Etc/UTC",
        )
        sessions.append(ctx)

    # Quality reports per trading day — avoids penalising weekend/holiday gaps
    # that would drag the whole-series score below MIN_QUALITY_SCORE.
    print("  Computing quality per trading day...")
    daily_quality = _compute_daily_quality(bars_m1, sessions)

    # Build snapshot index for M5/M15 bars — map from timestamp to index
    m5_ts = [b.timestamp_utc for b in bars_m5]
    m15_ts = [b.timestamp_utc for b in bars_m15]

    print("  Assembling snapshots for active sessions...")
    snapshots: list[MarketSnapshot] = []
    window_m1 = 100
    window_m5 = 100
    window_m15 = 100

    for i in range(n):
        if i < _WARMUP_BARS:
            continue

        ctx = sessions[i]
        if ctx.session_name not in _ACTIVE_SESSIONS:
            continue

        bar = bars_m1[i]
        fv = features[i]

        # Window of M1 bars ending at i
        start_m1 = max(0, i + 1 - window_m1)
        snap_m1 = bars_m1[start_m1:i + 1]

        # M5/M15 bars up to this timestamp
        bar_ts = bar.timestamp_utc
        # Binary search for last M5/M15 bar <= bar_ts
        snap_m5 = _window_up_to(bars_m5, m5_ts, bar_ts, window_m5)
        snap_m15 = _window_up_to(bars_m15, m15_ts, bar_ts, window_m15)

        trading_day = ctx.trading_day
        quality = daily_quality.get(trading_day)
        if quality is None:
            continue

        snapshots.append(
            MarketSnapshot(
                snapshot_id=new_snapshot_id(),
                symbol=instrument.symbol,
                timestamp_utc=bar_ts,
                base_timeframe=Timeframe.M1,
                instrument=instrument,
                session_context=ctx,
                latest_bar=bar,
                bars_m1=snap_m1,
                bars_m5=snap_m5,
                bars_m15=snap_m15,
                feature_vector=fv,
                quality_report=quality,
                snapshot_version=SNAPSHOT_VERSION,
            )
        )

    return snapshots


def _window_up_to(
    bars: list[MarketBar],
    timestamps: list[datetime],
    up_to: datetime,
    window: int,
) -> list[MarketBar]:
    """Return the last `window` bars with timestamp <= up_to."""
    # Simple linear scan from the end (timestamps are sorted)
    # For large datasets, bisect would be faster, but this is called per
    # snapshot and bars_m5/m15 are 5-15x smaller than bars_m1.
    import bisect
    idx = bisect.bisect_right(timestamps, up_to)
    start = max(0, idx - window)
    return bars[start:idx]


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    csv_path = _root / "data" / "raw" / "us100_3months_m1.csv"

    instrument = _us100()
    config = _config()
    engines = [_or_engine(), _vwap_engine()]

    print()
    print("AION Real-Data Replay — US100.cash M1")
    print("=" * 50)
    print(f"Data file : {csv_path.name}")
    print(f"Instrument: {instrument.symbol}")
    print(f"Strategies: {', '.join(e.strategy_id for e in engines)}")
    print(f"Risk      : {config.risk_profile.max_risk_per_trade_pct}% per trade, "
          f"{config.risk_profile.max_daily_risk_pct}% daily cap")
    print(f"Account   : ${config.risk_profile.account_equity:,.0f}")
    print(f"Stop/Target: {config.stop_distance_points:.0f} / "
          f"{config.target_distance_points:.0f} points ({_PIP_SIZE:.1f} per pip)")
    print(f"Timeout   : {config.max_bars_open} bars (M1)")

    # ── Step 1: Load bars ────────────────────────────────────────────────────
    _section("Step 1: Loading CSV")
    t0 = time.perf_counter()
    bars_m1 = load_bars(csv_path, instrument, drop_last=True)
    t_load = time.perf_counter() - t0

    if not bars_m1:
        print("  ERROR: No bars loaded. Check the CSV file.")
        return

    first_ts = bars_m1[0].timestamp_utc
    last_ts = bars_m1[-1].timestamp_utc
    days = (last_ts - first_ts).days

    print(f"  Bars loaded : {len(bars_m1):,}")
    print(f"  Date range  : {first_ts.date()} to {last_ts.date()} ({days} days)")
    print(f"  Load time   : {t_load:.2f}s")

    # ── Step 2: Build snapshots ──────────────────────────────────────────────
    _section("Step 2: Building snapshots")
    t1 = time.perf_counter()
    snapshots = _build_snapshots(bars_m1, instrument)
    t_snap = time.perf_counter() - t1
    print(f"  Snapshots built : {len(snapshots):,}")
    print(f"  Build time      : {t_snap:.1f}s")

    if not snapshots:
        print("  WARNING: No snapshots in active sessions. Nothing to replay.")
        return

    # ── Step 3: Run paper trading loop ───────────────────────────────────────
    _section("Step 3: Running paper trading loop")
    t2 = time.perf_counter()
    result = run_paper_loop(snapshots, engines, config)
    t_loop = time.perf_counter() - t2
    print(f"  Loop time : {t_loop:.1f}s")

    # ── Step 4: Summary ──────────────────────────────────────────────────────
    _section("Results")
    print(format_summary(result))

    # ── Step 5: Extended metrics ─────────────────────────────────────────────
    _print_extended_metrics(result, len(snapshots))

    # ── Closed positions sample ──────────────────────────────────────────────
    closed = result.state.all_closed()
    if closed:
        _section(f"Sample Closed Trades (first 10 of {len(closed)})")
        for cp in closed[:10]:
            sid = cp.open_position.order.strategy_id
            direction = cp.open_position.order.direction.value
            fill = cp.open_position.fill.fill_price
            close_p = cp.close_price
            pnl_sign = "+" if cp.pnl_amount >= 0 else ""
            r_sign = "+" if cp.r_multiple >= 0 else ""
            ts = cp.open_position.fill.fill_timestamp.strftime("%Y-%m-%d %H:%M")
            print(
                f"  {ts} {sid:<28} {direction:<5} "
                f"fill={fill:>10.2f}  close={close_p:>10.2f}  "
                f"P&L={pnl_sign}${cp.pnl_amount:>8.2f}  "
                f"R={r_sign}{cp.r_multiple:.2f}  "
                f"({cp.close_reason.value})"
            )

    # ── Timing ───────────────────────────────────────────────────────────────
    total = time.perf_counter() - t0
    _section("Timing")
    print(f"  CSV load       : {t_load:>6.1f}s")
    print(f"  Snapshot build : {t_snap:>6.1f}s")
    print(f"  Paper loop     : {t_loop:>6.1f}s")
    print(f"  Total          : {total:>6.1f}s")
    print()


def _print_extended_metrics(result: PaperTradingResult, total_snapshots: int) -> None:
    """Print additional metrics beyond the standard summary."""
    s = result.summary
    closed = result.state.all_closed()

    _section("Extended Metrics")

    print(f"  Snapshots processed  : {total_snapshots:,}")
    print(f"  Total signals        : {s.total_signals}")
    print(f"  Risk approved        : {s.risk_approved}")
    print(f"  Trades executed      : {s.total_executed}")

    # Activation rate
    if s.total_signals > 0:
        activation_pct = s.total_executed / s.total_signals * 100
        print(f"  Activation rate      : {activation_pct:.1f}%")
    else:
        print(f"  Activation rate      : N/A (no signals)")

    # Win rate
    total_closed = s.positions_closed
    if total_closed > 0:
        win_pct = s.win_count / total_closed * 100
        print(f"  Win rate             : {win_pct:.1f}% ({s.win_count}/{total_closed})")
    else:
        print(f"  Win rate             : N/A (no closed positions)")

    # PnL in points (using pip_size = 1.0 for US100)
    print(f"  Total P&L            : ${s.total_pnl:+,.2f}")

    if s.avg_r_multiple is not None:
        print(f"  Avg R-multiple       : {s.avg_r_multiple:+.2f}R")

    # Per-strategy breakdown
    for bd in s.strategy_breakdown:
        signals = bd.signals
        executed = bd.executed
        act_pct = executed / signals * 100 if signals > 0 else 0
        wr = bd.win_count / bd.closed * 100 if bd.closed > 0 else 0
        print(f"  [{bd.strategy_id}]")
        print(f"    Signals: {signals}  Executed: {executed}  "
              f"Activation: {act_pct:.1f}%  "
              f"W/L: {bd.win_count}/{bd.loss_count}  "
              f"Win rate: {wr:.1f}%  "
              f"P&L: ${bd.pnl:+,.2f}")


if __name__ == "__main__":
    main()
