"""
tests/integration/test_historical_pipeline.py
───────────────────────────────────────────────
End-to-end integration test for the historical pipeline.

Covers:
  - Full pipeline runs on a synthetic CSV without errors
  - PipelineResult fields are all populated correctly
  - Bars are sorted ascending after the pipeline
  - Features list length matches bars_m1 list length
  - Quality report included and score > 0
  - Snapshot is populated with the correct symbol and version
  - Snapshot latest_bar matches the last M1 bar
  - drop_incomplete_last_bar=True drops exactly 1 bar
  - drop_incomplete_last_bar=False keeps all bars
  - persist=True writes Parquet and JSON files
  - Saved bars round-trip correctly through persistence layer
  - Saved features round-trip correctly
  - Saved snapshot round-trip correctly
  - Two pipeline runs on identical data produce different snapshot IDs
"""

from __future__ import annotations

import csv
import math
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from aion.core.constants import SNAPSHOT_VERSION, FEATURE_SET_VERSION
from aion.core.enums import AssetClass, Timeframe
from aion.core.models import InstrumentSpec
from aion.data.pipeline import PipelineError, run_historical_pipeline
from aion.data.persistence import load_bars, load_features, load_snapshot


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_UTC = timezone.utc
_BASE_TS = datetime(2024, 1, 15, 0, 0, 0, tzinfo=_UTC)


def make_eurusd() -> InstrumentSpec:
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


def write_synthetic_csv(path: Path, n_bars: int = 100) -> None:
    """
    Write a synthetic MT5-format CSV with `n_bars` M1 bars starting at
    _BASE_TS.  Prices drift gently upward so ATR and rolling stats are
    non-trivially computed.
    """
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>",
             "<TICKVOL>", "<VOL>", "<SPREAD>"]
        )
        base_price = 1.1000
        price_step = 0.0001
        for i in range(n_bars):
            ts = _BASE_TS + timedelta(minutes=i)
            close = base_price + i * price_step
            open_ = close - price_step * 0.4
            high = close + price_step * 0.6
            low = open_ - price_step * 0.6
            writer.writerow([
                ts.strftime("%Y.%m.%d"),
                ts.strftime("%H:%M:%S"),
                f"{open_:.5f}",
                f"{high:.5f}",
                f"{low:.5f}",
                f"{close:.5f}",
                "100",
                "0",
                "2",
            ])


# ─────────────────────────────────────────────────────────────────────────────
# Basic pipeline run
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def pipeline_result():
    """Run the pipeline once and share the result across tests in this module."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "EURUSD_M1.csv"
        write_synthetic_csv(csv_path, n_bars=120)
        result = run_historical_pipeline(
            csv_path,
            instrument=make_eurusd(),
            local_timezone="Etc/UTC",
            drop_incomplete_last_bar=True,
        )
    return result


def test_pipeline_runs_without_error(pipeline_result):
    assert pipeline_result is not None


def test_pipeline_run_id_generated(pipeline_result):
    assert pipeline_result.pipeline_run_id.startswith("run_")
    assert len(pipeline_result.pipeline_run_id) > 4


def test_bars_loaded_count(pipeline_result):
    assert pipeline_result.bars_loaded == 120


def test_bars_after_normalise(pipeline_result):
    assert pipeline_result.bars_after_normalise == 120


def test_bars_dropped_incomplete_is_one(pipeline_result):
    """With drop_incomplete_last_bar=True, exactly 1 bar is dropped."""
    assert pipeline_result.bars_dropped_incomplete == 1


def test_bars_m1_count(pipeline_result):
    """After dropping the last bar: 120 - 1 = 119 bars."""
    assert len(pipeline_result.bars_m1) == 119


def test_bars_m1_sorted_ascending(pipeline_result):
    timestamps = [b.timestamp_utc for b in pipeline_result.bars_m1]
    assert timestamps == sorted(timestamps)


def test_bars_m5_is_non_empty(pipeline_result):
    assert len(pipeline_result.bars_m5) > 0


def test_bars_m15_is_non_empty(pipeline_result):
    assert len(pipeline_result.bars_m15) > 0


def test_features_m1_length_matches_bars(pipeline_result):
    """One FeatureVector per M1 bar."""
    assert len(pipeline_result.features_m1) == len(pipeline_result.bars_m1)


def test_features_symbol_matches(pipeline_result):
    for fv in pipeline_result.features_m1:
        assert fv.symbol == "EURUSD"


def test_features_timeframe_is_m1(pipeline_result):
    for fv in pipeline_result.features_m1:
        assert fv.timeframe == Timeframe.M1


def test_features_version(pipeline_result):
    assert all(
        fv.feature_set_version == FEATURE_SET_VERSION
        for fv in pipeline_result.features_m1
    )


def test_quality_report_included(pipeline_result):
    assert pipeline_result.quality_report is not None
    assert pipeline_result.quality_report.rows_checked > 0


def test_quality_score_above_zero(pipeline_result):
    assert pipeline_result.quality_report.quality_score > 0.0


def test_quality_score_above_threshold(pipeline_result):
    """Clean synthetic data should produce high quality score."""
    assert pipeline_result.quality_report.quality_score >= 0.9


def test_elapsed_seconds_positive(pipeline_result):
    assert pipeline_result.elapsed_seconds > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot fields
# ─────────────────────────────────────────────────────────────────────────────


def test_snapshot_symbol(pipeline_result):
    assert pipeline_result.snapshot.symbol == "EURUSD"


def test_snapshot_version(pipeline_result):
    assert pipeline_result.snapshot.snapshot_version == SNAPSHOT_VERSION


def test_snapshot_latest_bar_is_last_m1_bar(pipeline_result):
    assert pipeline_result.snapshot.latest_bar == pipeline_result.bars_m1[-1]


def test_snapshot_is_usable(pipeline_result):
    assert pipeline_result.snapshot.is_usable is True


def test_snapshot_feature_vector_included(pipeline_result):
    assert pipeline_result.snapshot.feature_vector is not None


def test_snapshot_quality_report_included(pipeline_result):
    assert pipeline_result.snapshot.quality_report is not None


def test_snapshot_bars_m1_trimmed_to_window(pipeline_result):
    """Default window_m1=100 — 119 bars → trim to 100."""
    assert len(pipeline_result.snapshot.bars_m1) == 100


# ─────────────────────────────────────────────────────────────────────────────
# drop_incomplete_last_bar=False
# ─────────────────────────────────────────────────────────────────────────────


def test_keep_last_bar_retains_all():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "EURUSD_M1.csv"
        write_synthetic_csv(csv_path, n_bars=50)
        result = run_historical_pipeline(
            csv_path,
            instrument=make_eurusd(),
            local_timezone="Etc/UTC",
            drop_incomplete_last_bar=False,
        )
    assert result.bars_dropped_incomplete == 0
    assert result.bars_after_normalise == result.bars_loaded
    assert len(result.bars_m1) == 50


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────


def test_two_runs_produce_different_snapshot_ids():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "EURUSD_M1.csv"
        write_synthetic_csv(csv_path, n_bars=50)
        instrument = make_eurusd()
        r1 = run_historical_pipeline(csv_path, instrument=instrument, local_timezone="Etc/UTC")
        r2 = run_historical_pipeline(csv_path, instrument=instrument, local_timezone="Etc/UTC")
    assert r1.snapshot.snapshot_id != r2.snapshot.snapshot_id


def test_two_runs_same_bar_data():
    """Same CSV → same bar timestamps and prices."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "EURUSD_M1.csv"
        write_synthetic_csv(csv_path, n_bars=50)
        instrument = make_eurusd()
        r1 = run_historical_pipeline(csv_path, instrument=instrument, local_timezone="Etc/UTC")
        r2 = run_historical_pipeline(csv_path, instrument=instrument, local_timezone="Etc/UTC")
    assert len(r1.bars_m1) == len(r2.bars_m1)
    for b1, b2 in zip(r1.bars_m1, r2.bars_m1):
        assert b1.timestamp_utc == b2.timestamp_utc
        assert b1.close == pytest.approx(b2.close)


# ─────────────────────────────────────────────────────────────────────────────
# Persistence round-trip
# ─────────────────────────────────────────────────────────────────────────────


def test_persist_writes_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        csv_path = root / "EURUSD_M1.csv"
        write_synthetic_csv(csv_path, n_bars=50)

        result = run_historical_pipeline(
            csv_path,
            instrument=make_eurusd(),
            local_timezone="Etc/UTC",
            persist=True,
            persist_bar_root=root / "bars",
            persist_feature_root=root / "features",
            persist_snapshot_root=root / "snapshots",
        )

        assert len(result.persist_paths) > 0
        for path in result.persist_paths:
            assert path.exists(), f"Expected path to exist: {path}"


def test_persist_bars_round_trip():
    """Bars written to Parquet can be loaded back with identical values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        csv_path = root / "EURUSD_M1.csv"
        write_synthetic_csv(csv_path, n_bars=50)

        result = run_historical_pipeline(
            csv_path,
            instrument=make_eurusd(),
            local_timezone="Etc/UTC",
            drop_incomplete_last_bar=False,
            persist=True,
            persist_bar_root=root / "bars",
        )

        # Find written parquet files
        bar_files = list((root / "bars").rglob("*.parquet"))
        assert len(bar_files) > 0

        # Load and compare
        loaded_bars = []
        for bf in bar_files:
            loaded_bars.extend(load_bars(bf, Timeframe.M1))
        loaded_bars.sort(key=lambda b: b.timestamp_utc)

        assert len(loaded_bars) == len(result.bars_m1)
        for orig, loaded in zip(result.bars_m1, loaded_bars):
            assert orig.timestamp_utc == loaded.timestamp_utc
            assert orig.symbol == loaded.symbol
            assert orig.open == pytest.approx(loaded.open)
            assert orig.close == pytest.approx(loaded.close)
            assert orig.is_valid == loaded.is_valid


def test_persist_features_round_trip():
    """Features written to Parquet can be loaded back with identical values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        csv_path = root / "EURUSD_M1.csv"
        write_synthetic_csv(csv_path, n_bars=50)

        result = run_historical_pipeline(
            csv_path,
            instrument=make_eurusd(),
            local_timezone="Etc/UTC",
            drop_incomplete_last_bar=False,
            persist=True,
            persist_feature_root=root / "features",
        )

        feature_files = list((root / "features").rglob("*.parquet"))
        assert len(feature_files) > 0

        loaded_fvs = []
        for ff in feature_files:
            loaded_fvs.extend(load_features(ff, Timeframe.M1))
        loaded_fvs.sort(key=lambda fv: fv.timestamp_utc)

        assert len(loaded_fvs) == len(result.features_m1)
        for orig, loaded in zip(result.features_m1, loaded_fvs):
            assert orig.timestamp_utc == loaded.timestamp_utc
            assert orig.symbol == loaded.symbol
            # ATR may be None for early bars — compare with None-safe approx
            if orig.atr_14 is not None:
                assert loaded.atr_14 is not None
                assert orig.atr_14 == pytest.approx(loaded.atr_14, rel=1e-6)
            else:
                assert loaded.atr_14 is None


def test_persist_snapshot_round_trip():
    """Snapshot written to JSON can be loaded back and is identical."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        csv_path = root / "EURUSD_M1.csv"
        write_synthetic_csv(csv_path, n_bars=50)

        result = run_historical_pipeline(
            csv_path,
            instrument=make_eurusd(),
            local_timezone="Etc/UTC",
            persist=True,
            persist_snapshot_root=root / "snapshots",
        )

        snap_files = list((root / "snapshots").rglob("*.json"))
        assert len(snap_files) == 1

        loaded_snap = load_snapshot(snap_files[0])
        assert loaded_snap.snapshot_id == result.snapshot.snapshot_id
        assert loaded_snap.symbol == result.snapshot.symbol
        assert loaded_snap.timestamp_utc == result.snapshot.timestamp_utc
        assert loaded_snap.snapshot_version == result.snapshot.snapshot_version


# ─────────────────────────────────────────────────────────────────────────────
# Feature ATR availability
# ─────────────────────────────────────────────────────────────────────────────


def test_atr_available_after_14_bars():
    """With 120 bars, ATR-14 should be non-None for bars at index >= 13."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "EURUSD_M1.csv"
        write_synthetic_csv(csv_path, n_bars=120)
        result = run_historical_pipeline(
            csv_path,
            instrument=make_eurusd(),
            local_timezone="Etc/UTC",
            drop_incomplete_last_bar=False,
        )

    # At least the last half of features should have a valid ATR
    late_features = result.features_m1[20:]
    assert all(fv.atr_14 is not None for fv in late_features)


def test_return_1_none_for_first_bar():
    """return_1 requires a previous bar — first bar should be None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "EURUSD_M1.csv"
        write_synthetic_csv(csv_path, n_bars=30)
        result = run_historical_pipeline(
            csv_path,
            instrument=make_eurusd(),
            local_timezone="Etc/UTC",
            drop_incomplete_last_bar=False,
        )

    assert result.features_m1[0].return_1 is None


def test_return_1_available_from_second_bar():
    """return_1 is a log return — should be available from bar index 1 onward."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "EURUSD_M1.csv"
        write_synthetic_csv(csv_path, n_bars=30)
        result = run_historical_pipeline(
            csv_path,
            instrument=make_eurusd(),
            local_timezone="Etc/UTC",
            drop_incomplete_last_bar=False,
        )

    assert result.features_m1[1].return_1 is not None
