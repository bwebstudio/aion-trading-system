# Module 01 — Market Engine + Data Pipeline

## What it does

Converts raw OHLCV CSV data into a fully structured, validated, feature-enriched
`MarketSnapshot` — the single object that every downstream module (strategy engine,
regime detector, etc.) consumes.

**Pipeline stages:**

```
CSV file
  └─ csv_adapter     → list[RawBar]          (parse, no tz conversion)
  └─ normalizer      → list[MarketBar]        (broker tz → UTC → market tz, OHLC guard)
  └─ sort ascending by timestamp_utc
  └─ drop last bar   (incomplete-last-bar policy, configurable)
  └─ validator       → DataQualityReport      (8 quality checks, quality_score)
  └─ resampler       → bars_m5, bars_m15      (M1 → M5/M15, mean spread)
  └─ feature series  → list[FeatureVector]    (O(N) batch pass, no lookahead)
  └─ build_snapshot  → MarketSnapshot         (windowed bars + quality + features)
  └─ (optional) persist → Parquet + JSON
```

---

## Running the sample script

```bash
# Generate a synthetic 200-bar CSV and run the full pipeline
python scripts/run_pipeline_sample.py

# Run on your own CSV
python scripts/run_pipeline_sample.py path/to/EURUSD_M1.csv

# Run and persist outputs to data/pipeline_output/
python scripts/run_pipeline_sample.py --persist
```

**Expected output:**

```
Running pipeline on: data/sample/EURUSD_M1_sample.csv

------------------------------------------------------------
PIPELINE RESULT
------------------------------------------------------------
  Run ID        : run_4270fbde5dc8
  Elapsed       : 0.083s
...
```

---

## Running tests

```bash
# All tests (unit + integration)
python -m pytest tests/ -q

# Unit tests only
python -m pytest tests/unit/ -q

# Integration tests only
python -m pytest tests/integration/ -q

# Single module
python -m pytest tests/unit/test_normalizer.py -v
```

**Test count:** 200 tests (166 unit, 34 integration).

---

## Inspecting Parquet output

```bash
python scripts/inspect_parquet.py data/pipeline_output/bars/EURUSD/M1/2024-01.parquet
python scripts/inspect_parquet.py data/pipeline_output/features/EURUSD/M1/2024-01.parquet
```

---

## Storage layout

```
data/
  sample/                          # auto-generated synthetic CSV
  pipeline_output/
    bars/EURUSD/M1/2024-01.parquet
    features/EURUSD/M1/2024-01.parquet
    snapshots/EURUSD/20240115_031800_snap_abc123.json
```

---

## Important design decisions

### `drop_incomplete_last_bar` (default: True)

MT5 CSV exports frequently contain a partial last candle (still forming at export
time). With `drop_incomplete_last_bar=True`, the final M1 bar is always dropped
before building the snapshot.

Use `False` only when you can guarantee the source produces only closed bars
(e.g. a live feed that delivers bars on close).

### Missing bars — PROVISIONAL

The validator counts gaps using rounding arithmetic:

```python
missing = round(delta / expected_delta) - 1
```

This counts ALL gaps including weekends, holidays, and forex overnight periods.
High `missing_bars` counts are normal for multi-day datasets; they do not
necessarily mean data quality is poor. A calendar-aware gap filter (using session
definitions) will replace this in a future iteration.

### Session / VWAP reset policy

VWAP and session H/L reset when `session_open_utc` changes in the feature series
incremental pass. This means:

- A `LONDON` → `OVERLAP_LONDON_NY` transition causes a reset at NY open (13:30 UTC
  winter / 13:30 UTC summer), because the overlap anchor is the NY session definition.
- The London-morning VWAP (08:00–13:30) is separate from the overlap VWAP (13:30–close).
- A full London-day VWAP (reset only at 08:00 GMT/BST) can be added as an additional
  feature in a later feature set version.

### Snapshots as JSON

`MarketSnapshot` objects are persisted as JSON using Pydantic's `model_dump_json()`.
This is intentional: snapshots are self-describing, inspectable with any text editor,
and easy to replay without a database. Parquet was considered but rejected at this
stage — the structured fields of a snapshot (nested models, optional lists) do not
flatten cleanly into columnar form without lossy schema flattening.

### `timestamp_market` not stored

Parquet files for bars and features omit `timestamp_market`. It is derived on load
from `timestamp_utc` + market timezone. This keeps files smaller and prevents stale
data if a timezone database is updated.

---

## Known limitations

| # | Limitation | Impact | Future fix |
|---|-----------|--------|-----------|
| 1 | Missing-bars count includes weekends/holidays | Inflates missing_bars for multi-day CSV | Calendar-aware session filter |
| 2 | VWAP resets at LONDON→OVERLAP boundary | Session VWAP is not full-day London VWAP | Additional full-day VWAP feature |
| 3 | ATR uses simple rolling mean, not Wilder's EMA | Minor difference for window >= 14 | Optional Wilder variant in research env |
| 4 | No MT5 live connection yet | Historical CSV only | mt5_adapter.py stub is ready |
| 5 | Single timeframe feature series (M1 only) | No M5/M15 feature vectors in pipeline | Multi-TF feature pass in next iteration |
| 6 | No multi-symbol support in pipeline | One CSV / one instrument per run | Extend `run_historical_pipeline` to accept instrument list |
| 7 | Snapshot window sizes are global constants | Cannot vary per instrument | Pass window sizes per instrument spec |

---

## Key files

| File | Role |
|------|------|
| `aion/data/csv_adapter.py` | MT5 + generic CSV parsing → RawBar |
| `aion/data/normalizer.py` | RawBar → MarketBar (tz, OHLC guard) |
| `aion/data/sessions.py` | DST-correct session detection |
| `aion/data/validator.py` | 8 quality checks + quality_score formula |
| `aion/data/resampler.py` | M1 → M5/M15/H1 aggregation |
| `aion/data/features.py` | FeatureVector: rolling + session (O(N) batch) |
| `aion/data/snapshots.py` | Assemble MarketSnapshot |
| `aion/data/persistence.py` | Parquet (bars/features) + JSON (snapshots) |
| `aion/data/pipeline.py` | Orchestrates all stages → PipelineResult |
| `aion/core/models.py` | All domain models (Pydantic v2, frozen) |
| `aion/core/constants.py` | ATR_PERIOD, session defs, snapshot windows, thresholds |
| `tests/unit/` | 166 unit tests |
| `tests/integration/` | 34 integration tests |
