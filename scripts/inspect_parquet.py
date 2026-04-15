"""
scripts/inspect_parquet.py
───────────────────────────
Quick CLI to inspect any Parquet file produced by the pipeline.

Automatically detects whether the file contains bars or features based on
its column names, and prints a summary accordingly.

Usage:
    python scripts/inspect_parquet.py <path_to_parquet>
    python scripts/inspect_parquet.py data/bars/EURUSD/M1/2024-01.parquet
    python scripts/inspect_parquet.py data/features/EURUSD/M1/2024-01.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when run directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _detect_type(columns: list[str]) -> str:
    """Guess whether the Parquet file contains bars or feature vectors."""
    cols = set(columns)
    if "atr_14" in cols or "return_1" in cols:
        return "features"
    if {"open", "high", "low", "close"}.issubset(cols):
        return "bars"
    return "unknown"


def _print_bar_summary(df) -> None:
    import pandas as pd
    print("\n  Type   : MarketBar")
    print(f"  Symbol : {df['symbol'].unique().tolist()}")
    print(f"  TF     : {df['timeframe'].unique().tolist()}")
    ts = pd.to_datetime(df["timestamp_utc"])
    print(f"  From   : {ts.min()}")
    print(f"  To     : {ts.max()}")
    print(f"  OHLC range: {df['low'].min():.5f} – {df['high'].max():.5f}")
    print(f"  is_valid=False : {(~df['is_valid']).sum()}")


def _print_feature_summary(df) -> None:
    import pandas as pd
    print("\n  Type   : FeatureVector")
    print(f"  Symbol : {df['symbol'].unique().tolist()}")
    print(f"  TF     : {df['timeframe'].unique().tolist()}")
    ts = pd.to_datetime(df["timestamp_utc"])
    print(f"  From   : {ts.min()}")
    print(f"  To     : {ts.max()}")
    # Count columns with any non-null value
    float_cols = [c for c in df.columns if df[c].dtype.kind == "f"]
    null_counts = {c: df[c].isna().sum() for c in float_cols}
    fully_null = [c for c, n in null_counts.items() if n == len(df)]
    if fully_null:
        print(f"  All-null features : {fully_null}")
    else:
        print(f"  All feature columns have at least one non-null value.")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python inspect_parquet.py <path_to_parquet>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    try:
        import pandas as pd
    except ImportError:
        print("pandas is required: pip install pandas pyarrow")
        sys.exit(1)

    df = pd.read_parquet(path)

    print(f"\nFile : {path}")
    print(f"Rows : {len(df):,}")
    print(f"Cols : {list(df.columns)}")

    file_type = _detect_type(list(df.columns))

    if file_type == "bars":
        _print_bar_summary(df)
    elif file_type == "features":
        _print_feature_summary(df)

    print(f"\nTypes:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nLast  5 rows:\n{df.tail()}")
    print(f"\nStats:\n{df.describe()}")


if __name__ == "__main__":
    main()
