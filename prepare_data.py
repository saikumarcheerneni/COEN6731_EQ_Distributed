"""
prepare_data.py
─────────────────────────────────────────────────────────────────
Download earthquake.csv from Kaggle, then run:
    python prepare_data.py --input earthquake.csv --workers 2

Splits into N worker shards inside ./data_shards/
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


COL_MAP_HINTS = {
    "Magnitude": ["mag", "magnitude"],
    "Depth":     ["depth"],
    "Latitude":  ["lat", "latitude"],
    "Longitude": ["lon", "long", "longitude"],
}


def normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for target, hints in COL_MAP_HINTS.items():
        for col in df.columns:
            if any(h in col.lower() for h in hints):
                rename[col] = target
                break
    return df.rename(columns=rename)


def prepare(input_path: str, num_workers: int, max_rows: int = None):
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Raw shape: {df.shape}  columns: {list(df.columns)}")

    df = normalise_cols(df)
    needed = ["Magnitude", "Depth", "Latitude", "Longitude"]
    df = df[needed].dropna()
    print(f"After cleaning: {len(df)} rows")

    # Limit rows if requested
    if max_rows and max_rows < len(df):
        df = df.iloc[:max_rows]
        print(f"Using first {max_rows} rows (--max_rows)")

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split
    out = Path("data_shards")
    out.mkdir(exist_ok=True)
    shards = np.array_split(df, num_workers)
    for i, shard in enumerate(shards):
        p = out / f"shard_{i}.csv"
        shard.to_csv(p, index=False)
        print(f"  shard_{i}.csv  →  {len(shard)} rows")

    print(f"\n✅  {num_workers} shards ready in ./data_shards/")
    print("Next: python core/parameter_server.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",   default="earthquake.csv")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max_rows", type=int, default=None)
    args = p.parse_args()
    prepare(args.input, args.workers, args.max_rows)
