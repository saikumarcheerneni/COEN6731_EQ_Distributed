"""
prepare_data.py
─────────────────────────────────────────────────────────────────
Two modes:
  1. Real CSV:   python prepare_data.py --input earthquake.csv --workers 2
  2. Synthetic:  python prepare_data.py --synthetic --workers 2 --rows 100000
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


def generate_synthetic(n: int = 100_000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic earthquake dataset — no CSV needed."""
    rng = np.random.default_rng(seed)
    print(f"Generating {n:,} synthetic earthquake records...")

    zones = [
        # lat_mean, lat_std, lon_mean, lon_std, depth_mean, depth_std, weight
        (35.5,  5.0,  139.5,  3.0,  30, 25, 0.12),   # Japan
        (-6.0,  4.0,  107.0,  6.0,  60, 40, 0.10),   # Indonesia
        (55.0,  4.0,  160.0,  5.0,  50, 30, 0.08),   # Kamchatka
        (60.0,  3.0, -150.0,  5.0,  40, 30, 0.08),   # Alaska
        (-25.0, 6.0,  -70.0,  2.0, 100, 60, 0.10),   # South America
        (18.0,  3.0,  -97.0,  2.0,  25, 20, 0.07),   # Central America
        (38.0,  4.0,   35.0,  5.0,  15, 15, 0.06),   # Turkey
        (28.0,  3.0,   85.0,  4.0,  20, 15, 0.06),   # Himalayas
        (37.5,  2.0, -119.5,  1.5,  10, 10, 0.05),   # California
        (0.0,  15.0,    0.0, 30.0,   5,  5, 0.28),   # Global background
    ]

    rows = []
    for lat_m, lat_s, lon_m, lon_s, dep_m, dep_s, w in zones:
        n_zone = int(n * w)
        lat = rng.normal(lat_m, lat_s, n_zone).clip(-90, 90)
        lon = rng.normal(lon_m, lon_s, n_zone).clip(-180, 180)
        dep = (rng.exponential(dep_m, n_zone) + abs(rng.normal(0, dep_s, n_zone))).clip(0, 700)
        mag = (rng.exponential(0.9, n_zone) + 1.5 + dep * 0.003 + rng.normal(0, 0.3, n_zone)).clip(1.0, 9.5)
        rows.append(pd.DataFrame({
            "Magnitude": mag.astype(np.float32),
            "Depth":     dep.astype(np.float32),
            "Latitude":  lat.astype(np.float32),
            "Longitude": lon.astype(np.float32),
        }))

    df = pd.concat(rows, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True).iloc[:n]
    print(f"  Magnitude range: {df.Magnitude.min():.2f} – {df.Magnitude.max():.2f}")
    print(f"  Depth range:     {df.Depth.min():.1f} – {df.Depth.max():.1f} km")
    return df


def _split_and_save(df: pd.DataFrame, num_workers: int):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    out = Path("data_shards")
    out.mkdir(exist_ok=True)
    indices = np.array_split(np.arange(len(df)), num_workers)
    shards  = [df.iloc[idx] for idx in indices]
    for i, shard in enumerate(shards):
        p = out / f"shard_{i}.csv"
        shard.to_csv(p, index=False)
        print(f"  shard_{i}.csv  →  {len(shard):,} rows")
    print(f"\n✅  {num_workers} shards ready in ./data_shards/")
    print("Next steps:")
    print("  python core/parameter_server.py --workers 2")
    print("  python core/worker.py --id 0")
    print("  python core/worker.py --id 1")


def prepare(input_path: str, num_workers: int, max_rows: int = None):
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Raw shape: {df.shape}  columns: {list(df.columns)}")

    df = normalise_cols(df)
    needed  = ["Magnitude", "Depth", "Latitude", "Longitude"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}  Available: {list(df.columns)}")
        return

    df = df[needed].dropna()
    print(f"After cleaning: {len(df):,} rows")

    if max_rows and max_rows < len(df):
        df = df.iloc[:max_rows]
        print(f"Using first {max_rows:,} rows (--max_rows)")

    _split_and_save(df, num_workers)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",     default="earthquake.csv")
    p.add_argument("--synthetic", action="store_true",
                   help="Generate synthetic data — no CSV needed")
    p.add_argument("--workers",   type=int, default=2)
    p.add_argument("--rows",      type=int, default=100_000,
                   help="Row count for synthetic mode")
    p.add_argument("--max_rows",  type=int, default=None)
    args = p.parse_args()

    if args.synthetic:
        df = generate_synthetic(args.rows)
        _split_and_save(df, args.workers)
    else:
        prepare(args.input, args.workers, args.max_rows)