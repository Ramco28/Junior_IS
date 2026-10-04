"""Features 4 and 5: normalize my data and compute the per-aircraft features.

    python scripts/build_features.py --historical data/historical/trajectories_ohio_20260920T1400_1h.parquet
    python scripts/build_features.py --live      # all stored snapshots in data/snapshots/

The result is saved as a Parquet file in data/features/. I also print the
5 reports with the largest speed mismatch, which is a preview of what the
rule-based checks (feature 6) will flag.
"""
import argparse
from pathlib import Path

import pandas as pd

import _path  # noqa: F401  (lets me import adsb from here)
from adsb.config import FEATURES_DIR
from adsb.features import add_features
from adsb.normalize import load_snapshots, normalize_historical


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # exactly one of the two sources must be given
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--historical", metavar="PARQUET", help="a file saved by scripts/historical.py")
    source.add_argument("--live", action="store_true", help="use all snapshots in data/snapshots/")
    args = ap.parse_args()

    if args.live:
        normalized = load_snapshots()
        name = "features_live.parquet"
    else:
        normalized = normalize_historical(pd.read_parquet(args.historical))
        # trajectories_ohio_..._1h.parquet becomes features_ohio_..._1h.parquet
        name = Path(args.historical).name.replace("trajectories_", "features_")

    df = add_features(normalized)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FEATURES_DIR / name
    df.to_parquet(path, index=False)

    print(f"Rows:           {len(df):,}")
    print(f"Aircraft:       {df['icao24'].nunique():,}")
    print(f"Trajectories:   {df['trajectory_id'].nunique():,}")
    print(f"Saved {path}")

    # I rank by the size of the mismatch, whether the implied speed is too
    # high or too low (abs() removes the sign)
    top = df.loc[df["speed_mismatch_mps"].abs().sort_values(ascending=False).index[:5]]
    top = top.assign(utc=pd.to_datetime(top["time"], unit="s").dt.strftime("%H:%M:%S"))
    cols = ["trajectory_id", "callsign", "utc", "dt_s", "distance_m",
            "velocity_mps", "implied_speed_mps", "speed_mismatch_mps"]
    print("\nLargest speed mismatches (implied speed minus reported speed):")
    print(top[cols].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
