"""Feature 3: query the OpenSky historical database and save the trajectories.

    python scripts/historical.py --start "2026-09-20 14:00" --hours 1
    python scripts/historical.py --start "2026-09-20 14:00" --hours 2 --sample 5

I query state_vectors_data4 over Ohio for a UTC time window, split the reports
into one trajectory per flight, and save them as a Parquet file in
data/historical/. The first run opens a browser so I can log in to OpenSky.
"""
import argparse
from datetime import timedelta

import _path  # noqa: F401  (lets me import adsb from here)
from adsb.config import HISTORICAL_DIR, OHIO_BBOX
from adsb.historical import fetch_state_vectors, parse_utc, split_trajectories


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help='UTC start, e.g. "2026-09-20 14:00"')
    ap.add_argument("--hours", type=float, default=1.0, help="length of the window (default 1)")
    ap.add_argument("--sample", type=int, default=10, help="keep one report every N seconds (default 10)")
    ap.add_argument("--gap", type=int, default=900, help="silence in seconds that starts a new trajectory")
    args = ap.parse_args()

    start = parse_utc(args.start)
    end = start + timedelta(hours=args.hours)
    print(f"Querying {start:%Y-%m-%d %H:%M} to {end:%H:%M} UTC over Ohio...")

    df = fetch_state_vectors(start, end, OHIO_BBOX, args.sample)
    df = split_trajectories(df, args.gap)

    # I use Parquet because it is much smaller and faster to load than CSV
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    path = HISTORICAL_DIR / f"trajectories_ohio_{start:%Y%m%dT%H%M}_{args.hours:g}h.parquet"
    df.to_parquet(path, index=False)

    print(f"State vectors:  {len(df):,}")
    print(f"Aircraft:       {df['icao24'].nunique():,}")
    print(f"Trajectories:   {df['trajectory_id'].nunique():,}")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
