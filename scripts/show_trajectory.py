"""Print one saved trajectory in time order. I use this in my demo.

    python scripts/show_trajectory.py                  # longest airborne trajectory in the newest file
    python scripts/show_trajectory.py --id a1b2c3_0    # a specific trajectory
"""
import argparse
from datetime import datetime, timezone

import pandas as pd

import _path  # noqa: F401  (lets me import adsb from here)
from adsb.config import HISTORICAL_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="parquet file (default: newest in data/historical/)")
    ap.add_argument("--id", help="trajectory_id to show (default: the longest airborne one)")
    ap.add_argument("--rows", type=int, default=12)
    args = ap.parse_args()

    files = list(HISTORICAL_DIR.glob("*.parquet"))
    if not args.file and not files:
        raise SystemExit("No trajectories yet. Run scripts/historical.py first.")
    path = args.file or max(files, key=lambda p: p.stat().st_mtime)  # newest file
    df = pd.read_parquet(path)
    # by default I pick the trajectory with the most airborne reports, because the
    # longest one overall can be a plane parked at an airport for the whole hour
    airborne = df[df["onground"] == False]  # noqa: E712 (column can hold None)
    tid = args.id or airborne["trajectory_id"].value_counts().idxmax()
    t = df[df["trajectory_id"] == tid].sort_values("time")

    first, last = (datetime.fromtimestamp(x, timezone.utc) for x in (t["time"].iloc[0], t["time"].iloc[-1]))
    print(f"{path.name if hasattr(path, 'name') else path}")
    print(f"Trajectory {tid}  callsign {t['callsign'].iloc[0] or '?'}  "
          f"{len(t)} reports  {first:%H:%M} to {last:%H:%M} UTC\n")
    # turn the Unix time into a readable clock time for the table
    t = t.assign(utc=pd.to_datetime(t["time"], unit="s").dt.strftime("%H:%M:%S"))
    print(t[["utc", "lat", "lon", "baroaltitude", "velocity", "heading"]].head(args.rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
