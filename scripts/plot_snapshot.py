"""Plot where the aircraft are in the newest saved snapshot. I use this in my demo.

    python scripts/plot_snapshot.py
    python scripts/plot_snapshot.py --file data/snapshots/states_20260929T150000Z.json.gz
"""
import argparse
import gzip
import json
from datetime import datetime, timezone

import matplotlib.pyplot as plt

import _path  # noqa: F401  (lets me import adsb from here)
from adsb.config import SNAPSHOT_DIR
from adsb.live import states_to_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="snapshot file (default: newest in data/snapshots/)")
    args = ap.parse_args()

    files = list(SNAPSHOT_DIR.glob("states_*.json.gz"))
    if not args.file and not files:
        raise SystemExit("No snapshots yet. Run scripts/snapshot.py --ohio --save first.")
    path = args.file or max(files)  # the names are UTC times, so the biggest name is the newest
    with gzip.open(path, "rt") as f:
        snap = json.load(f)
    # keep only aircraft that have a position and are in the air
    df = states_to_frame(snap).dropna(subset=["latitude", "longitude"])
    df = df[~df["on_ground"].astype(bool)]
    when = datetime.fromtimestamp(snap["time"], timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    fig, ax = plt.subplots(figsize=(12, 6.5))
    dot_size = 3 if len(df) > 500 else 30  # tiny dots for the whole world, bigger ones for Ohio
    # each dot is one aircraft, colored by altitude
    sc = ax.scatter(df["longitude"], df["latitude"], c=df["baro_altitude"].fillna(0),
                    s=dot_size, cmap="viridis", vmin=0, vmax=13000)
    fig.colorbar(sc, ax=ax, label="Barometric altitude (m)")
    ax.set_title(f"{len(df):,} airborne aircraft, {when}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
