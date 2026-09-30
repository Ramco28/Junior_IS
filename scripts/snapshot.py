"""Feature 1: log in to OpenSky and get one live snapshot.

    python scripts/snapshot.py                 # whole world
    python scripts/snapshot.py --ohio          # only aircraft over Ohio
    python scripts/snapshot.py --ohio --save   # also save it to data/snapshots/
"""
import argparse
import gzip
import json
from datetime import datetime, timezone

import _path  # noqa: F401  (lets me import adsb from here)
from adsb.auth import TokenManager
from adsb.config import OHIO_BBOX, SNAPSHOT_DIR
from adsb.live import fetch_states, states_to_frame


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ohio", action="store_true", help="limit to the Ohio bounding box")
    ap.add_argument("--save", action="store_true", help="also save the snapshot to data/snapshots/")
    args = ap.parse_args()

    tm = TokenManager.from_json_file()
    # I get the token first on its own, so a login problem shows up here
    # and not in the middle of the data request
    print("Authenticating with OpenSky... ", end="", flush=True)
    tm.token()
    print("ok")

    snap = fetch_states(tm, bbox=OHIO_BBOX if args.ohio else None)
    df = states_to_frame(snap)
    when = datetime.fromtimestamp(snap["time"], timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print(f"\nSnapshot at {when}")
    print(f"Aircraft tracked: {len(df):,}")
    print(f"Airborne: {(~df['on_ground'].astype(bool)).sum():,}")
    if "credits_remaining" in snap:
        print(f"API credits left today: {snap['credits_remaining']:,}")
    # show the first few aircraft that have a position
    cols = ["icao24", "callsign", "origin_country", "latitude", "longitude", "baro_altitude", "velocity"]
    print("\n" + df[cols].dropna(subset=["latitude"]).head(8).to_string(index=False))

    if args.save:
        # I keep the raw JSON (gzipped) so the later features can reuse it
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.fromtimestamp(snap["time"], timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = SNAPSHOT_DIR / f"states_{stamp}.json.gz"
        with gzip.open(path, "wt") as f:
            json.dump(snap, f)
        print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
