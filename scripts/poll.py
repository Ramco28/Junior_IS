"""Feature 2: ask the live API for a new snapshot every few seconds and save each one.

    python scripts/poll.py --ohio --interval 30
    python scripts/poll.py --ohio --interval 30 --count 3
    python scripts/poll.py --interval 120 --count 50

Each snapshot is saved as it comes in data/snapshots/states_<UTC time>.json.gz,
so I can rerun the later steps (parsing, features, detectors) on the same data.
Stop it with Ctrl+C.

Credits: my OpenSky account gets a daily budget of API credits, and asking for
the whole world costs more than a small area. Polling Ohio is cheap, polling
the world every 30 s is not, so I print the credits left after each request.
"""
import argparse
import gzip
import json
import time
from datetime import datetime, timezone

import requests

import _path  # noqa: F401  (lets me import adsb from here)
from adsb.auth import TokenManager
from adsb.config import OHIO_BBOX, SNAPSHOT_DIR
from adsb.live import RateLimited, fetch_states


def save(snap: dict):
    # the file name is the snapshot's UTC time, so sorting by name sorts by time
    stamp = datetime.fromtimestamp(snap["time"], timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = SNAPSHOT_DIR / f"states_{stamp}.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(snap, f)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=int, default=60, help="seconds between requests (default 60)")
    ap.add_argument("--count", type=int, default=0, help="stop after this many snapshots (0 = run forever)")
    ap.add_argument("--ohio", action="store_true", help="limit to the Ohio bounding box")
    args = ap.parse_args()

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    tm = TokenManager.from_json_file()  # one TokenManager for the whole run, it renews the token by itself
    bbox = OHIO_BBOX if args.ohio else None
    last_time = None
    saved = 0
    print(f"Polling every {args.interval} s ({'Ohio' if bbox else 'world'}). Ctrl+C to stop.")

    try:
        while args.count == 0 or saved < args.count:
            started = time.time()
            try:
                snap = fetch_states(tm, bbox)
                # if OpenSky has not updated yet I get the same snapshot twice, no point saving it
                if snap["time"] == last_time:
                    print("  same data as last request, skipped")
                else:
                    path = save(snap)
                    last_time = snap["time"]
                    saved += 1
                    credits = snap.get("credits_remaining", "?")
                    print(f"  [{saved}] {len(snap['states']):>6,} aircraft -> {path.name}  (credits left: {credits})")
            except RateLimited as e:
                # out of credits: wait as long as OpenSky says, then try again
                print(f"  out of credits, sleeping {e.retry_after_s} s")
                time.sleep(e.retry_after_s)
                continue
            except requests.RequestException as e:
                # a network error should not stop the whole poller
                print(f"  request failed ({e}); will retry next cycle")
            if args.count and saved >= args.count:
                break  # I have all the snapshots I asked for, no need to wait again
            # sleep what is left of the interval so requests stay evenly spaced
            time.sleep(max(0, args.interval - (time.time() - started)))
    except KeyboardInterrupt:
        pass
    total = len(list(SNAPSHOT_DIR.glob("states_*.json.gz")))
    print(f"\nStopped. Saved {saved} this run, {total} snapshots stored in total.")


if __name__ == "__main__":
    main()
