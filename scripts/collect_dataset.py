"""Feature 7: collect the historical dataset I label and train on.

    python scripts/collect_dataset.py

One hour of data is too little to train and test a model, so this downloads
the same one-hour query (see scripts/historical.py) for several days and
several times of day. Every query covers exactly one hour, so it reads one
hour partition and stays far below OpenSky's limits (30 minutes, 100 GB).

Windows that are already in data/historical/ are skipped, so I can run the
script again if a query fails halfway.
"""
from datetime import timedelta

import _path  # noqa: F401  (lets me import adsb from here)
from adsb.config import HISTORICAL_DIR, OHIO_BBOX
from adsb.historical import fetch_state_vectors, parse_utc, split_trajectories

# Two days out of every three over about three weeks: 16 days, with every
# weekday at least twice. I started with the first row (every third day) and
# added the second row later, because 8 days gave too few anomalous flights
# to hold out a validation set and a test set.
DAYS = ["2026-09-15", "2026-09-18", "2026-09-21", "2026-09-24",
        "2026-09-27", "2026-09-30", "2026-10-03", "2026-10-06",
        "2026-09-16", "2026-09-19", "2026-09-22", "2026-09-25",
        "2026-09-28", "2026-10-01", "2026-10-04", "2026-10-07"]

# Start hours in UTC. Ohio is 4 hours behind UTC in summer.
HOURS = [
    "02:00",  # 10 pm in Ohio: quiet night traffic
    "12:00",  # 8 am: morning airline rush
    "17:00",  # 1 pm: midday, when most small aircraft fly
    "22:00",  # 6 pm: evening rush
]

SAMPLE_S = 10  # one report every 10 s per aircraft, like scripts/historical.py


def main():
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    windows = [parse_utc(f"{day} {hour}") for day in DAYS for hour in HOURS]
    total_rows = 0
    failed = []
    for i, start in enumerate(windows, 1):
        # same file name as scripts/historical.py, so the other scripts find it
        path = HISTORICAL_DIR / f"trajectories_ohio_{start:%Y%m%dT%H%M}_1h.parquet"
        label = f"[{i}/{len(windows)}] {start:%Y-%m-%d %H:%M} UTC"
        if path.exists():
            print(f"{label}: already have it")
            continue
        try:
            df = fetch_state_vectors(start, start + timedelta(hours=1), OHIO_BBOX, SAMPLE_S)
        except Exception as e:  # one failed query should not stop the others
            print(f"{label}: FAILED ({e})")
            failed.append(start)
            continue
        df = split_trajectories(df)
        df.to_parquet(path, index=False)
        total_rows += len(df)
        print(f"{label}: {len(df):,} rows, {df['icao24'].nunique():,} aircraft")

    print(f"\nDownloaded {total_rows:,} new rows. Failed windows: {len(failed)}")
    for start in failed:
        print(f"  {start:%Y-%m-%d %H:%M}")


if __name__ == "__main__":
    main()
