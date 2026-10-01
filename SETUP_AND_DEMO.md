# Setup and Demo

## Setup (once)

```bash
cd ~/Projects/Junior_IS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

I keep the repo in `~/Projects` and not on the Desktop, because my Desktop syncs
to iCloud and macOS moves files to the cloud when the disk is low, which made
the scripts hang while they downloaded again.

Live API credentials: log in to opensky-network.org, open the Account page,
create an API client, and download `credentials.json` into the repo root.
It is listed in `.gitignore`, so it will never be committed.

Anonymous fallback: if `credentials.json` is missing or OpenSky rejects it,
`snapshot.py` and `poll.py` print one warning line and keep going without a
login. Anonymous access gets 400 credits per day (4,000 when logged in), so
`poll.py` then waits 300 s between requests by default (60 s when logged in).

Historical database: `scripts/historical.py` logs in as `ramco_28`. The first
run opens a browser window for OpenSky login (same as the Trino CLI), and the
token is cached in the macOS keychain afterwards.

## Features

| Issue | Script | What it does |
|---|---|---|
| #1 | `scripts/snapshot.py` | Authenticates (OAuth2) and prints one live snapshot |
| #2 | `scripts/poll.py` | Polls on a schedule, stores each raw snapshot as `.json.gz` |
| #3 | `scripts/historical.py` | Queries `state_vectors_data4` over Ohio, splits flights into trajectories, saves Parquet |
| helper | `scripts/show_trajectory.py` | Prints one saved trajectory in time order |
| helper | `scripts/plot_snapshot.py` | Scatter plot of the newest snapshot, colored by altitude |

## Demo run sheet (60 seconds)

Before class:

1. Run the historical query once (it can take minutes):
   `python scripts/historical.py --start "2026-09-20 14:00" --hours 1`
2. About 10 minutes before presenting, start the poller in its own tab:
   `python scripts/poll.py --ohio --interval 30`
3. Pre-type the commands below in separate terminal tabs. Increase the terminal font size.
4. Keep a screen recording of a successful run as a backup.

Live:

| Time | Tab | Command |
|---|---|---|
| 0:00 | 1 | `python scripts/snapshot.py --ohio --save` |
| 0:15 | 1 | `python scripts/plot_snapshot.py` |
| 0:25 | 2 | (poller already running, point at the growing count) |
| 0:40 | 3 | `python scripts/show_trajectory.py` |
