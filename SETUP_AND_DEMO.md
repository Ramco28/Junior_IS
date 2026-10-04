# Setup and Demo

## Setup (once)

```bash
brew install python@3.12
cd ~/Projects/Junior_IS
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The virtual environment uses the native (Apple Silicon) Python from Homebrew,
not Anaconda. My Anaconda was an Intel build and stopped running after I
updated macOS, which broke the old environment. If the Trino login cannot be
saved after rebuilding the environment, delete the old
`trino.opensky-network.org@ramco_28` entry in Keychain Access and log in again.

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

## Demo run sheet (1 to 2 minutes)

Before class:

1. Run the historical query once, so the browser login is done and cached:
   `python scripts/historical.py --start "2026-09-20 14:00" --hours 1`
2. About 10 minutes before presenting, start the poller in a second terminal:
   `python scripts/poll.py --ohio --interval 30`
3. Increase the editor and terminal font size.
4. Keep a screen recording of a successful run as a backup.

Live, in this order:

| Step | Terminal | Command | Takes about |
|---|---|---|---|
| 1 | 1 | `python scripts/snapshot.py --ohio --save` | 3 s |
| 2 | 1 | `python scripts/plot_snapshot.py` | 2 s, then close the window |
| 3 | 2 | (poller already running, point at the growing count) | |
| 4 | 1 | `python scripts/historical.py --start "2026-09-20 14:00" --hours 1` | 6 s with the login cached |
| 5 | 1 | `python scripts/show_trajectory.py` | 1 s |
