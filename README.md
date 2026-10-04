# Junior_IS
### How effectively can machine learning models detect anomalous or potentially unsafe aircraft behavior in ADS-B flight data compared to rule-based approaches?

## Feature Calendar

This project compares machine learning and rule-based approaches to detecting anomalous
aircraft behavior in ADS-B flight data. The table below tracks the features I'm building,
with due dates and time estimates set through Planning Poker. Each row links to its
GitHub issue. The stretch goals listed at the end will be tackled if time permits.

| # | Feature | Due Date | Estimate | Notes | Status |
|---|---------|----------|----------|-------|--------|
| 1 | [Authenticate & retrieve single OpenSky snapshot](https://github.com/Ramco28/Junior_IS/issues/1) | 09/24 | 2h | Foundation for all data ingestion | Done |
| 2 | [Poll live API and store snapshots](https://github.com/Ramco28/Junior_IS/issues/2) | 09/24 | 3h | Adds scheduling + persistence | Done |
| 3 | [Query historical DB, save labeled trajectories](https://github.com/Ramco28/Junior_IS/issues/3) | 09/24 | 4h | Source data for ML training/testing | Done |
| 4 | [Parse raw state vector into normalized record](https://github.com/Ramco28/Junior_IS/issues/4) | 09/24 | 2h | ICAO, callsign, lat/lon, altitude, speed, timestamp | Done |
| 5 | [Compute derived per-aircraft features](https://github.com/Ramco28/Junior_IS/issues/5) | 09/24 | 5h | Altitude/speed change rate, distance between positions | Done |
| 18 | [Enrich aircraft type from the OpenSky aircraft database](https://github.com/Ramco28/Junior_IS/issues/18) | 10/01 | 2h | Type and manufacturer by icao24, so rule thresholds can differ by aircraft type | Not started |
| 6 | [Rule-based anomaly checks (speed, duplicate ICAO, altitude/speed change, route deviation)](https://github.com/Ramco28/Junior_IS/issues/6) | 10/01 | 14h | Four configurable-threshold checks | Not started |
| 7 | [Label historical dataset via weak labeling](https://github.com/Ramco28/Junior_IS/issues/7) | 10/01 | 4h | Uses rule-based checks as initial labels | Not started |
| 8 | [Train baseline ML model](https://github.com/Ramco28/Junior_IS/issues/8) | 10/22 | 8h | e.g. isolation forest or autoencoder | Not started |
| 9 | [Evaluate ML model vs. rule-based detector](https://github.com/Ramco28/Junior_IS/issues/9) | 10/22 | 4h | Agreement/disagreement statistics on held-out data | Not started |
| 10 | [Build backend service running both detectors](https://github.com/Ramco28/Junior_IS/issues/10) | 10/22 | 8h | Shared output format for live data | Not started |
| 11 | [Live map view](https://github.com/Ramco28/Junior_IS/issues/11) | 10/29 | 6h | Displays currently tracked aircraft | Not started |
| 12 | [Highlight flagged aircraft on map](https://github.com/Ramco28/Junior_IS/issues/12) | 10/29 | 2h | Distinct styling from unflagged aircraft | Not started |
| 13 | [Side panel of flagged aircraft](https://github.com/Ramco28/Junior_IS/issues/13) | 10/29 | 4h | Anomaly type, score, detector source | Not started |
| 14 | [Click handling + trajectory detail view](https://github.com/Ramco28/Junior_IS/issues/14) | 10/29 | 4h | Centers map, opens detail on selection | Not started |
| 15 | [Filter and toggle controls](https://github.com/Ramco28/Junior_IS/issues/15) | 10/29 | 3h | Filter by detector; toggle anomaly categories | Not started |
| 16 | [Summary report generation script](https://github.com/Ramco28/Junior_IS/issues/16) | 11/12 | 3h | Precision, recall, agreement metrics | Not started |

## Stretch Goals

| # | Feature | Due Date | Estimate | Notes | Status |
|---|---------|----------|----------|-------|--------|
| 17 | [Historical playback, sequence-based ML model, exportable anomaly report](https://github.com/Ramco28/Junior_IS/issues/17) | if time permits | 18h | Combined stretch bucket | Not started |

## Running the project

Setup (once):

```bash
git clone https://github.com/Ramco28/Junior_IS && cd Junior_IS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The live scripts read my OpenSky API client from `credentials.json` in the repo
root (gitignored). Without it they fall back to anonymous access.

Main command for each finished feature:

| # | Command | What it does |
|---|---------|--------------|
| 1 | `python scripts/snapshot.py --ohio --save` | Logs in with OAuth2 and prints one live snapshot over Ohio |
| 2 | `python scripts/poll.py --ohio --interval 30` | Asks for a new snapshot every 30 s and saves each one |
| 3 | `python scripts/historical.py --start "2026-09-20 14:00" --hours 1` | Queries one hour of past flights over Ohio and saves them as trajectories |
| 4, 5 | `python scripts/build_features.py --live` | Puts the stored snapshots in the normalized format and computes the per-aircraft features (use `--historical <file>` for a historical file) |

Folders:

```
Junior_IS/
  adsb/         the library: config, OAuth2 login, live API, historical database
  scripts/      one command line script per feature, plus two demo helpers
  tests/        pytest tests, run with: python -m pytest
  data/         snapshots and trajectories the scripts save (gitignored)
  README.md
  SETUP_AND_DEMO.md
  requirements.txt
```

See [SETUP_AND_DEMO.md](SETUP_AND_DEMO.md) for the details and the demo run sheet.
