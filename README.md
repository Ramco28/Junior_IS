# Junior_IS
### How effectively can machine learning models detect anomalous or potentially unsafe aircraft behavior in ADS-B flight data compared to rule-based approaches?

## Feature Calendar

This project compares machine learning and rule-based approaches to detecting anomalous
aircraft behavior in ADS-B flight data. The table below tracks the features I'm building,
with due dates and time estimates set through Planning Poker. Each row links to its
GitHub issue.

With my professor's agreement, the scope of this Junior I.S. was reduced: it ends with the
baseline model (#8), its comparison with the rules (#9) and the summary report (#16). The
live backend, the dashboard and the stretch goals continue in my Senior I.S.

| # | Feature | Due Date | Estimate | Notes | Status |
|---|---------|----------|----------|-------|--------|
| 1 | [Authenticate & retrieve single OpenSky snapshot](https://github.com/Ramco28/Junior_IS/issues/1) | 09/24 | 2h | Foundation for all data ingestion | Done |
| 2 | [Poll live API and store snapshots](https://github.com/Ramco28/Junior_IS/issues/2) | 09/24 | 3h | Adds scheduling + persistence | Done |
| 3 | [Query historical DB, save labeled trajectories](https://github.com/Ramco28/Junior_IS/issues/3) | 09/24 | 4h | Source data for ML training/testing | Done |
| 4 | [Parse raw state vector into normalized record](https://github.com/Ramco28/Junior_IS/issues/4) | 09/24 | 2h | ICAO, callsign, lat/lon, altitude, speed, timestamp | Done |
| 5 | [Compute derived per-aircraft features](https://github.com/Ramco28/Junior_IS/issues/5) | 09/24 | 5h | Altitude/speed change rate, distance between positions | Done |
| 18 | [Enrich aircraft type from the OpenSky aircraft database](https://github.com/Ramco28/Junior_IS/issues/18) | 10/01 | 2h | Type and manufacturer by icao24, so rule thresholds can differ by aircraft type | Done |
| 6 | [Rule-based anomaly checks (speed, duplicate ICAO, altitude/speed change, route deviation)](https://github.com/Ramco28/Junior_IS/issues/6) | 10/01 | 14h | Four configurable-threshold checks | Done |
| 7 | [Label historical dataset via weak labeling](https://github.com/Ramco28/Junior_IS/issues/7) | 10/01 | 4h | Uses rule-based checks as initial labels | Done |
| 8 | [Train baseline ML model](https://github.com/Ramco28/Junior_IS/issues/8) | 11/01 | 8h | e.g. isolation forest or autoencoder | Done |
| 9 | [Evaluate ML model vs. rule-based detector](https://github.com/Ramco28/Junior_IS/issues/9) | 11/08 | 4h | Agreement/disagreement statistics on held-out data | Done |
| 16 | [Summary report generation script](https://github.com/Ramco28/Junior_IS/issues/16) | 11/12 | 3h | Precision, recall, agreement metrics | Done |

## Stretch Goals

These features are not built this semester. They continue in my Senior I.S. next year.

| # | Feature | Due Date | Estimate | Notes | Status |
|---|---------|----------|----------|-------|--------|
| 10 | [Build backend service running both detectors](https://github.com/Ramco28/Junior_IS/issues/10) | Senior I.S. | 8h | Shared output format for live data | Planned |
| 11 | [Live map view](https://github.com/Ramco28/Junior_IS/issues/11) | Senior I.S. | 6h | Displays currently tracked aircraft | Planned |
| 12 | [Highlight flagged aircraft on map](https://github.com/Ramco28/Junior_IS/issues/12) | Senior I.S. | 2h | Distinct styling from unflagged aircraft | Planned |
| 13 | [Side panel of flagged aircraft](https://github.com/Ramco28/Junior_IS/issues/13) | Senior I.S. | 4h | Anomaly type, score, detector source | Planned |
| 14 | [Click handling + trajectory detail view](https://github.com/Ramco28/Junior_IS/issues/14) | Senior I.S. | 4h | Centers map, opens detail on selection | Planned |
| 15 | [Filter and toggle controls](https://github.com/Ramco28/Junior_IS/issues/15) | Senior I.S. | 3h | Filter by detector; toggle anomaly categories | Planned |
| 17 | [Historical playback, sequence-based ML model, exportable anomaly report](https://github.com/Ramco28/Junior_IS/issues/17) | Senior I.S. | 18h | Combined stretch bucket | Planned |
| | Evaluation on injected anomalies with known ground truth | Senior I.S. | TBD | Synthetic anomalies of each category (including duplicate ICAO), so precision, recall and false alarm rate can be measured against a true answer | Planned |

## Results (Junior I.S.)

I compared an isolation forest with my rule-based checks, flight by flight, on three held-out
test days (October 4, 6 and 7, 2026) of ADS-B data over Ohio. There is no ground truth, so the
numbers below measure how much the forest agrees with the weak labels from my rules, not how
often it is right.

| | Forest flagged | Forest not flagged |
|---|---|---|
| Rules anomalous | 10 (both) | 21 (rules only) |
| Rules normal | 44 (forest only) | 7,159 (neither) |

Against the weak labels, the forest has a precision of 18.5% and a recall of 32.3%, and it flags
0.6% of the flights my rules call normal.

I inspected every disagreement by hand. Of the 44 flights only the forest flagged, 2 were genuine
anomalies, 5 equipment faults and 37 false alarms; of the 21 flights only the rules flagged, 0 were
genuine anomalies, 13 equipment faults and 8 false alarms.

`python scripts/summary_report.py` regenerates the numbers, the LaTeX tables and the figure from the
saved results. They are written to `data/report/` (`metrics.json`, `table_*.tex`,
`score_distribution.png` and `.pdf`).

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
| 18 | `python scripts/download_aircraft_db.py` | Downloads the OpenSky aircraft database once (about 110 MB), used to tell small airplanes, helicopters and jets apart |
| 6 | `python scripts/run_rules.py --live` | Runs the rule-based checks (speed, duplicate ICAO, altitude and speed change, route deviation) on a feature file and saves the flags |
| 7 | `python scripts/label_dataset.py` | Labels every historical report and trajectory with the rule-based checks (weak labels) and assigns whole days to train, validation and test |
| 8 | `python scripts/train_model.py` | Trains an isolation forest on the training days, sets its threshold and tree count on the validation days, and flags reports on every day |
| 9 | `python scripts/compare_detectors.py` | Compares the forest with the rules per flight on the test days and writes the disagreements to check by hand |
| 16 | `python scripts/summary_report.py` | Collects the final numbers in `metrics.json` and writes the LaTeX tables and the score figure for the paper |

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
