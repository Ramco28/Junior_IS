# Junior_IS
### How effectively can machine learning models detect anomalous or potentially unsafe aircraft behavior in ADS-B flight data compared to rule-based approaches?

## Feature Calendar

This project compares machine learning and rule-based approaches to detecting anomalous aircraft behavior in ADS-B flight data. The table below tracks the features I'm building, with due dates and time estimates set through Planning Poker. Each row links to its GitHub issue. The stretch goals listed at the end will be tackled if time permits.

| # | Feature | Due Date | Estimate | Notes |
|---|---------|----------|----------|-------|
| 1 | [Authenticate & retrieve single OpenSky snapshot](#1) | 09/24 | 2h | Foundation for all data ingestion |
| 2 | [Poll live API and store snapshots](#2) | 09/24 | 3h | Adds scheduling + persistence |
| 3 | [Query historical DB, save labeled trajectories](#3) | 09/24 | 4h | Source data for ML training/testing |
| 4 | [Parse raw state vector into normalized record](#4) | 09/24 | 2h | ICAO, callsign, lat/lon, altitude, speed, timestamp |
| 5 | [Compute derived per-aircraft features](#5) | 09/24 | 5h | Altitude/speed change rate, distance between positions |
| 6 | [Rule-based anomaly checks (speed, duplicate ICAO, altitude/speed change, route deviation)](#6) | 10/01 | 14h | Four configurable-threshold checks |
| 7 | [Label historical dataset via weak labeling](#7) | 10/01 | 4h | Uses rule-based checks as initial labels |
| 8 | [Train baseline ML model](#8) | 10/22 | 8h | e.g. isolation forest or autoencoder |
| 9 | [Evaluate ML model vs. rule-based detector](#9) | 10/22 | 4h | Agreement/disagreement statistics on held-out data |
| 10 | [Build backend service running both detectors](#10) | 10/22 | 8h | Shared output format for live data |
| 11 | [Live map view](#11) | 10/29 | 6h | Displays currently tracked aircraft |
| 12 | [Highlight flagged aircraft on map](#12) | 10/29 | 2h | Distinct styling from unflagged aircraft |
| 13 | [Side panel of flagged aircraft](#13) | 10/29 | 4h | Anomaly type, score, detector source |
| 14 | [Click handling + trajectory detail view](#14) | 10/29 | 4h | Centers map, opens detail on selection |
| 15 | [Filter and toggle controls](#15) | 10/29 | 3h | Filter by detector; toggle anomaly categories |
| 16 | [Summary report generation script](#16) | 11/12 | 3h | Precision, recall, agreement metrics |

## Stretch Goals

| # | Feature | Due Date | Estimate | Notes |
|---|---------|----------|----------|-------|
| 17 | [Historical playback, sequence-based ML model, exportable anomaly report](#17) | if time permits | 18h | Combined stretch bucket |
