"""Putting live and historical data in one common table format (feature 4).

The live API and the Trino database describe the same thing (state vectors)
but with different column names. Here I turn both into the same table, with
the same columns and units, so my detectors can run on either source.
"""
import gzip
import json

import pandas as pd

from .config import SNAPSHOT_DIR
from .historical import split_trajectories
from .live import states_to_frame

# The one format every later step works on. Units stay the way OpenSky gives
# them (meters, meters per second, degrees), I do not convert anything.
NORMALIZED_COLUMNS = [
    "time",               # Unix seconds when the POSITION was received (int)
    "icao24",             # aircraft address, lowercase hex
    "callsign",           # stripped, missing if empty
    "lat",                # degrees
    "lon",                # degrees
    "baro_altitude_m",    # altitude from air pressure
    "geo_altitude_m",     # altitude from GPS
    "velocity_mps",       # ground speed
    "heading_deg",        # direction of travel, 0 = north, clockwise
    "vertical_rate_mps",  # positive = climbing
    "on_ground",          # True or False
    "squawk",             # transponder code, kept as text ("7700", "0421")
    "category",           # aircraft type code (live only, missing in historical)
    "source",             # "live" or "historical"
    "trajectory_id",      # icao24 + "_" + flight number
]

# columns that must be decimal numbers in both sources
FLOAT_COLUMNS = ["lat", "lon", "baro_altitude_m", "geo_altitude_m",
                 "velocity_mps", "heading_deg", "vertical_rate_mps"]

# live API name -> normalized name
LIVE_RENAME = {
    "time_position": "time",
    "latitude": "lat",
    "longitude": "lon",
    "baro_altitude": "baro_altitude_m",
    "geo_altitude": "geo_altitude_m",
    "velocity": "velocity_mps",
    "true_track": "heading_deg",
    "vertical_rate": "vertical_rate_mps",
}

# Trino column name -> normalized name
HISTORICAL_RENAME = {
    "baroaltitude": "baro_altitude_m",
    "geoaltitude": "geo_altitude_m",
    "velocity": "velocity_mps",
    "heading": "heading_deg",
    "vertrate": "vertical_rate_mps",
    "onground": "on_ground",
}


def _finish(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """The cleaning steps that are the same for both sources.

    I only remove rows I cannot use at all: rows with no position, and rows
    that repeat a position I already have. I NEVER remove or change a value
    because it looks impossible (a speed of 2000 m/s, a jump of 50 km, a
    sudden altitude change). Those are exactly the anomalies my detectors
    have to find, so they must survive this step.
    """
    df = df.copy()  # so I never change the table that was passed in
    df["source"] = source
    if "category" not in df.columns:
        df["category"] = None  # the historical table has no category

    # 1. a row without a position or a position time is unusable
    df = df.dropna(subset=["time", "lat", "lon"])

    # 2. same types in both sources
    df["time"] = df["time"].round().astype("int64")
    df["icao24"] = df["icao24"].str.lower()
    df[FLOAT_COLUMNS] = df[FLOAT_COLUMNS].astype("float64")
    df["on_ground"] = df["on_ground"].astype(bool)
    df["category"] = df["category"].astype("Int64")  # whole numbers that can be missing

    # 3. the same aircraft with the same position time is the same report seen
    #    twice (the position did not update in between), so I keep the first
    df = df.drop_duplicates(subset=["icao24", "time"], keep="first")

    # 4. same 15 minute gap rule for both sources (it also sorts by aircraft and time)
    df = split_trajectories(df)

    # 5. an empty callsign means "not known", so I store it as missing
    df["callsign"] = df["callsign"].where(df["callsign"] != "")

    return df[NORMALIZED_COLUMNS].reset_index(drop=True)


def normalize_live(snapshot: dict) -> pd.DataFrame:
    """Normalize one raw /states/all snapshot (the dict fetch_states returns).

    For the time I use time_position, the moment the position was received,
    and not the snapshot time. An aircraft can appear in a snapshot with a
    position that is several minutes old.
    """
    df = states_to_frame(snapshot).rename(columns=LIVE_RENAME)
    return _finish(df, "live")


def load_snapshots(folder=SNAPSHOT_DIR) -> pd.DataFrame:
    """Load every stored snapshot in a folder and normalize them as one table."""
    rows = []
    # the file names are UTC times, so sorted() gives them in time order
    for path in sorted(folder.glob("states_*.json.gz")):
        with gzip.open(path, "rt") as f:
            rows.extend(json.load(f)["states"])
    # all the aircraft rows together look like one very large snapshot
    return normalize_live({"states": rows})


def normalize_historical(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize rows from state_vectors_data4 (straight from Trino or from my Parquet file)."""
    df = df.sort_values("time")  # so "keep the first" below means the earliest row
    if "lastposupdate" in df.columns:
        # the table repeats the last position every second, so the row time is
        # not the position time. lastposupdate is, and it becomes my time.
        df = df.drop(columns="time").rename(columns={"lastposupdate": "time"})
    df = df.drop(columns="trajectory_id", errors="ignore")  # recomputed after cleaning
    return _finish(df.rename(columns=HISTORICAL_RENAME), "historical")
