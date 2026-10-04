"""Getting past flights from the OpenSky Trino database (feature 3)."""
from datetime import datetime, timezone

import pandas as pd

from .config import TRINO_CATALOG, TRINO_HOST, TRINO_PORT, TRINO_SCHEMA, TRINO_USER

# the columns of state_vectors_data4 that I need
# The table has one row per aircraft per second, and it repeats the last known
# position until a new one arrives. lastposupdate is the time the position was
# really received, so I need it to know which rows are a new position and
# which rows are just a repeat of an old one.
COLUMNS = ["time", "icao24", "callsign", "lat", "lon", "velocity", "heading",
           "vertrate", "baroaltitude", "geoaltitude", "onground", "squawk",
           "lastposupdate"]


def connect():
    """Connect to OpenSky's Trino server.

    The first query opens a browser so I can log in (same as my Trino CLI).
    After that the login is cached, so I don't have to do it every time.
    """
    import trino  # only imported here so the live scripts work without trino installed
    return trino.dbapi.connect(
        host=TRINO_HOST, port=TRINO_PORT, user=TRINO_USER,
        catalog=TRINO_CATALOG, schema=TRINO_SCHEMA, http_scheme="https",
        auth=trino.auth.OAuth2Authentication(),
    )


def build_query(start: datetime, end: datetime, bbox, sample_s: int = 10) -> str:
    """Build the SQL for every state vector between start and end inside bbox.

    The table is split into one partition per hour, and the hour column is the
    Unix time that hour starts. Filtering on hour lets Trino skip all the other
    partitions, which matters because OpenSky stops any query after 30 minutes
    or 100 GB scanned. "time % sample_s = 0" keeps one report every sample_s
    seconds per aircraft, since the table has about one per second.
    """
    t0, t1 = int(start.timestamp()), int(end.timestamp())
    # h0 is the start rounded down to its hour, h1 is the end rounded up to the
    # next hour. For 14:00 to 15:00 this only reads the 14:00 partition.
    h0 = t0 - t0 % 3600
    h1 = t1 if t1 % 3600 == 0 else t1 - t1 % 3600 + 3600
    lat_min, lat_max, lon_min, lon_max = bbox
    return f"""
        SELECT {", ".join(COLUMNS)}
        FROM state_vectors_data4
        WHERE hour >= {h0} AND hour < {h1}
          AND time >= {t0} AND time < {t1}
          AND time % {sample_s} = 0
          AND lat BETWEEN {lat_min} AND {lat_max}
          AND lon BETWEEN {lon_min} AND {lon_max}
          AND lat IS NOT NULL AND lon IS NOT NULL
    """


def fetch_state_vectors(start: datetime, end: datetime, bbox, sample_s: int = 10) -> pd.DataFrame:
    # run the query and put the rows in a DataFrame
    conn = connect()
    cur = conn.cursor()
    cur.execute(build_query(start, end, bbox, sample_s))
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=COLUMNS)


def split_trajectories(df: pd.DataFrame, max_gap_s: int = 900) -> pd.DataFrame:
    """Give every row a trajectory_id so I can look at each flight on its own.

    I group the rows by aircraft (icao24) and sort them by time. If an aircraft
    is silent for more than max_gap_s seconds (15 min by default), I count what
    comes next as a new flight, for example it landed and took off again.
    The ids look like "a1b2c3_0", "a1b2c3_1".
    """
    if df.empty:
        return df.assign(trajectory_id=pd.Series(dtype=str))
    df = df.sort_values(["icao24", "time"]).reset_index(drop=True)
    df["callsign"] = df["callsign"].fillna("").str.strip()
    gap = df.groupby("icao24")["time"].diff()  # seconds since this aircraft's last report (NaN for its first)
    new_segment = gap.isna() | (gap > max_gap_s)  # True where a new flight starts
    seg_num = new_segment.astype(int).groupby(df["icao24"]).cumsum() - 1  # flight number per aircraft, from 0
    df["trajectory_id"] = df["icao24"] + "_" + seg_num.astype(str)
    return df


def parse_utc(text: str) -> datetime:
    """Turn "YYYY-MM-DD HH:MM" into a UTC datetime."""
    return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
