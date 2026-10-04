"""Derived per-aircraft features (feature 5).

For each trajectory I compare every report with the one before it. That gives
me what the aircraft DID between two reports (distance, climb, turn), which I
can compare with what the aircraft SAYS it is doing (reported speed and
vertical rate). A spoofed or glitched position shows up as a mismatch.
"""
import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000  # mean radius of the Earth in meters

# the columns add_features() adds to a normalized table
FEATURE_COLUMNS = [
    "dt_s",                # seconds since the previous report of this trajectory
    "distance_m",          # distance from the previous position
    "implied_speed_mps",   # distance_m / dt_s
    "speed_mismatch_mps",  # implied speed minus reported velocity
    "alt_rate_mps",        # change in barometric altitude / dt_s
    "vrate_mismatch_mps",  # alt_rate minus reported vertical rate
    "accel_mps2",          # change in reported velocity / dt_s
    "turn_rate_dps",       # change in heading (degrees) / dt_s
]


def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters between two points on the Earth, given in degrees.

    The haversine formula gives the shortest distance over the surface of a
    sphere. It works on single numbers and on whole pandas columns.
    """
    # the formula needs radians, not degrees
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def wrap_degrees(delta):
    """Bring a heading change into the range -180 to 180 degrees.

    Headings wrap around at 360. Going from 350 to 10 is a right turn of 20
    degrees, but 10 - 350 = -340. Adding 180, taking the remainder of the
    division by 360 and subtracting 180 again gives +20.
    """
    return (delta + 180) % 360 - 180


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the FEATURE_COLUMNS to a normalized table (see normalize.py).

    Everything is computed inside one trajectory, so the last report of one
    flight is never compared with the first report of another. The first row
    of each trajectory has no previous report, so its features are missing.
    """
    df = df.sort_values(["trajectory_id", "time"]).reset_index(drop=True)
    g = df.groupby("trajectory_id")

    # diff() is "this row minus the previous row", restarted for each trajectory
    dt = g["time"].diff().astype("float64")
    # a time gap of 0 would mean dividing by zero and getting infinity, so I
    # replace it with missing: every rate below is then missing too
    dt = dt.where(dt > 0)
    df["dt_s"] = dt

    # shift() gives the previous row's value, so I have both ends of each step
    df["distance_m"] = haversine_m(g["lat"].shift(), g["lon"].shift(), df["lat"], df["lon"])
    df["implied_speed_mps"] = df["distance_m"] / dt
    df["speed_mismatch_mps"] = df["implied_speed_mps"] - df["velocity_mps"]

    df["alt_rate_mps"] = g["baro_altitude_m"].diff() / dt
    df["vrate_mismatch_mps"] = df["alt_rate_mps"] - df["vertical_rate_mps"]

    df["accel_mps2"] = g["velocity_mps"].diff() / dt
    df["turn_rate_dps"] = wrap_degrees(g["heading_deg"].diff()) / dt
    return df
