"""Tests for feature 4. Run them from the repo root with: python -m pytest"""
import pandas as pd

from adsb.normalize import NORMALIZED_COLUMNS, normalize_historical, normalize_live


def live_row(icao24="abc123", t=1000, lat=40.0, lon=-82.0, velocity=100.0):
    # one aircraft in the order of the /states/all fields (see STATE_FIELDS in config.py)
    return [icao24, "TEST1   ", "United States", t, t, lon, lat, 3000.0, False,
            velocity, 90.0, 0.0, None, 3100.0, "1200", False, 0, 2]


def historical_frame():
    # two rows the way my Trino query returns them
    return pd.DataFrame({
        "time": [1000, 1010], "icao24": ["abc123", "abc123"], "callsign": ["TEST1   ", "TEST1   "],
        "lat": [40.0, 40.01], "lon": [-82.0, -82.0], "velocity": [100.0, 100.0],
        "heading": [0.0, 0.0], "vertrate": [0.0, 0.0], "baroaltitude": [3000.0, 3000.0],
        "geoaltitude": [3100.0, 3100.0], "onground": [False, False], "squawk": ["1200", "1200"],
        "lastposupdate": [999.6, 1009.7],
    })


def test_live_and_historical_have_identical_columns():
    live = normalize_live({"time": 1012, "states": [live_row(t=1000), live_row(t=1010)]})
    hist = normalize_historical(historical_frame())
    assert list(live.columns) == NORMALIZED_COLUMNS
    assert list(hist.columns) == NORMALIZED_COLUMNS
    # same types too, so later code never needs to know the source
    assert list(live.dtypes) == list(hist.dtypes)


def test_implausible_value_is_kept():
    # 2000 m/s is about six times the speed of sound: impossible for an
    # aircraft, and exactly what a detector must see
    df = normalize_live({"time": 1002, "states": [live_row(velocity=2000.0)]})
    assert len(df) == 1
    assert df["velocity_mps"].iloc[0] == 2000.0


def test_rows_without_position_and_duplicates_are_dropped():
    states = [
        live_row(t=1000),
        live_row(t=1000),                 # same aircraft, same position time: a repeat
        live_row(t=1010, lat=None),       # no position
        live_row(icao24="ABC999", t=1000),
    ]
    df = normalize_live({"time": 1012, "states": states})
    assert len(df) == 2
    assert set(df["icao24"]) == {"abc123", "abc999"}  # lowercase


def test_historical_uses_position_time():
    df = normalize_historical(historical_frame())
    assert list(df["time"]) == [1000, 1010]  # lastposupdate 999.6 and 1009.7, rounded
    assert df["category"].isna().all()       # the historical table has no category
    assert df["callsign"].iloc[0] == "TEST1"


def test_batch_time_is_the_delivery_time():
    # live: the snapshot time, not the position time
    live = normalize_live({"time": 1003, "states": [live_row(t=1000)]})
    assert list(live["time"]) == [1000]
    assert list(live["batch_time"]) == [1003]
    # historical: the table's row time from before lastposupdate replaced it
    hist = normalize_historical(historical_frame())
    assert list(hist["batch_time"]) == [1000, 1010]
