"""Tests for feature 5. Run them from the repo root with: python -m pytest"""
import numpy as np
import pandas as pd
import pytest

from adsb.features import FEATURE_COLUMNS, add_features, haversine_m, wrap_degrees


def track(times, lats, headings=None, trajectory_id="abc123_0"):
    # a small normalized table: one aircraft flying north along longitude -82
    n = len(times)
    return pd.DataFrame({
        "time": times, "trajectory_id": [trajectory_id] * n,
        "lat": lats, "lon": [-82.0] * n,
        "baro_altitude_m": [3000.0] * n, "velocity_mps": [100.0] * n,
        "heading_deg": headings or [0.0] * n, "vertical_rate_mps": [0.0] * n,
    })


def test_haversine_one_degree_of_latitude():
    # one degree of latitude is about 111.2 km everywhere on the Earth
    assert haversine_m(40.0, -82.0, 41.0, -82.0) == pytest.approx(111_195, rel=0.001)


def test_heading_wraps_around_north():
    assert wrap_degrees(10 - 350) == 20    # right turn through north, not -340
    assert wrap_degrees(350 - 10) == -20   # left turn through north, not +340
    df = add_features(track([0, 10], [40.0, 40.01], headings=[350.0, 10.0]))
    assert df["turn_rate_dps"].iloc[1] == pytest.approx(2.0)  # 20 degrees in 10 s


def test_dt_of_zero_gives_missing_not_infinity():
    df = add_features(track([0, 0, 10], [40.0, 40.01, 40.02]))
    row = df.iloc[1]  # same time as the row before it
    assert pd.isna(row["dt_s"])
    assert pd.isna(row["implied_speed_mps"])
    # no infinity anywhere in the feature columns
    assert not np.isinf(df[FEATURE_COLUMNS].to_numpy(dtype="float64")).any()


def test_first_row_of_each_trajectory_is_missing():
    a = track([0, 10], [40.0, 40.01], trajectory_id="aaa_0")
    b = track([5, 15], [41.0, 41.01], trajectory_id="bbb_0")
    df = add_features(pd.concat([a, b]))
    first_rows = df.groupby("trajectory_id").head(1)
    assert first_rows[FEATURE_COLUMNS].isna().all().all()
    # and the second trajectory is not compared with the first one
    assert df["distance_m"].max() < 2000


def test_implied_speed_matches_distance_over_time():
    # 0.009 degrees of latitude is about 1000.8 m; in 10 s that is about 100 m/s
    df = add_features(track([0, 10], [40.0, 40.009]))
    assert df["implied_speed_mps"].iloc[1] == pytest.approx(100.08, rel=0.001)
    assert df["speed_mismatch_mps"].iloc[1] == pytest.approx(0.08, abs=0.1)
