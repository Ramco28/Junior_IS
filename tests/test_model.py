"""Tests for feature 8. Run them from the repo root with: python -m pytest"""
import numpy as np
import pandas as pd
import pytest

from adsb.model import (FEATURES, add_repeated, budget_threshold, choose_n_estimators,
                        flight_scores, forest_flags, model_rows, score_rows, train_forest)
from adsb.rules import FLAG_COLUMNS


def reports(n_flights=50, n=10, split="train", seed=0):
    """Fake labeled reports: many normal flights with small random noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_flights):
        for i in range(n):
            rows.append({
                "trajectory_id": f"{split}_{f:03d}", "time": 1000 + 10 * i, "batch_time": 1000 + 10 * i,
                "icao24": f"a{f:05d}", "aircraft_class": "jet", "split": split, "day": "2026-09-15",
                "on_ground": False, "batch_artifact": False, "dt_s": 10.0,
                "velocity_mps": 230.0 + rng.normal(0, 5), "heading_deg": 90.0 + rng.normal(0, 2),
                "speed_mismatch_mps": rng.normal(0, 3), "vrate_mismatch_mps": rng.normal(0, 0.5),
                "alt_rate_mps": rng.normal(0, 1), "accel_mps2": rng.normal(0, 0.1),
                "turn_rate_dps": rng.normal(0, 0.2),
            })
    return pd.DataFrame(rows)


def test_obvious_outlier_scores_higher_than_normal_reports():
    rows, _ = model_rows(reports())
    forest = train_forest(rows, n_estimators=100)
    odd = rows.iloc[[5]].copy()
    odd["speed_mismatch_mps"] = 200.0   # positions move 200 m/s faster than the reported speed
    odd["alt_rate_mps"] = 80.0          # and the altitude jumps by 800 m in 10 s
    normal_scores = score_rows(forest, rows)
    assert score_rows(forest, odd).iloc[0] > normal_scores.max()


def test_flags_have_the_shared_columns():
    rows, _ = model_rows(reports())
    forest = train_forest(rows, n_estimators=50)
    scores = score_rows(forest, rows)
    flags = forest_flags(rows, scores, threshold=float(scores.quantile(0.99)))
    assert list(flags.columns) == FLAG_COLUMNS
    assert set(flags["detector"]) == {"iforest"} and set(flags["check"]) == {"iforest_score"}
    assert not flags["is_artifact"].any()
    assert (flags["value"] >= flags["threshold"]).all()


def test_training_never_sees_validation_or_test_days():
    mixed, _ = model_rows(pd.concat([reports(split="train"), reports(split="validation", seed=1)]))
    with pytest.raises(ValueError):
        train_forest(mixed, n_estimators=10)
    with pytest.raises(ValueError):
        train_forest(model_rows(reports(split="test"))[0], n_estimators=10)
    # training reports alone are accepted
    train_forest(mixed[mixed["split"] == "train"], n_estimators=10)


def test_reports_with_a_missing_input_are_left_out_and_counted():
    df = reports(n_flights=2, n=5)
    df.loc[3, "alt_rate_mps"] = np.nan
    df.loc[7, "turn_rate_dps"] = np.nan
    rows, missing = model_rows(df)
    assert rows[FEATURES].notna().all().all()
    assert missing.loc["alt_rate_mps", "train"] == 1
    assert missing.loc["turn_rate_dps", "train"] == 1
    assert missing.loc["any feature", "train"] == 2


def test_same_filter_as_the_rules():
    df = reports(n_flights=1, n=6)
    df.loc[2, "on_ground"] = True      # on the ground: this step and the next one are not judged
    df.loc[4, "dt_s"] = 90.0           # gap too long
    df.loc[5, "batch_artifact"] = True # unreliable batch
    rows, _ = model_rows(df)
    # report 0 has no previous report; 2 and 3 touch the ground; 4 and 5 fail the other filters
    assert list(rows["time"]) == [1010]


def test_repeated_is_computed_per_flight():
    df = reports(n_flights=2, n=3)
    df["velocity_mps"] = 230.0
    df["heading_deg"] = 90.0           # every report equals the previous one
    out = add_repeated(df)
    first = out.groupby("trajectory_id").head(1)
    assert (first["repeated"] == 0).all()                 # never "repeated" at the start of a flight
    assert out["repeated"].sum() == 4                      # 2 flights x 2 later reports


def test_budget_threshold_flags_exactly_the_budget():
    per_flight = pd.Series([0.9, 0.4, 0.7, 0.5, 0.8], index=list("abcde"))
    threshold = budget_threshold(per_flight, 2)
    assert set(per_flight.index[per_flight >= threshold]) == {"a", "e"}


def test_flight_score_is_its_highest_report():
    rows = pd.DataFrame({"trajectory_id": ["x", "x", "y"]})
    assert flight_scores(rows, pd.Series([0.3, 0.6, 0.5])).to_dict() == {"x": 0.6, "y": 0.5}


def test_tree_count_is_chosen_by_stability():
    a, b = set(range(100)), set(range(97)) | {200, 201, 202}   # 97% in common
    c = set(range(60)) | set(range(300, 340))                  # 60% in common with b
    assert choose_n_estimators({100: c, 200: b, 400: a})[0] == 200  # 100 vs 200 too different
    assert choose_n_estimators({100: a, 200: b, 400: a})[0] == 100
    assert choose_n_estimators({100: c, 200: a, 400: c})[0] == 400  # never 95%: the largest
