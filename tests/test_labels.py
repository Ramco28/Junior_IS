"""Tests for feature 7. Run them from the repo root with: python -m pytest"""
import pandas as pd

from adsb.labels import add_split, label_rows, label_trajectories, label_window


def features(trajectory_id="abc123_0", n=4, artifact_rows=(), source="database"):
    # a tiny feature table: n reports, 10 s apart
    return pd.DataFrame({
        "time": [1000 + 10 * i for i in range(n)],
        "trajectory_id": trajectory_id,
        "icao24": trajectory_id.split("_")[0],
        "aircraft_class": "jet",
        "aircraft_class_source": source,
        "batch_artifact": [i in artifact_rows for i in range(n)],
    })


def flags(rows=()):
    # rows of (trajectory_id, time, check, is_artifact)
    return pd.DataFrame(list(rows), columns=["trajectory_id", "time", "check", "is_artifact"])


def trajectories(feat, fl, window="20260915T1200"):
    return label_trajectories(add_split(label_window(feat, fl, window)))


def test_artifact_rows_become_uncertain_not_normal():
    # report 1 is in an unreliable batch, and its speed_mismatch flag was blamed on the data
    out = label_rows(features(artifact_rows=[1]), flags([("abc123_0", 1010, "speed_mismatch", True)]))
    assert list(out["label"]) == ["normal", "uncertain", "normal", "normal"]
    assert list(out["checks"]) == ["", "", "", ""]  # artifact flags are not listed


def test_real_flag_wins_over_uncertain():
    # speed_reported does not depend on the batch, so it counts even in an unreliable batch
    out = label_rows(features(artifact_rows=[1]), flags([("abc123_0", 1010, "speed_reported", False)]))
    assert list(out["label"]) == ["normal", "anomaly", "normal", "normal"]


def test_trajectory_with_one_flagged_row_is_anomalous():
    t = trajectories(features(), flags([("abc123_0", 1020, "climb_rate", False)]))
    assert len(t) == 1
    assert bool(t["anomalous"].iloc[0])
    assert t["flagged_rows"].iloc[0] == 1
    assert t["checks"].iloc[0] == "climb_rate"
    # a trajectory with no flag is not anomalous
    clean = trajectories(features(), flags())
    assert not bool(clean["anomalous"].iloc[0])
    assert clean["flagged_rows"].iloc[0] == 0


def test_route_deviation_is_a_trajectory_label_not_a_row_label():
    # the route check flags every report of the trajectory
    fl = flags([("abc123_0", 1000 + 10 * i, "route_deviation", False) for i in range(4)])
    rows = label_rows(features(), fl)
    assert set(rows["label"]) == {"normal"}             # no single report is anomalous
    assert set(rows["checks"]) == {"route_deviation"}   # but the check is still recorded
    t = trajectories(features(), fl)
    assert bool(t["anomalous"].iloc[0]) and bool(t["route_deviation"].iloc[0])
    assert t["flagged_rows"].iloc[0] == 0
    assert not bool(t["duplicate_icao"].iloc[0])


def test_several_checks_on_one_row_are_all_kept():
    fl = flags([("abc123_0", 1010, "speed_mismatch", False), ("abc123_0", 1010, "acceleration", False),
                ("abc123_0", 1010, "route_deviation", False)])
    rows = label_rows(features(), fl)
    assert rows["checks"].iloc[1] == "acceleration,route_deviation,speed_mismatch"
    assert rows["label"].iloc[1] == "anomaly"


def test_class_mismatch_does_not_count_as_an_anomaly():
    t = trajectories(features(source="mismatch"), flags())
    assert bool(t["class_mismatch"].iloc[0])
    assert not bool(t["anomalous"].iloc[0])
    assert t["flagged_rows"].iloc[0] == 0


def test_same_aircraft_on_two_days_gets_two_different_ids():
    a = label_window(features(), flags(), "20260915T1200")
    b = label_window(features(), flags(), "20261006T1200")
    assert set(a["trajectory_id"]) == {"20260915T1200_abc123_0"}
    assert set(b["trajectory_id"]) == {"20261006T1200_abc123_0"}
    assert set(a["day"]) == {"2026-09-15"}


def test_split_never_puts_a_trajectory_or_a_day_in_two_sets():
    windows = ["20260915T0200", "20260915T1200", "20260930T1200", "20261003T1700", "20261006T2200"]
    rows = add_split(pd.concat([label_window(features(), flags(), w) for w in windows]))
    assert (rows.groupby("trajectory_id")["split"].nunique() == 1).all()
    assert (rows.groupby("day")["split"].nunique() == 1).all()
    by_day = rows.drop_duplicates("day").set_index("day")["split"].to_dict()
    assert by_day == {"2026-09-15": "train", "2026-09-30": "validation",
                      "2026-10-03": "test", "2026-10-06": "test"}
