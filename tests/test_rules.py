"""Tests for feature 6. Run them from the repo root with: python -m pytest"""
import pandas as pd

from adsb.features import add_features
from adsb.normalize import normalize_live
from adsb.rules import find_artifact_batches, mark_artifacts

SPEED = 200.0                    # every fake aircraft flies north at 200 m/s
DEG_PER_S = SPEED / 111_195      # degrees of latitude covered in one second


def fake_snapshot(snap_time, n_aircraft=12, lagging=0, lag_s=10):
    """One fake /states/all snapshot.

    Each aircraft says its position is from snap_time. For the first `lagging`
    aircraft that is not true: the position is where the aircraft was lag_s
    seconds earlier, but the timestamp still says snap_time.
    """
    states = []
    for i in range(n_aircraft):
        true_time = snap_time - (lag_s if i < lagging else 0)
        lat = 39.0 + true_time * DEG_PER_S
        lon = -84.0 + 0.1 * i  # the aircraft fly side by side
        states.append([f"aaa{i:03d}", f"FAKE{i}", "United States", snap_time, snap_time, lon, lat,
                       10000.0, False, SPEED, 0.0, 0.0, None, 10100.0, "1200", False, 0, 0])
    return {"time": snap_time, "states": states}


def features_for(snapshots):
    normalized = pd.concat([normalize_live(s) for s in snapshots], ignore_index=True)
    return add_features(normalized)


def test_lagging_snapshot_is_marked_as_artifact():
    # snapshots every 30 s; in the one at 1060, 10 of 12 aircraft lag by 10 s
    snaps = [fake_snapshot(1000), fake_snapshot(1030),
             fake_snapshot(1060, lagging=10), fake_snapshot(1090), fake_snapshot(1120)]
    df = features_for(snaps)
    bad = find_artifact_batches(df)
    # 1060 is too slow (20 s of flying stamped as 30 s), and 1090 is too fast
    # because it makes up the missing distance: the same pattern as my real data
    assert list(bad["batch_time"]) == [1060, 1090]
    assert bad["median_mismatch_mps"].iloc[0] < -50
    assert bad["median_mismatch_mps"].iloc[1] > 50

    marked = mark_artifacts(df)
    assert set(marked.loc[marked["batch_artifact"], "batch_time"]) == {1060, 1090}


def test_one_odd_aircraft_does_not_mark_the_batch():
    # only 1 of 12 aircraft lags: that is an aircraft problem, not a batch problem
    snaps = [fake_snapshot(1000), fake_snapshot(1030),
             fake_snapshot(1060, lagging=1), fake_snapshot(1090)]
    df = features_for(snaps)
    assert find_artifact_batches(df).empty
    # the odd aircraft itself still has a large mismatch for the detectors to find
    assert df["speed_mismatch_mps"].abs().max() > 50


def test_small_batch_is_not_judged():
    # 3 aircraft, all lagging: too few to say the batch is the problem
    snaps = [fake_snapshot(1000, n_aircraft=3), fake_snapshot(1030, n_aircraft=3),
             fake_snapshot(1060, n_aircraft=3, lagging=3), fake_snapshot(1090, n_aircraft=3)]
    df = features_for(snaps)
    assert find_artifact_batches(df).empty
    # with the minimum lowered, the same batches are found
    assert list(find_artifact_batches(df, min_aircraft=3)["batch_time"]) == [1060, 1090]
