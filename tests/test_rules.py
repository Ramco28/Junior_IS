"""Tests for feature 6. Run them from the repo root with: python -m pytest"""
import numpy as np
import pandas as pd

from adsb.features import add_features
from adsb.normalize import normalize_live
from adsb.rules import (FLAG_COLUMNS, check_changes, check_duplicate_icao,
                        check_route_deviation, check_speed, find_artifact_batches,
                        find_class_mismatches, mark_artifacts, run_all_checks)

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


# ---------------------------------------------------------------------------
# The four aircraft checks. Each one gets a real anomaly that must be flagged
# and a normal flight that must not be.
# ---------------------------------------------------------------------------

M_PER_DEG = 111_195  # meters in one degree of latitude

# the columns a table has BEFORE add_features (what flight() starts from)
RAW_COLUMNS = ["time", "batch_time", "icao24", "callsign", "trajectory_id", "lat", "lon",
               "baro_altitude_m", "velocity_mps", "heading_deg", "vertical_rate_mps",
               "on_ground", "aircraft_class", "aircraft_class_source"]


def recompute(df):
    # after changing positions or speeds by hand, the features must be computed again
    return add_features(df[RAW_COLUMNS])


def flight(icao24="abc123", cls="jet", source="database", n=12, step_s=10,
           speed=250.0, reported_speed=None, altitude=10000.0, climb=0.0,
           start_lat=40.0, lon=-82.0, start_time=1000):
    """A fake aircraft flying north at `speed` m/s, as a feature table.

    reported_speed is what the aircraft SAYS its speed is (default: the truth).
    climb is the real climb rate in m/s.
    """
    times = [start_time + i * step_s for i in range(n)]
    df = pd.DataFrame({
        "time": times, "batch_time": times, "icao24": icao24, "callsign": "TEST1",
        "trajectory_id": icao24 + "_0",
        "lat": [start_lat + speed * (t - start_time) / M_PER_DEG for t in times],
        "lon": lon,
        "baro_altitude_m": [altitude + climb * (t - start_time) for t in times],
        "velocity_mps": speed if reported_speed is None else reported_speed,
        "heading_deg": 0.0, "vertical_rate_mps": climb, "on_ground": False,
        "aircraft_class": cls, "aircraft_class_source": source,
    })
    return add_features(df)


def checks_in(flags):
    return set(flags["check"])


def test_normal_flights_get_no_flags():
    # a jet at 250 m/s (486 kt), a light aircraft at 60 m/s (117 kt), a helicopter at 50 m/s (97 kt)
    for cls, speed in (("jet", 250.0), ("light", 60.0), ("rotorcraft", 50.0)):
        flags = run_all_checks(flight(cls=cls, speed=speed))
        assert flags.empty, (cls, flags)


def test_flags_table_has_the_shared_format():
    flags = run_all_checks(flight(cls="light", speed=250.0))
    assert list(flags.columns) == FLAG_COLUMNS
    assert set(flags["detector"]) == {"rules"}


def test_speed_light_aircraft_at_jet_speed_is_flagged():
    # 250 m/s (486 kt) is normal for a jet and impossible for a light aircraft
    flags = check_speed(flight(cls="light", speed=250.0))
    assert checks_in(flags) == {"speed_reported", "speed_implied"}
    assert set(flags["threshold"]) == {200.0}


def test_speed_mismatch_is_flagged():
    # the positions move at 250 m/s but the aircraft says 100 m/s
    flags = check_speed(flight(cls="jet", speed=250.0, reported_speed=100.0))
    assert checks_in(flags) == {"speed_mismatch"}


def test_callsign_only_light_aircraft_uses_the_unknown_limits():
    # "light" guessed from an N-number: could be a business jet, so 250 m/s is allowed
    assert check_speed(flight(cls="light", source="callsign", speed=250.0)).empty


def test_single_jump_is_a_position_jump_not_a_duplicate():
    df = flight(n=12)
    # from the 6th report on, the aircraft is suddenly 1 degree (111 km) further north
    df.loc[5:, "lat"] += 1.0
    df = recompute(df)
    assert checks_in(check_speed(df)) == {"position_jump"}
    assert len(check_speed(df)) == 1
    assert check_duplicate_icao(df).empty


def test_alternating_track_is_a_duplicate_icao():
    # two aircraft 2 degrees (222 km) apart send the same icao24, their reports interleave
    a = flight(icao24="dup001", n=6, step_s=20, start_time=1000, start_lat=40.0)
    b = flight(icao24="dup001", n=6, step_s=20, start_time=1010, start_lat=42.0)
    df = recompute(pd.concat([a, b]))
    dup = check_duplicate_icao(df)
    assert len(dup) == 11                       # every step jumps to the other aircraft
    assert checks_in(dup) == {"duplicate_icao"}
    assert check_speed(df).empty                # not reported again as jumps or speed problems
    # a normal flight is not a duplicate
    assert check_duplicate_icao(flight()).empty


def test_climb_rate_is_flagged():
    # 60 m/s is about 11,800 ft/min: beyond even an emergency descent rate for a jet
    assert checks_in(check_changes(flight(cls="jet", climb=60.0))) == {"climb_rate"}
    # 10 m/s (about 1,970 ft/min) is a normal airliner climb
    assert check_changes(flight(cls="jet", climb=10.0)).empty
    # 20 m/s (3,940 ft/min) is fine for a light turboprop, 30 m/s (5,900 ft/min) is not
    assert check_changes(flight(cls="light", speed=80.0, climb=20.0)).empty
    assert checks_in(check_changes(flight(cls="light", speed=80.0, climb=30.0))) == {"climb_rate"}


def test_acceleration_is_flagged():
    df = flight(cls="jet")
    # the reported speed goes up by 80 m/s every 10 s: 8 m/s2, far more than any airliner
    df["velocity_mps"] = [150.0 + 80.0 * i for i in range(len(df))]
    df = recompute(df)
    assert "acceleration" in checks_in(check_changes(df))
    assert check_changes(flight(cls="jet")).empty


def detour(cls="jet", altitude=10000.0, n=91, bulge_deg=1.0):
    # 15 minutes north at 250 m/s (225 km), with a sideways bulge in the middle.
    # At latitude 41, one degree of longitude is about 84 km.
    df = flight(cls=cls, n=n, altitude=altitude)
    df["lon"] = -82.0 + bulge_deg * np.sin(np.linspace(0, np.pi, n))
    return recompute(df)


def test_route_deviation_is_flagged_for_a_cruising_jet():
    flags = check_route_deviation(detour(bulge_deg=1.0))
    assert checks_in(flags) == {"route_deviation"}
    assert flags["value"].max() > 80_000        # about 84 km off the direct line
    assert set(flags["threshold"]) == {50_000.0}
    # a straight flight of the same length is not flagged
    assert check_route_deviation(flight(n=91)).empty
    # a small bend (about 17 km) stays inside the corridor
    assert check_route_deviation(detour(bulge_deg=0.2)).empty


def test_route_is_not_judged_for_low_jets_or_short_flights():
    # the same detour at 3,000 m (about 9,800 ft): an arrival or departure, not judged
    assert check_route_deviation(detour(altitude=3000.0)).empty
    # a light aircraft is judged at any altitude
    assert not check_route_deviation(detour(cls="light", altitude=1500.0)).empty
    # 5 minutes is too short to say what the route is
    assert check_route_deviation(detour(n=31)).empty


def test_only_step_based_flags_can_be_artifacts():
    df = flight(cls="light", speed=250.0)
    df["batch_artifact"] = True  # pretend every batch is unreliable
    flags = run_all_checks(df)
    by_check = flags.groupby("check")["is_artifact"].all()
    assert by_check["speed_implied"]            # compares two positions: can be a data problem
    assert not by_check["speed_reported"]       # what the aircraft says does not depend on the batch


def test_class_mismatch_lists_a_helicopter_at_jet_speed():
    # registered as a helicopter (database) but cruising at 217 m/s (422 kt)
    odd = find_class_mismatches(flight(icao24="c06b2c", cls="rotorcraft", speed=217.0))
    assert list(odd["icao24"]) == ["c06b2c"]
    assert odd["max_speed_mps"].iloc[0] == 110.0
    # a real helicopter speed is not listed, and neither is a class that is only a guess
    assert find_class_mismatches(flight(cls="rotorcraft", speed=50.0)).empty
    assert find_class_mismatches(flight(cls="light", source="callsign", speed=250.0)).empty
