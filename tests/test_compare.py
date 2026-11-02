"""Tests for feature 9. Run them from the repo root with: python -m pytest"""
import pandas as pd

from adsb.compare import DISAGREEMENT_COLUMNS, agreement, confusion, disagreements, flight_table
from adsb.model import FEATURES

THRESHOLD = 0.7


def trajectories():
    # four test flights and one training flight
    return pd.DataFrame({
        "trajectory_id": ["A", "B", "C", "D", "T"],
        "icao24": ["a1", "b1", "c1", "d1", "t1"],
        "aircraft_class": ["jet", "light", "jet", "jet", "jet"],
        "split": ["test", "test", "test", "test", "train"],
        "flagged_rows": [3, 0, 0, 1, 1],              # A and D are flagged by report-level checks
        "anomalous": [True, True, False, True, True],  # B is anomalous by route deviation only
        "route_deviation": [False, True, False, False, False],
        "duplicate_icao": [False] * 5,
        "checks": ["speed_mismatch", "route_deviation", "", "climb_rate", "acceleration"],
    })


def scores():
    # A: five reports, all above the threshold. B: one report above. C: all below.
    # D has no scored report at all. T is a training flight.
    return pd.DataFrame({
        "trajectory_id": ["A"] * 5 + ["B", "B", "C", "C", "T"],
        "time": [1, 2, 3, 4, 5, 1, 2, 1, 2, 1],
        "score": [0.9, 0.8, 0.85, 0.75, 0.95, 0.5, 0.72, 0.4, 0.5, 0.99],
    })


def flights():
    return flight_table(trajectories(), scores(), THRESHOLD)


def test_confusion_counts_flights_not_reports():
    table = confusion(flights())
    # A has five flagged reports but counts once
    assert table.loc["rules anomalous", "forest flagged"] == 1
    assert table.loc["rules normal", "forest flagged"] == 1      # B
    assert table.loc["rules normal", "forest not flagged"] == 1  # C
    assert table.loc["rules anomalous", "forest not flagged"] == 0
    assert table.to_numpy().sum() == 3                            # D is not scored, T is not a test flight


def test_flight_without_a_scorable_report_is_not_scored():
    f = flights().set_index("trajectory_id")
    assert not f.loc["D", "scored"]
    assert pd.isna(f.loc["D", "forest_flagged"])   # "not scored", never "not flagged"
    assert f.loc["D", "scorable_reports"] == 0
    assert "T" not in f.index


def test_route_only_flight_is_not_rules_anomalous():
    f = flights().set_index("trajectory_id")
    assert not f.loc["B", "rules_anomalous"]
    assert f.loc["B", "route_only"]
    assert not f.loc["C", "route_only"]


def test_agreement_numbers():
    result = agreement(confusion(flights()))
    assert result["precision"] == 0.5             # forest flags A and B, the rules only A
    assert result["recall"] == 1.0                # the rules flag A, the forest too
    assert result["rule_normal_flagged"] == 0.5   # of B and C, the forest flags B


def test_disagreements_file_has_the_right_columns():
    rows = scores()
    rows = rows[rows["trajectory_id"] != "T"].assign(lat=40.0, lon=-82.0, callsign="TEST", aircraft_class_source="database")
    for feature in FEATURES:
        rows[feature] = 0.0
    odd = disagreements(flights(), rows)
    assert list(odd.columns) == DISAGREEMENT_COLUMNS
    assert list(odd["trajectory_id"]) == ["B"]      # only one detector flags B
    assert list(odd["flagged_by"]) == ["forest only"]
    assert odd["my_verdict"].iloc[0] == ""
    assert odd["max_score"].iloc[0] == 0.72         # its highest report
