"""Tests for feature 18. Run them from the repo root with: python -m pytest"""
import pandas as pd

from adsb.aircraft import add_aircraft_class, class_from_icao_code

# a tiny fake aircraft database, so the tests need no download
FAKE_DB = pd.DataFrame({
    "icao24": ["aaa001", "aaa002", "aaa003"],
    "icao_class": ["L2J", "H1T", None],   # a jet, a helicopter, and an entry with no class
    "wtc": ["M", "L", None],
})


def table(rows):
    # rows of (icao24, callsign, category) as a small normalized table
    df = pd.DataFrame(rows, columns=["icao24", "callsign", "category"])
    df["category"] = df["category"].astype("Int64")
    return df


def classes(rows):
    out = add_aircraft_class(table(rows), FAKE_DB)
    return list(zip(out["aircraft_class"], out["aircraft_class_source"]))


def test_database_beats_category_and_callsign():
    # the database says jet; the category says light and the callsign looks private
    assert classes([("aaa001", "N123AB", 2)]) == [("jet", "database")]


def test_category_beats_callsign():
    # not in the database; the category says rotorcraft, the callsign would say light
    assert classes([("bbb001", "N123AB", 8)]) == [("rotorcraft", "category")]
    # in the database but without a class code: the category is used as well
    assert classes([("aaa003", "N123AB", 8)]) == [("rotorcraft", "category")]


def test_callsign_is_the_last_fallback():
    assert classes([("bbb002", "N123AB", None)]) == [("light", "callsign")]
    assert classes([("bbb003", "N5", 0)]) == [("light", "callsign")]  # category 0 = no information
    # an airline flight number is not an N-number
    assert classes([("bbb004", "SWA351", None)]) == [("unknown", "none")]
    assert classes([("bbb005", None, None)]) == [("unknown", "none")]


def test_category_small_stays_unknown():
    # category 3 mixes regional jets and large propeller aircraft
    assert classes([("bbb006", "RPA5629", 3)]) == [("unknown", "none")]


def test_glider_and_ultralight_categories_are_light():
    assert classes([("bbb007", None, 9)]) == [("light", "category")]
    assert classes([("bbb008", None, 12)]) == [("light", "category")]


def test_all_rows_of_one_aircraft_get_the_same_class():
    # the callsign is only present in the second report
    rows = [("bbb009", None, None), ("bbb009", "N777XY", None), ("bbb009", None, None)]
    assert classes(rows) == [("light", "callsign")] * 3


def test_seaplanes_and_amphibians_are_classified_by_engine():
    assert class_from_icao_code("S1P", "L") == "light"   # piston float plane
    assert class_from_icao_code("A1P", "L") == "light"   # piston amphibian
    assert class_from_icao_code("A2J", "M") == "jet"     # jet amphibian
    assert class_from_icao_code("A2T", "M") == "jet"     # medium turboprop amphibian
    assert class_from_icao_code("S1T", "L") == "light"   # light turboprop on floats


def test_rotorcraft_is_checked_before_engine_type():
    assert class_from_icao_code("H1T", "L") == "rotorcraft"  # turbine helicopter, not a turboprop
    assert class_from_icao_code("H2T", "M") == "rotorcraft"
    assert class_from_icao_code("H1P", "L") == "rotorcraft"
    assert class_from_icao_code("G1P", "L") == "rotorcraft"  # gyrocopter


def test_turboprops_are_split_by_weight():
    assert class_from_icao_code("L1T", "L") == "light"    # for example a Cessna Caravan
    assert class_from_icao_code("L2T", "M") == "jet"      # for example a Dash 8
    assert class_from_icao_code("L2T", None) is None      # weight not known: no guess
    assert class_from_icao_code(None, None) is None
    assert class_from_icao_code("L2", "M") is None        # malformed code
