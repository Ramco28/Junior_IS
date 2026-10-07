"""Finding out what kind of aircraft is behind an icao24 (feature 18).

Most aircraft send no usable category on ADS-B, so I look them up in the
OpenSky aircraft database. I sort every aircraft into one of four classes,
because a rule like "too fast" needs a different limit for a small piston
airplane, a helicopter and an airliner.
"""
import numpy as np
import pandas as pd

from .config import AIRCRAFT_DB_FILE

CLASSES = ["light", "rotorcraft", "jet", "unknown"]

# The live API category (ADS-B emitter category) -> my class.
# Everything not listed here stays unknown, including 0 and 1 (no information)
# and 3 ("small", 15,500 to 75,000 lbs), which mixes regional jets and large
# propeller aircraft so I cannot tell which limits to use.
CATEGORY_CLASS = {
    2: "light",        # light, under 15,500 lbs
    9: "light",        # glider or sailplane
    12: "light",       # ultralight, hang-glider, paraglider
    8: "rotorcraft",
    4: "jet",          # large, 75,000 to 300,000 lbs
    5: "jet",          # high vortex large (for example Boeing 757)
    6: "jet",          # heavy, over 300,000 lbs
    7: "jet",          # high performance (fast, over 400 knots)
}

# A US registration used as a callsign: N, then 1 to 5 characters that start
# with a digit from 1 to 9 and may end with one or two letters (never I or O).
# Small private airplanes usually fly under their registration, airlines use a
# flight number like SWA351.
N_NUMBER = r"N[1-9](\d{0,4}|\d{0,3}[A-HJ-NP-Z]|\d{0,2}[A-HJ-NP-Z]{2})"


def load_aircraft_db(path=AIRCRAFT_DB_FILE) -> pd.DataFrame:
    """Load my slim copy of the OpenSky aircraft database (one row per icao24)."""
    if not path.exists():
        raise SystemExit("No aircraft database yet. Run scripts/download_aircraft_db.py first.")
    return pd.read_parquet(path)


def class_from_icao_code(code, wtc):
    """Turn an ICAO aircraft class code into one of my classes, or None.

    The code has three characters, for example L2J:
      1st: kind of aircraft. L landplane, S seaplane, A amphibian,
           H helicopter, G gyrocopter.
      2nd: number of engines.
      3rd: engine type. P piston, T turboprop, J jet, E electric.
    wtc is the wake turbulence category of the type: L (light, up to 7,000 kg),
    M (medium) or H (heavy).
    """
    if not isinstance(code, str) or len(code) != 3:
        return None  # missing or malformed code
    kind, engine = code[0], code[2]
    # rotorcraft first: a helicopter with a turbine engine (H1T) must not be
    # mistaken for a turboprop airplane
    if kind in ("H", "G"):
        return "rotorcraft"
    # seaplanes and amphibians are airplanes on floats, so I classify them by
    # their engine exactly like landplanes
    if kind not in ("L", "S", "A"):
        return None
    if engine == "J":
        return "jet"
    if engine == "P":
        return "light"
    # turboprop or electric: a Cessna Caravan is small, a Dash 8 is an airliner,
    # so the weight decides. "L/M" (some versions light, some medium) counts as light.
    if isinstance(wtc, str):
        return "light" if wtc.startswith("L") else "jet"
    return None


def add_aircraft_class(df: pd.DataFrame, db: pd.DataFrame) -> pd.DataFrame:
    """Add aircraft_class and aircraft_class_source to a normalized table.

    I try three sources, from the most to the least reliable:
      1. database: the aircraft database, looked up by icao24
      2. category: the category the aircraft itself sends (live data only)
      3. callsign: a callsign that looks like a US N-number means "light".
         This is a weak guess, business jets also fly under N-numbers.
    If none of them gives an answer the class is "unknown" and the source "none".
    """
    df = df.copy()
    db = db.set_index("icao24")

    # 1. database: look up the class code and weight category of each aircraft
    codes = df["icao24"].map(db["icao_class"])
    wtcs = df["icao24"].map(db["wtc"])
    from_database = pd.Series([class_from_icao_code(c, w) for c, w in zip(codes, wtcs)],
                              index=df.index, dtype=object)

    # An aircraft does not send its category and callsign in every report, so I
    # use the first value each aircraft ever gave ("first" skips missing values).
    # That way all rows of one aircraft get the same class.
    category = df.groupby("icao24")["category"].transform("first")
    callsign = df.groupby("icao24")["callsign"].transform("first")

    # 2. category sent by the aircraft
    from_category = category.map(CATEGORY_CLASS).astype(object)

    # 3. callsign that looks like an N-number
    is_n_number = callsign.str.fullmatch(N_NUMBER).fillna(False).astype(bool)
    from_callsign = pd.Series(np.where(is_n_number, "light", None), index=df.index, dtype=object)

    # I fill in the weakest source first and let each stronger source overwrite it
    df["aircraft_class"] = "unknown"
    df["aircraft_class_source"] = "none"
    for source, guess in [("callsign", from_callsign), ("category", from_category),
                          ("database", from_database)]:
        known = guess.notna()
        df.loc[known, "aircraft_class"] = guess[known]
        df.loc[known, "aircraft_class_source"] = source
    return df
