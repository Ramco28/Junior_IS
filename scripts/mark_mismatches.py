"""Decide the class mismatches over the whole historical dataset (features 6 and 7).

    python scripts/mark_mismatches.py

A class mismatch is an aircraft that does not fly like the class the aircraft
database gives it (see rules.find_class_mismatches). build_features.py can
only decide this with the one hour it is working on, and one hour is not
enough: an aircraft that climbs or descends during that hour looks slower
than it really is, and is missed.

This script reads ALL the historical feature files in data/features/, makes
one decision per aircraft from all its reports, and writes that decision into
every file. Run it after build_features.py and before run_rules.py.
"""
import pandas as pd

import _path  # noqa: F401  (lets me import adsb from here)
from adsb.config import FEATURES_DIR
from adsb.rules import apply_class_mismatches, find_class_mismatches

# the only columns the decision needs (loading everything would use much more memory)
NEEDED = ["icao24", "callsign", "time", "on_ground", "velocity_mps", "baro_altitude_m",
          "aircraft_class", "aircraft_class_source"]
KT_PER_MPS = 1.944
FT_PER_M = 3.281


def main():
    files = sorted(FEATURES_DIR.glob("features_ohio_*.parquet"))
    if not files:
        raise SystemExit("No historical feature files. Run scripts/build_features.py first.")

    # 1. one table with every report of every aircraft, over all the hours
    everything = pd.concat([pd.read_parquet(f, columns=NEEDED) for f in files], ignore_index=True)
    mismatches = find_class_mismatches(everything)

    # 2. write the same decision into each file
    changed_files = 0
    for path in files:
        df = pd.read_parquet(path)
        marked = apply_class_mismatches(df, mismatches["icao24"])
        if not marked["aircraft_class_source"].equals(df["aircraft_class_source"]):
            marked.to_parquet(path, index=False)
            changed_files += 1

    print(f"Feature files read:    {len(files)} ({len(everything):,} reports)")
    print(f"Class mismatches:      {len(mismatches)} aircraft")
    print(f"Files that changed:    {changed_files}")
    if not mismatches.empty:
        print("By registered class and reason:")
        print(mismatches.groupby(["aircraft_class", "reason"]).size().to_string())
        show = mismatches.assign(median_kt=(mismatches["median_speed_mps"] * KT_PER_MPS).round(),
                                 high_alt_ft=(mismatches["high_altitude_m"] * FT_PER_M).round(-2))
        print("\n" + show[["icao24", "callsign", "aircraft_class", "reason", "median_kt",
                           "high_alt_ft", "reports"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
