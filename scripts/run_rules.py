"""Feature 6: run the rule-based anomaly checks on a feature file.

    python scripts/run_rules.py --live
    python scripts/run_rules.py --historical data/features/features_ohio_20260920T1400_1h.parquet

The input is a file made by scripts/build_features.py (--live uses
data/features/features_live.parquet). All flags are saved as a Parquet file
in data/flags/, and I print a summary: flags per check and aircraft class,
data artifacts separately, aircraft that do not match their registered class,
and the aircraft with the most flags.
"""
import argparse
from pathlib import Path

import pandas as pd

import _path  # noqa: F401  (lets me import adsb from here)
from adsb.config import FEATURES_DIR, FLAGS_DIR
from adsb.rules import list_marked_mismatches, run_all_checks

KT_PER_MPS = 1.944   # knots in one m/s
FT_PER_M = 3.281     # feet in one meter


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--historical", metavar="FEATURES", help="a features file saved by scripts/build_features.py")
    source.add_argument("--live", action="store_true", help="use data/features/features_live.parquet")
    args = ap.parse_args()

    path = FEATURES_DIR / "features_live.parquet" if args.live else Path(args.historical)
    if not path.exists():
        raise SystemExit(f"No feature file at {path}. Run scripts/build_features.py first.")
    df = pd.read_parquet(path)
    if "aircraft_class" not in df.columns:
        raise SystemExit(f"{path.name} has no aircraft_class column. Run scripts/build_features.py again.")

    flags = run_all_checks(df)

    FLAGS_DIR.mkdir(parents=True, exist_ok=True)
    out = FLAGS_DIR / path.name.replace("features_", "flags_")
    flags.to_parquet(out, index=False)

    anomalies = flags[~flags["is_artifact"]]   # blamed on the aircraft
    artifacts = flags[flags["is_artifact"]]    # blamed on the data
    print(f"Reports checked:    {len(df):,} in {df['trajectory_id'].nunique():,} trajectories")
    print(f"Aircraft anomalies: {len(anomalies):,} flags in {anomalies['trajectory_id'].nunique():,} trajectories")
    print(f"Data artifacts:     {len(artifacts):,} flags (from unreliable batches, not counted above)")
    print(f"Saved {out}")

    if not anomalies.empty:
        # one line per check and class: how many flags, in how many trajectories,
        # and what share of that class's trajectories this is
        class_sizes = df.groupby("aircraft_class")["trajectory_id"].nunique()
        table = anomalies.groupby(["check", "aircraft_class"]).agg(
            flags=("time", "size"), trajectories=("trajectory_id", "nunique")).reset_index()
        table["% of class"] = (100 * table["trajectories"] / table["aircraft_class"].map(class_sizes)).round(1)
        print("\nAircraft anomalies per check and class:")
        print(table.to_string(index=False))

    if not artifacts.empty:
        print("\nData artifacts per check:")
        print(artifacts["check"].value_counts().to_string())

    # a separate finding: the aircraft is not what the database says it is.
    # The decision was made when the features were built (and, for the
    # historical dataset, by scripts/mark_mismatches.py over all the hours).
    odd = list_marked_mismatches(df)
    print(f"\nAircraft that do not behave like their registered class: {len(odd)}")
    if not odd.empty:
        odd = odd.assign(median_kt=(odd["median_speed_mps"] * KT_PER_MPS).round(),
                         class_limit_kt=(odd["max_speed_mps"] * KT_PER_MPS).round(),
                         high_alt_ft=(odd["high_altitude_m"] * FT_PER_M).round(-2))
        print(odd[["icao24", "callsign", "aircraft_class", "reason", "median_kt", "class_limit_kt",
                   "high_alt_ft", "reports"]].to_string(index=False))

    if not anomalies.empty:
        # the 10 aircraft with the most flags, and which checks they triggered
        top = anomalies.groupby("icao24").agg(
            aircraft_class=("aircraft_class", "first"),
            flags=("time", "size"),
            checks=("check", lambda c: ", ".join(sorted(set(c)))),
        ).sort_values("flags", ascending=False).head(10).reset_index()
        callsigns = df.groupby("icao24")["callsign"].first()
        top.insert(1, "callsign", top["icao24"].map(callsigns))
        print("\nAircraft with the most flags:")
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
