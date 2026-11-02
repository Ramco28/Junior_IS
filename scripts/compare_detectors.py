"""Feature 9: compare the isolation forest with the rules, per flight, on the test days.

    python scripts/compare_detectors.py

It uses the saved outputs of feature 8 (data/models/) and the labeled
trajectories of feature 7, and retrains nothing. It prints the per-flight
comparison and writes to data/report/:
    comparison.json      every number printed here
    flights_test.parquet one row per test flight with both detectors' answers
    disagreements.csv    the flights only one detector flags, with empty
                         my_verdict and my_note columns for my manual inspection
If disagreements.csv already has a verdict or a note in it, the script stops
before writing anything, so my manual work is never overwritten.
The comparison is against my rules' weak labels, not against ground truth.
"""
import json

import pandas as pd

import _path  # noqa: F401  (lets me import adsb from here)
from adsb.compare import (SPLIT, agreement, by_length, confusion, disagreements, flight_table,
                          has_manual_work, per_class)
from adsb.config import LABELED_DIR, MODELS_DIR, REPORT_DIR
from adsb.model import model_rows

# what model_rows needs to rebuild the forest's inputs, plus position and identity
COLUMNS = ["trajectory_id", "time", "batch_time", "icao24", "callsign", "aircraft_class",
           "aircraft_class_source", "split", "day", "on_ground", "batch_artifact", "dt_s",
           "lat", "lon", "velocity_mps", "heading_deg", "speed_mismatch_mps", "vrate_mismatch_mps",
           "alt_rate_mps", "accel_mps2", "turn_rate_dps"]


def pct(x):
    return "n/a" if pd.isna(x) else f"{100 * x:.1f}%"


def main():
    csv_path = REPORT_DIR / "disagreements.csv"
    if has_manual_work(csv_path):
        raise SystemExit(f"STOPPED: {csv_path} already has my_verdict or my_note filled in. "
                         "Move or rename it first if you really want a new one.")
    settings = json.loads((MODELS_DIR / "iforest_settings.json").read_text())
    threshold = settings["threshold"]
    scores = pd.read_parquet(MODELS_DIR / "iforest_scores.parquet")
    scores = scores[scores["split"] == SPLIT]
    trajectories = pd.read_parquet(LABELED_DIR / "labeled_trajectories.parquet")

    flights = flight_table(trajectories, scores, threshold)
    table = confusion(flights)
    overall = agreement(table)
    classes = per_class(flights)
    lengths = by_length(flights)

    # the forest's inputs on each scored test report, to describe the disagreements
    labeled = pd.read_parquet(LABELED_DIR / "labeled_rows.parquet", columns=COLUMNS,
                              filters=[("split", "==", SPLIT)])
    rows, _ = model_rows(labeled)
    rows = rows.merge(scores[["trajectory_id", "time", "score"]], on=["trajectory_id", "time"], how="inner")
    odd = disagreements(flights, rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    odd.to_csv(csv_path, index=False)
    flights.drop(columns=["checks"]).to_parquet(REPORT_DIR / "flights_test.parquet", index=False)

    n = len(flights)
    not_scored = flights[~flights["scored"]]
    route_only = flights[flights["route_only"]]
    print(f"Test days: {', '.join(settings['test_days'])}. Forest: {settings['n_estimators']} trees, "
          f"threshold {threshold:.4f}.")
    print(f"Test flights: {n:,}. Scored by the forest: {int(flights['scored'].sum()):,}. "
          f"Not scored (no usable report): {len(not_scored):,}, of which {int(not_scored['rules_anomalous'].sum())} "
          f"are rules anomalous. Not scored flights are left out of the comparison.")
    print("\nOnly the REPORT-level rule checks are compared (speed, position jump, climb rate, acceleration).")
    print("Route deviation and duplicate ICAO judge whole trajectories, so they are not part of it.")
    print(f"Test flights flagged ONLY by a trajectory-level check: {len(route_only)} "
          f"(route deviation {int(route_only['route_deviation'].sum())}, "
          f"duplicate ICAO {int(route_only['duplicate_icao'].sum())}).")

    print("\nPer-flight comparison on the scored test flights:")
    print(table.to_string())
    print(f"\nAgainst the rules' weak labels (not ground truth):")
    print(f"  precision {pct(overall['precision'])}: of the {overall['both'] + overall['forest_only']} flights "
          f"the forest flags, {overall['both']} are also flagged by the rules")
    print(f"  recall    {pct(overall['recall'])}: of the {overall['both'] + overall['rules_only']} flights "
          f"the rules flag, the forest also flags {overall['both']}")
    print(f"  the forest flags {pct(overall['rule_normal_flagged'])} of the flights the rules call normal "
          f"({overall['forest_only']} of {overall['forest_only'] + overall['neither']:,})")

    show = classes.copy()
    for col in ["precision", "recall", "rule_normal_flagged"]:
        show[col] = show[col].map(pct)
    print("\nPer aircraft class:")
    print(show[["scored_flights", "not_scored", "both", "rules_only", "forest_only", "neither",
                "precision", "recall", "rule_normal_flagged"]].to_string())

    show = lengths.copy()
    show["rules_flagged"] = show["rules_flagged"].map(pct)
    show["forest_flagged"] = show["forest_flagged"].map(pct)
    print("\nFlagged rate by flight length (quartiles of the number of scorable reports):")
    print(show.to_string())

    counts = odd["flagged_by"].value_counts()
    print(f"\nDisagreements: {len(odd)} flights ({counts.get('forest only', 0)} forest only, "
          f"{counts.get('rules only', 0)} rules only). Saved {REPORT_DIR / 'disagreements.csv'}")

    result = {
        "split": SPLIT, "test_days": settings["test_days"], "threshold": threshold,
        "n_estimators": settings["n_estimators"],
        "flights": n, "scored": int(flights["scored"].sum()), "not_scored": len(not_scored),
        "not_scored_rules_anomalous": int(not_scored["rules_anomalous"].sum()),
        "trajectory_level_only": len(route_only),
        "trajectory_level_only_route_deviation": int(route_only["route_deviation"].sum()),
        "trajectory_level_only_duplicate_icao": int(route_only["duplicate_icao"].sum()),
        "confusion": overall,
        "per_class": classes.to_dict(orient="index"),
        "by_length": lengths.reset_index(names="group").to_dict(orient="records"),
        "disagreements": {"forest_only": int(counts.get("forest only", 0)),
                          "rules_only": int(counts.get("rules only", 0))},
    }
    (REPORT_DIR / "comparison.json").write_text(json.dumps(result, indent=2, default=float) + "\n")

    print("\nIn plain words:")
    print(f"  On the {int(flights['scored'].sum()):,} test flights the forest could score, both detectors flag "
          f"{overall['both']}, both say normal for {overall['neither']:,}, and they disagree on "
          f"{overall['rules_only'] + overall['forest_only']}.")
    print(f"  The forest finds {pct(overall['recall'])} of the flights my rules flag, and "
          f"{pct(overall['precision'])} of what it flags is also flagged by my rules.")
    print("  The disagreements are the interesting part: I check them by hand in disagreements.csv.")


if __name__ == "__main__":
    main()
