"""Feature 8: train the isolation forest baseline and flag reports with it.

    python scripts/train_model.py

It reads data/labeled/ (made by scripts/label_dataset.py), and:
  1. trains one forest per tree count in N_ESTIMATORS_GRID, on the TRAINING days only
  2. on the VALIDATION days, sets each forest's threshold to the alert budget
     (as many flights as the rules flag) and picks the tree count by stability
  3. scores every report of every day with the chosen forest
It saves to data/models/: iforest.joblib (the forest), iforest_settings.json
(every choice and count), iforest_scores.parquet (the score of every scored
report); and the flags to data/flags/iforest_flags.parquet.
"""
import json
import time

import joblib
import pandas as pd

import _path  # noqa: F401  (lets me import adsb from here)
from adsb.config import FLAGS_DIR, LABELED_DIR, MODELS_DIR
from adsb.labels import TEST_DAYS, VALIDATION_DAYS
from adsb.model import (CHECK, DETECTOR, FEATURES, MAX_SAMPLES, N_ESTIMATORS_GRID,
                        RANDOM_STATE, STABILITY_MIN_OVERLAP, budget_threshold,
                        choose_n_estimators, flagged_flights, flight_scores,
                        forest_flags, model_rows, score_rows, train_forest)

# only the columns the forest and its flags need (the full file is 240 MB)
COLUMNS = ["trajectory_id", "time", "batch_time", "icao24", "aircraft_class", "split", "day",
           "on_ground", "batch_artifact", "dt_s", "velocity_mps", "heading_deg",
           "speed_mismatch_mps", "vrate_mismatch_mps", "alt_rate_mps", "accel_mps2", "turn_rate_dps"]
SPLITS = ["train", "validation", "test"]


def main():
    labeled = pd.read_parquet(LABELED_DIR / "labeled_rows.parquet", columns=COLUMNS)
    trajectories = pd.read_parquet(LABELED_DIR / "labeled_trajectories.parquet")
    rows, missing = model_rows(labeled)
    missing = missing.reindex(columns=SPLITS, fill_value=0)

    print(f"Reports in the labeled dataset: {len(labeled):,}")
    print("Reports the forest uses, per split:", rows["split"].value_counts().reindex(SPLITS).to_dict())
    print("\nReports left out because an input is missing (per feature and split):")
    print(missing.to_string())

    train = rows[rows["split"] == "train"]
    val = rows[rows["split"] == "validation"]

    # The alert budget: the share of validation flights the rules flag at report
    # level, among the flights the forest can score. Only this NUMBER comes from
    # the rules.
    val_flights = set(val["trajectory_id"])
    rule_flagged = set(trajectories.loc[trajectories["flagged_rows"] > 0, "trajectory_id"])
    budget_share = len(val_flights & rule_flagged) / len(val_flights)
    n_budget = round(budget_share * len(val_flights))
    print(f"\nAlert budget: the rules flag {len(val_flights & rule_flagged)} of the {len(val_flights):,} "
          f"scorable validation flights ({100 * budget_share:.2f}%), so each forest flags {n_budget}.")

    forests, thresholds, flagged = {}, {}, {}
    print("\nTree count   threshold   flagged validation flights   training time")
    for n in N_ESTIMATORS_GRID:
        started = time.time()
        forests[n] = train_forest(train, n)
        per_flight = flight_scores(val, score_rows(forests[n], val))
        thresholds[n] = budget_threshold(per_flight, n_budget)
        flagged[n] = flagged_flights(per_flight, thresholds[n])
        print(f"{n:>10}   {thresholds[n]:.4f}      {len(flagged[n]):>5}                     {time.time() - started:.0f} s")

    chosen, overlaps = choose_n_estimators(flagged)
    print("\nStability: share of flagged validation flights in common")
    for (n, m), share in overlaps.items():
        print(f"  {n} vs {m} trees: {100 * share:.1f}%")
    print(f"Chosen: {chosen} trees (smallest count with at least {100 * STABILITY_MIN_OVERLAP:.0f}% in common "
          f"with the next one, or the largest), threshold {thresholds[chosen]:.4f}")

    # score every day with the chosen forest (validation and test are never trained on)
    forest, threshold = forests[chosen], thresholds[chosen]
    scores = score_rows(forest, rows)
    flags = forest_flags(rows, scores, threshold)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FLAGS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(forest, MODELS_DIR / "iforest.joblib")
    rows[["trajectory_id", "time", "icao24", "aircraft_class", "split", "day"]].assign(score=scores) \
        .to_parquet(MODELS_DIR / "iforest_scores.parquet", index=False)
    flags.to_parquet(FLAGS_DIR / "iforest_flags.parquet", index=False)

    # flights per split: flagged by the forest, and with no scorable report at all
    scored_flights = rows.groupby("split")["trajectory_id"].nunique()
    all_flights = trajectories.groupby("split")["trajectory_id"].nunique()
    flagged_per_split = flags.merge(rows[["trajectory_id", "split"]].drop_duplicates(), on="trajectory_id") \
        .groupby("split")["trajectory_id"].nunique()
    per_split = pd.DataFrame({"flights": all_flights, "scored": scored_flights,
                              "not scored": all_flights - scored_flights,
                              "flagged": flagged_per_split}).reindex(SPLITS).fillna(0).astype(int)
    per_split["flagged % of scored"] = (100 * per_split["flagged"] / per_split["scored"]).round(2)

    settings = {
        "detector": DETECTOR, "check": CHECK, "features": FEATURES,
        "n_estimators": chosen, "n_estimators_grid": N_ESTIMATORS_GRID,
        "max_samples": MAX_SAMPLES, "random_state": RANDOM_STATE,
        "threshold": threshold, "thresholds_by_n_estimators": {str(n): t for n, t in thresholds.items()},
        "threshold_criterion": "alert budget: flag as many validation flights as the rules flag at report "
                               "level, among the validation flights the forest can score",
        "n_estimators_criterion": f"stability: smallest count whose flagged validation flights are at least "
                                  f"{100 * STABILITY_MIN_OVERLAP:.0f}% the same as with the next larger count",
        "overlaps": {f"{n}_vs_{m}": share for (n, m), share in overlaps.items()},
        "budget_share": budget_share, "budget_flights": n_budget,
        "validation_scorable_flights": len(val_flights),
        "training_days": sorted(train["day"].unique()),
        "validation_days": VALIDATION_DAYS, "test_days": TEST_DAYS,
        "reports_used": rows["split"].value_counts().reindex(SPLITS).astype(int).to_dict(),
        "reports_left_out_missing_input": {f: missing.loc[f].astype(int).to_dict() for f in missing.index},
        "flights_per_split": per_split.drop(columns="flagged % of scored").to_dict(orient="index"),
    }
    (MODELS_DIR / "iforest_settings.json").write_text(json.dumps(settings, indent=2, default=int) + "\n")

    print("\nFlights per split:")
    print(per_split.to_string())
    print(f"\nFlagged reports: {len(flags):,}")
    print(f"Saved {MODELS_DIR / 'iforest.joblib'}, iforest_settings.json, iforest_scores.parquet")
    print(f"Saved {FLAGS_DIR / 'iforest_flags.parquet'}")


if __name__ == "__main__":
    main()
