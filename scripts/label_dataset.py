"""Feature 7: label the historical dataset with my rule-based checks (weak labeling).

    python scripts/label_dataset.py

It reads every historical feature file in data/features/ and its flags file in
data/flags/ (made by build_features.py and run_rules.py), labels them, and
saves two files in data/labeled/:
    labeled_rows.parquet          one row per report
    labeled_trajectories.parquet  one row per trajectory
Then it prints the class balance: how rare the anomalies are, overall, per
aircraft class, per check and per split.
"""
import pandas as pd

import _path  # noqa: F401  (lets me import adsb from here)
from adsb.config import FEATURES_DIR, FLAGS_DIR, LABELED_DIR
from adsb.labels import (REPORT_LEVEL_CHECKS, TRAJECTORY_LEVEL_CHECKS, add_split,
                         label_trajectories, label_window)


def share_table(df: pd.DataFrame, by: str, positive: pd.Series, name: str) -> pd.DataFrame:
    """Count the rows of df per group, and how many of them are `positive`."""
    table = pd.DataFrame({"total": df.groupby(by).size(),
                          name: positive.groupby(df[by]).sum()})
    table["%"] = (100 * table[name] / table["total"]).round(2)
    return table


def main():
    files = sorted(FEATURES_DIR.glob("features_ohio_*.parquet"))
    if not files:
        raise SystemExit("No historical feature files. Run scripts/build_features.py first.")
    parts = []
    for path in files:
        flags_path = FLAGS_DIR / path.name.replace("features_", "flags_")
        if not flags_path.exists():
            raise SystemExit(f"No flags for {path.name}. Run scripts/run_rules.py on it first.")
        # features_ohio_20260915T1200_1h.parquet -> window "20260915T1200"
        window = path.name.split("_")[2]
        parts.append(label_window(pd.read_parquet(path), pd.read_parquet(flags_path), window))
    rows = add_split(pd.concat(parts, ignore_index=True))
    trajectories = label_trajectories(rows)

    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(LABELED_DIR / "labeled_rows.parquet", index=False)
    trajectories.to_parquet(LABELED_DIR / "labeled_trajectories.parquet", index=False)

    is_anomaly = rows["label"] == "anomaly"
    print(f"Hours: {rows['window'].nunique()}   Days: {rows['day'].nunique()}")
    print(f"Saved {LABELED_DIR / 'labeled_rows.parquet'} and labeled_trajectories.parquet")

    print("\n=== Report level (one row per report) ===")
    counts = rows["label"].value_counts()
    for label in ["normal", "anomaly", "uncertain"]:
        n = counts.get(label, 0)
        print(f"{label:<10} {n:>10,}  {100 * n / len(rows):.3f}%")
    print("\nAnomaly reports per split:")
    print(share_table(rows, "split", is_anomaly, "anomaly").to_string())
    print("\nAnomaly reports per aircraft class:")
    print(share_table(rows, "aircraft_class", is_anomaly, "anomaly").to_string())
    print("\nReports flagged per check (a report can be flagged by more than one):")
    for check in REPORT_LEVEL_CHECKS:
        hit = rows["checks"].str.contains(check)
        print(f"  {check:<16} {hit.sum():>6,} reports in {rows.loc[hit, 'trajectory_id'].nunique():>4,} trajectories")

    print("\n=== Trajectory level (one row per flight) ===")
    n = len(trajectories)
    print(f"trajectories        {n:>7,}")
    print(f"anomalous           {trajectories['anomalous'].sum():>7,}  {100 * trajectories['anomalous'].mean():.2f}%")
    print(f"  with flagged reports {(trajectories['flagged_rows'] > 0).sum():>4,}")
    for check in TRAJECTORY_LEVEL_CHECKS:
        print(f"  {check:<20} {trajectories[check].sum():>4,}")
    print(f"class mismatch      {trajectories['class_mismatch'].sum():>7,}  (about the aircraft, not counted as anomalous)")
    print("\nAnomalous trajectories per split:")
    print(share_table(trajectories, "split", trajectories["anomalous"], "anomalous").to_string())
    print("\nAnomalous trajectories per aircraft class:")
    print(share_table(trajectories, "aircraft_class", trajectories["anomalous"], "anomalous").to_string())

    # why accuracy is the wrong score here
    normal_share = 100 * (rows["label"] == "normal").mean()
    print(f"\nNote: a detector that always answers \"normal\" would be right on {normal_share:.2f}% of the")
    print("reports and would find no anomaly at all. So accuracy says almost nothing here.")
    print("What matters is precision (of the reports it flags, how many are labeled anomaly)")
    print("and recall (of the reports labeled anomaly, how many it finds).")


if __name__ == "__main__":
    main()
