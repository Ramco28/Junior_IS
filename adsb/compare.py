"""Comparing the isolation forest with my rule-based checks, per flight (feature 9).

There is no ground truth this semester: nobody has said which flights really
were anomalous. So I compare the forest with the WEAK LABELS from my rules,
and then I look at the flights where the two detectors disagree myself. The
precision and recall below say how much the forest agrees with my rules, not
how often it is right.

I count FLIGHTS, not reports. A few flights hold many flagged reports, and a
report count would mostly describe those flights. Here every flight counts
once: flagged or not, by each detector.

Like with like: the forest judges one report at a time, so it is compared
with the REPORT-level rule checks only (flagged_rows > 0). Route deviation and
duplicate ICAO judge a whole trajectory; they are left out of the comparison
and counted on their own.
"""
import numpy as np
import pandas as pd

from .model import FEATURES

SPLIT = "test"          # the comparison only uses the test days
N_LENGTH_GROUPS = 4     # quartiles of the number of scorable reports per flight

# the columns of data/report/disagreements.csv, in this order
DISAGREEMENT_COLUMNS = (
    ["trajectory_id", "icao24", "callsign", "aircraft_class", "aircraft_class_source",
     "flagged_by", "rule_checks", "max_score", "score_rank", "scorable_reports",
     "report_time_utc", "lat", "lon"]
    + FEATURES
    + ["my_verdict"]
)


def flight_table(trajectories: pd.DataFrame, scores: pd.DataFrame, threshold: float,
                 split: str = SPLIT) -> pd.DataFrame:
    """One row per flight of the split, with what each detector says about it.

    trajectories  labeled_trajectories.parquet (feature 7)
    scores        iforest_scores.parquet: one row per scored report (feature 8)

    Columns added:
      rules_anomalous   flagged by at least one REPORT-level check
      route_only        anomalous for the rules only because of a trajectory-level
                        check (route deviation or duplicate ICAO)
      scored            the forest could score at least one of its reports
      scorable_reports  how many of its reports the forest scored
      max_score         its highest forest score (missing if not scored)
      score_rank        1 = highest max_score among the scored flights of the split
      forest_flagged    max_score at or above the threshold. Missing when the
                        flight is not scored: "not scored" is not "not flagged".
    """
    flights = trajectories[trajectories["split"] == split].copy()
    flights["rules_anomalous"] = flights["flagged_rows"] > 0
    flights["route_only"] = flights["anomalous"] & ~flights["rules_anomalous"]

    per_flight = scores.groupby("trajectory_id")["score"].agg(["max", "size"])
    flights["max_score"] = flights["trajectory_id"].map(per_flight["max"])
    flights["scorable_reports"] = flights["trajectory_id"].map(per_flight["size"]).fillna(0).astype(int)
    flights["scored"] = flights["scorable_reports"] > 0
    flights["score_rank"] = flights["max_score"].rank(ascending=False, method="min").astype("Int64")
    flights["forest_flagged"] = (flights["max_score"] >= threshold).astype("boolean")
    flights.loc[~flights["scored"], "forest_flagged"] = pd.NA
    return flights.reset_index(drop=True)


def confusion(flights: pd.DataFrame) -> pd.DataFrame:
    """The 2x2 table of SCORED flights: rules (rows) against forest (columns)."""
    scored = flights[flights["scored"]]
    table = pd.crosstab(scored["rules_anomalous"].map({True: "rules anomalous", False: "rules normal"}),
                        scored["forest_flagged"].astype(bool).map({True: "forest flagged", False: "forest not flagged"}))
    return table.reindex(index=["rules anomalous", "rules normal"],
                         columns=["forest flagged", "forest not flagged"], fill_value=0)


def agreement(table: pd.DataFrame) -> dict:
    """Precision and recall of the forest AGAINST THE WEAK LABELS, from a 2x2 table.

    precision           of the flights the forest flags, the share the rules also flag
    recall              of the flights the rules flag, the share the forest also flags
    rule_normal_flagged of the flights the rules call normal, the share the forest flags
    A value is missing (NaN) when its denominator is 0.
    """
    both = table.loc["rules anomalous", "forest flagged"]
    rules_only = table.loc["rules anomalous", "forest not flagged"]
    forest_only = table.loc["rules normal", "forest flagged"]
    neither = table.loc["rules normal", "forest not flagged"]

    def share(a, b):
        return a / b if b else np.nan

    return {"both": int(both), "rules_only": int(rules_only), "forest_only": int(forest_only),
            "neither": int(neither),
            "precision": share(both, both + forest_only),
            "recall": share(both, both + rules_only),
            "rule_normal_flagged": share(forest_only, forest_only + neither)}


def per_class(flights: pd.DataFrame) -> pd.DataFrame:
    """The same numbers for each aircraft class."""
    rows = {}
    for cls, group in flights.groupby("aircraft_class"):
        result = agreement(confusion(group))
        result["scored_flights"] = int(group["scored"].sum())
        result["not_scored"] = int((~group["scored"]).sum())
        rows[cls] = result
    table = pd.DataFrame(rows).T
    counts = ["both", "rules_only", "forest_only", "neither", "scored_flights", "not_scored"]
    return table.astype({c: int for c in counts} | {c: float for c in ["precision", "recall", "rule_normal_flagged"]})


def by_length(flights: pd.DataFrame, groups: int = N_LENGTH_GROUPS) -> pd.DataFrame:
    """Flagged rate of each detector by flight length.

    A flight's forest score is its HIGHEST report score, so a long flight has
    more chances to cross the threshold by chance. The rules flag a flight if
    ANY report breaks a rule, so they have the same bias. I split the scored
    flights into quartiles of their number of scorable reports and show how
    often each detector flags a flight in each quartile.
    """
    scored = flights[flights["scored"]].copy()
    scored["length_group"] = pd.qcut(scored["scorable_reports"], groups, duplicates="drop")
    table = scored.groupby("length_group", observed=True).agg(
        flights=("trajectory_id", "size"),
        min_reports=("scorable_reports", "min"),
        max_reports=("scorable_reports", "max"),
        rules_flagged=("rules_anomalous", "mean"),
        forest_flagged=("forest_flagged", lambda f: f.astype(bool).mean()),
    )
    table.index = [f"Q{i}" for i in range(1, len(table) + 1)]
    return table


def disagreements(flights: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """The scored flights that only ONE detector flags, ready for my manual verdict.

    rows are the scored reports of the split with their score and the model
    inputs (model.model_rows plus the score). For each flight I show its most
    anomalous report according to the forest (its highest score): when and
    where it was, and the seven values the forest saw. my_verdict stays empty;
    I fill it in myself with: genuine anomaly, equipment fault, or false alarm.
    """
    scored = flights[flights["scored"]]
    forest = scored["forest_flagged"].astype(bool)
    odd = scored[forest != scored["rules_anomalous"]].copy()
    odd["flagged_by"] = np.where(odd["forest_flagged"].astype(bool), "forest only", "rules only")
    odd["rule_checks"] = odd["checks"]

    # the report with the highest score of each flight
    top = rows.sort_values("score", ascending=False).drop_duplicates("trajectory_id")
    top = top.set_index("trajectory_id")
    for col in ["lat", "lon", "callsign", "aircraft_class_source"] + FEATURES:
        odd[col] = odd["trajectory_id"].map(top[col])
    odd["report_time_utc"] = pd.to_datetime(odd["trajectory_id"].map(top["time"]), unit="s") \
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    odd["my_verdict"] = ""
    return odd.sort_values(["flagged_by", "score_rank"])[DISAGREEMENT_COLUMNS].reset_index(drop=True)
