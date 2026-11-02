"""The baseline machine learning detector: an isolation forest (feature 8).

An isolation forest is unsupervised: it never sees my labels. It builds many
random trees that split the data on random features at random values. A
report that is unusual gets isolated (alone in a leaf) after only a few
splits, a normal report needs many. The fewer splits needed on average, the
higher the anomaly score.

The forest only learns what is common in the training days, without my
expert knowledge (no aircraft class, no thresholds). The one thing it takes
from my rules is HOW MANY flights to flag (see budget_threshold), never WHICH
flights, so the comparison in feature 9 stays fair.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .rules import CHECK_DT_RANGE_S, FLAG_COLUMNS, _airborne_step

# The seven inputs (chosen after looking at the repeated reports, see my notes).
# speed_mismatch_mps keeps its sign: together with turn_rate_dps the forest can
# learn that a too LOW implied speed is normal in a turn, like my turn guard.
FEATURES = [
    "speed_mismatch_mps",   # implied minus reported speed
    "vrate_mismatch_mps",   # measured minus reported climb rate
    "alt_rate_mps",         # climb rate measured from the altitudes
    "velocity_mps",         # reported speed: gives the other numbers their scale
    "accel_mps2",           # change in reported speed
    "turn_rate_dps",        # change in heading
    "repeated",             # 1 if speed AND heading are exactly the previous report's
]

RANDOM_STATE = 0                  # fixed, so every run gives the same forest
# Tree counts tried on the validation days. I first tried [100, 200, 400]: no
# pair reached the 95% stability rule (200 vs 400 was 92.9%), so 400 was only a
# fallback. I doubled the grid twice to see if the flagged flights settle. They
# did not: 400 vs 800 and 800 vs 1600 are also 92.9% (39 of 42 flights in
# common), so 1600 is chosen as the fallback again. I stopped there and report
# it as a limitation: about 3 of the 42 flagged flights depend on randomness.
N_ESTIMATORS_GRID = [100, 200, 400, 800, 1600]
MAX_SAMPLES = 256                 # reports per tree (scikit-learn's default): enough to isolate
                                  # outliers, and each tree stays small and fast
STABILITY_MIN_OVERLAP = 0.95      # see choose_n_estimators
DETECTOR = "iforest"
CHECK = "iforest_score"


def add_repeated(df: pd.DataFrame) -> pd.DataFrame:
    """Add the column repeated: 1 if speed and heading equal the previous report of the SAME flight.

    About a quarter of the reports repeat both values exactly, mostly jets in
    steady cruise. On those rows acceleration and turn rate are exactly 0, and
    this column tells the forest why. The first report of a flight has no
    previous report, so it is never "repeated".
    """
    df = df.sort_values(["trajectory_id", "time"]).reset_index(drop=True)
    g = df.groupby("trajectory_id")
    same = (df["velocity_mps"] == g["velocity_mps"].shift()) & (df["heading_deg"] == g["heading_deg"].shift())
    return df.assign(repeated=same.astype(int))


def model_rows(df: pd.DataFrame):
    """The reports the forest can use, and a count of the ones it cannot.

    I use the same filter as the rules, so both detectors judge the same
    reports: airborne at both ends of the step, 5 to 60 s after the previous
    report, and not in an unreliable batch. Then I leave out every report with
    a missing input (the forest cannot work with NaN). I do not fill them in:
    a filled-in 0 or median would be an invented, normal-looking value.

    Returns (rows, missing). missing is a table with one line per feature plus
    "any feature", one column per split, counting the reports left out.
    """
    df = add_repeated(df)
    usable = df[_airborne_step(df)
                & df["dt_s"].between(*CHECK_DT_RANGE_S)
                & ~df["batch_artifact"]]
    na = usable[FEATURES].isna()
    missing = na.groupby(usable["split"]).sum().T
    missing.loc["any feature"] = na.any(axis=1).groupby(usable["split"]).sum()
    rows = usable[~na.any(axis=1)].reset_index(drop=True)
    return rows, missing


def train_forest(rows: pd.DataFrame, n_estimators: int, random_state: int = RANDOM_STATE) -> IsolationForest:
    """Fit an isolation forest on TRAINING reports only.

    I check this here and not only in the script: if a validation or test day
    ever got into training, the forest would be judged on data it has seen,
    and every number in feature 9 would be too good.
    """
    if (rows["split"] != "train").any():
        raise ValueError("train_forest only accepts reports from the training days")
    forest = IsolationForest(n_estimators=n_estimators, max_samples=MAX_SAMPLES,
                             random_state=random_state, n_jobs=-1)
    return forest.fit(rows[FEATURES].to_numpy(dtype="float64"))


def score_rows(forest: IsolationForest, rows: pd.DataFrame) -> pd.Series:
    """Anomaly score of every report: HIGHER means more anomalous.

    scikit-learn's score_samples is the other way round (lower = more
    abnormal), so I flip the sign. That way a bigger number reads as "more
    suspicious" in my tables and my figure, like the rule values.
    """
    return pd.Series(-forest.score_samples(rows[FEATURES].to_numpy(dtype="float64")),
                     index=rows.index, name="score")


def flight_scores(rows: pd.DataFrame, scores: pd.Series) -> pd.Series:
    """One score per flight: the highest score of its reports.

    A flight is flagged if ANY of its reports is above the threshold, which is
    the same as its highest score being above it. Limitation: a long flight has
    more reports, so more chances to cross the threshold by chance.
    """
    return scores.groupby(rows["trajectory_id"]).max()


def budget_threshold(per_flight: pd.Series, n_flagged: int) -> float:
    """The threshold that flags exactly the n_flagged highest-scoring flights.

    This is my alert budget: the forest flags as many flights as the rules do,
    so the two detectors can be compared on equal terms. n_flagged comes from
    the rules, but only as a number.
    """
    if n_flagged < 1:
        raise ValueError("the alert budget must be at least one flight")
    return float(per_flight.sort_values(ascending=False).iloc[n_flagged - 1])


def flagged_flights(per_flight: pd.Series, threshold: float) -> set:
    return set(per_flight.index[per_flight >= threshold])


def overlap(a: set, b: set) -> float:
    """Share of flagged flights two forests have in common (1.0 = the same flights)."""
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a), len(b))


def choose_n_estimators(flagged_by_n: dict, min_overlap: float = STABILITY_MIN_OVERLAP):
    """Pick the tree count by STABILITY, without looking at the rules.

    A forest with few trees gives slightly random scores: another random seed
    would flag other flights. With more trees the scores settle. I take the
    smallest tree count whose flagged flights are at least 95% the same as
    with the next larger count in the grid: adding trees no longer changes the
    answer. If no pair reaches 95%, I take the largest count.

    flagged_by_n maps each tree count to the set of flights it flags at the
    budget threshold. Returns (chosen count, {(n, next n): overlap}).
    """
    counts = sorted(flagged_by_n)
    overlaps = {(n, m): overlap(flagged_by_n[n], flagged_by_n[m]) for n, m in zip(counts, counts[1:])}
    for (n, _), share in overlaps.items():
        if share >= min_overlap:
            return n, overlaps
    return counts[-1], overlaps


def forest_flags(rows: pd.DataFrame, scores: pd.Series, threshold: float) -> pd.DataFrame:
    """The reports above the threshold, in the same table format as the rule flags.

    is_artifact is always False: unreliable batches were already left out by
    model_rows, so the forest never scores a report from one.
    """
    mask = scores >= threshold
    out = rows.loc[mask, ["time", "batch_time", "icao24", "trajectory_id", "aircraft_class"]].copy()
    out["detector"] = DETECTOR
    out["check"] = CHECK
    out["value"] = scores[mask].astype("float64")
    out["threshold"] = float(threshold)
    out["is_artifact"] = False
    return out[FLAG_COLUMNS].reset_index(drop=True)
