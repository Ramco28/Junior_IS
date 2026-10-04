"""Rule-based checks on the feature table (feature 6).

So far this file has one rule, and it is about the DATA, not about one
aircraft: finding deliveries (batches) I cannot trust.
"""
import pandas as pd

# Default thresholds. Every function takes them as arguments, so they can be
# changed without touching the code below.
ARTIFACT_MEDIAN_MPS = 20.0     # a batch is unreliable if its median speed mismatch is further from 0 than this
ARTIFACT_DT_RANGE_S = (5, 60)  # only reports with a time gap in this range are used (works for 10 s and 30 s data)
ARTIFACT_MIN_AIRCRAFT = 10     # a batch with fewer usable aircraft than this cannot be judged


def find_artifact_batches(df: pd.DataFrame,
                          median_threshold_mps: float = ARTIFACT_MEDIAN_MPS,
                          dt_range_s: tuple = ARTIFACT_DT_RANGE_S,
                          min_aircraft: int = ARTIFACT_MIN_AIRCRAFT) -> pd.DataFrame:
    """Find the batches where most aircraft are wrong in the same way.

    df is a feature table (see features.add_features). A batch is all the rows
    with the same batch_time: one live snapshot, or one timestamp of the
    historical table.

    The idea: if ONE aircraft's implied speed does not match its reported
    speed, that aircraft is suspicious. If the MEDIAN aircraft of a batch is
    off, half of the aircraft are wrong together, and it is much more likely
    that the delivery itself is wrong (for example positions that are older
    than their timestamps). I use the median and not the mean because a few
    aircraft with a huge mismatch cannot move the median.

    Returns one row per unreliable batch: batch_time, aircraft, median_mismatch_mps.
    """
    # only reports that can be judged fairly: in the air, with a normal time
    # gap (a very short or very long gap makes the implied speed unreliable)
    usable = df[(~df["on_ground"])
                & df["dt_s"].between(*dt_range_s)
                & df["speed_mismatch_mps"].notna()]
    per_batch = usable.groupby("batch_time").agg(
        aircraft=("icao24", "nunique"),
        median_mismatch_mps=("speed_mismatch_mps", "median"),
    )
    unreliable = per_batch[(per_batch["aircraft"] >= min_aircraft)
                           & (per_batch["median_mismatch_mps"].abs() > median_threshold_mps)]
    return unreliable.reset_index()


def mark_artifacts(df: pd.DataFrame, **thresholds) -> pd.DataFrame:
    """Add a True/False column batch_artifact: is this row from an unreliable batch?

    The aircraft checks can then report a flag on such a row as a data
    artifact and not as an aircraft anomaly. Extra arguments are passed on to
    find_artifact_batches (for example median_threshold_mps=30).
    """
    bad_times = find_artifact_batches(df, **thresholds)["batch_time"]
    return df.assign(batch_artifact=df["batch_time"].isin(bad_times))
