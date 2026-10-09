"""Weak labels for the historical dataset (feature 7).

I have no ground truth: nobody has marked which flights were really anomalous.
So I use my own rule-based checks (rules.py) as labels. A report that a rule
flagged is labeled "anomaly", the others "normal". This is called weak
labeling: the labels are cheap and cover everything, but they are only as
good as the rules.

THE MAIN LIMITATION. The labels come from my own rules, so a model trained on
them will partly learn to imitate the rules. If it agrees with them 99% of the
time, that does not prove it finds anomalies, only that it copies well. This
is why the evaluation (feature 9) must look hardest at where the two detectors
DISAGREE: reports the model flags and the rules do not, and the other way
around. Those are the cases that can teach me something.

TWO LEVELS OF LABELS. Some checks judge one report (is this speed possible?),
others judge a whole flight (did it stray from its route? does the track
alternate between two aircraft?). A model that looks at one report at a time
cannot learn a property of the whole trajectory: nothing in a single report
says the flight is off its route. So:
  - the ROW label uses only the report-level checks (REPORT_LEVEL_CHECKS);
  - route_deviation and duplicate_icao are columns of the TRAJECTORY table.
Feature 9 must then compare like with like: a report-level model against the
row labels, a trajectory-level model against the trajectory labels.

EVALUATE PER FLIGHT. The flagged reports are not spread evenly: a few flights
hold a large share of them (one flight can have 60 flagged reports, most have
one or two). If feature 9 counted reports, its scores would mostly describe
those few flights, and a detector could look good by getting one flight right.
So feature 9 should count FLIGHTS: a flight counts once, as found or missed,
however many of its reports are flagged. The trajectory table has what this
needs (flagged_rows says how many reports of the flight are labeled anomaly).
"""
import pandas as pd

# checks that judge a single report: these decide the row label
REPORT_LEVEL_CHECKS = ["speed_reported", "speed_implied", "speed_mismatch",
                       "position_jump", "climb_rate", "acceleration"]
# checks that judge a whole trajectory: columns of the trajectory table only
TRAJECTORY_LEVEL_CHECKS = ["route_deviation", "duplicate_icao"]

# Whole days held out from training (see add_split). The validation day is for
# tuning the model, the test days are only used once, for the final numbers.
VALIDATION_DAYS = ["2026-09-30"]
TEST_DAYS = ["2026-10-03", "2026-10-06"]


def label_rows(features: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Add two columns to a feature table: label and checks.

    features is a table from scripts/build_features.py, flags is the matching
    table from scripts/run_rules.py.

    checks  the names of all the checks that flagged this report, separated by
            commas ("" if none). Flags that were blamed on the data
            (is_artifact) are not listed. Trajectory-level checks are listed
            too, so nothing is lost, but they do not change the label.
    label   "anomaly"    flagged by at least one report-level check
            "uncertain"  not flagged, but it came in an unreliable batch, so I
                         cannot say it is normal either
            "normal"     everything else
    """
    real = flags[~flags["is_artifact"]]  # flags I blame on the aircraft, not on the data
    key = ["trajectory_id", "time"]

    if real.empty:
        # nothing was flagged: no checks to list and no anomaly
        df = features.assign(checks="")
        is_anomaly = pd.Series(False, index=df.index)
    else:
        # one line of text per flagged report: "climb_rate,speed_mismatch"
        checks = (real.groupby(key)["check"]
                  .agg(lambda names: ",".join(sorted(set(names))))
                  .rename("checks").reset_index())
        # the reports flagged by a check that judges a single report
        report_flagged = (real[real["check"].isin(REPORT_LEVEL_CHECKS)][key]
                          .drop_duplicates().assign(report_flagged=True))
        df = features.merge(checks, on=key, how="left").merge(report_flagged, on=key, how="left")
        df["checks"] = df["checks"].fillna("")
        is_anomaly = df.pop("report_flagged").fillna(False).astype(bool)

    df["label"] = "normal"
    df.loc[df["batch_artifact"], "label"] = "uncertain"
    # a real flag wins over "uncertain": the checks that can be caused by a bad
    # batch were already removed above, so what is left does not depend on it
    df.loc[is_anomaly, "label"] = "anomaly"
    return df


def label_window(features: pd.DataFrame, flags: pd.DataFrame, window: str) -> pd.DataFrame:
    """Label one collected hour and make it safe to combine with the others.

    window is the start of the hour as text, for example "20260915T1200".
    A trajectory_id like "a1b2c3_0" is only unique inside one file: the same
    aircraft has the same id on another day. So I put the window in front
    ("20260915T1200_a1b2c3_0") and add the window and the day as columns.
    """
    df = label_rows(features, flags)
    df["trajectory_id"] = window + "_" + df["trajectory_id"]
    df["window"] = window
    df["day"] = f"{window[0:4]}-{window[4:6]}-{window[6:8]}"  # "2026-09-15"
    return df


def add_split(df: pd.DataFrame, validation_days=VALIDATION_DAYS, test_days=TEST_DAYS) -> pd.DataFrame:
    """Add a split column: "train", "validation" or "test", decided by the DAY.

    I hold out whole days and never random rows. Two reports of the same
    flight, 10 seconds apart, are almost identical. With a random split, one
    would land in training and the other in the test set, and the model would
    be tested on flights it has already partly seen. Its score would look much
    better than it really is. Holding out whole days makes sure that no
    flight, and no aircraft on that day, is on both sides.
    """
    split = pd.Series("train", index=df.index)
    split[df["day"].isin(validation_days)] = "validation"
    split[df["day"].isin(test_days)] = "test"
    return df.assign(split=split)


def label_trajectories(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per trajectory, from a labeled row table (label_window + add_split).

    anomalous        True if the trajectory has any real flag from ANY check,
                     report-level or trajectory-level
    flagged_rows     how many of its reports are labeled "anomaly"
    route_deviation  True if the route check flagged it
    duplicate_icao   True if the duplicate ICAO check flagged it
    class_mismatch   True if the aircraft does not fly like its registered
                     class (rules.find_class_mismatches). This is a finding
                     about the aircraft's IDENTITY, not about this flight, so
                     it has its own column and does not make the trajectory
                     anomalous.
    """
    work = rows.assign(
        _flagged=rows["label"] == "anomaly",
        _uncertain=rows["label"] == "uncertain",
        _any_check=rows["checks"] != "",
        _route=rows["checks"].str.contains("route_deviation"),
        _duplicate=rows["checks"].str.contains("duplicate_icao"),
        _mismatch=rows["aircraft_class_source"] == "mismatch",
    )
    out = work.groupby("trajectory_id").agg(
        icao24=("icao24", "first"),
        aircraft_class=("aircraft_class", "first"),
        window=("window", "first"),
        day=("day", "first"),
        split=("split", "first"),
        rows=("time", "size"),
        flagged_rows=("_flagged", "sum"),
        uncertain_rows=("_uncertain", "sum"),
        anomalous=("_any_check", "any"),
        route_deviation=("_route", "any"),
        duplicate_icao=("_duplicate", "any"),
        class_mismatch=("_mismatch", "any"),
    ).reset_index()
    # every check that flagged any report of the trajectory, as one text
    flagged = rows[rows["checks"] != ""]
    all_checks = flagged.groupby("trajectory_id")["checks"].agg(
        lambda texts: ",".join(sorted(set(",".join(texts).split(",")))))
    out["checks"] = out["trajectory_id"].map(all_checks).fillna("")
    return out
