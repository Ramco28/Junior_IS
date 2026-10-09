"""Rule-based anomaly checks on the feature table (feature 6).

Four kinds of checks, each a function that returns a table of flags:
  speed              check_speed            (speed_reported, speed_implied, speed_mismatch, position_jump)
  duplicate ICAO     check_duplicate_icao   (duplicate_icao)
  altitude or speed  check_changes          (climb_rate, acceleration)
  change
  route deviation    check_route_deviation  (route_deviation)
plus one check about the DATA and not about an aircraft: find_artifact_batches.

run_all_checks() runs everything and returns one flags table. The ML detector
(later features) will produce the same columns, so the two can be compared.
"""
import numpy as np
import pandas as pd

from .features import EARTH_RADIUS_M, haversine_m

# ---------------------------------------------------------------------------
# All thresholds are here, in one place. Every check takes them as arguments,
# so they can be changed without touching the code below.
# Units: m/s and meters, like the data. 1 m/s = 1.944 knots = 196.9 ft/min.
# ---------------------------------------------------------------------------

# Limits that depend on the kind of aircraft (see aircraft.py for the classes).
THRESHOLDS = {
    "light": {
        # Ground speed. The fastest light aircraft (single turboprops like the
        # TBM) cruise near 170 m/s (330 kt); with a 30 m/s (58 kt) tailwind
        # that is 200 m/s (389 kt).
        "max_speed_mps": 200.0,
        # Implied minus reported speed. On my data 99% of reports are within
        # about 25 m/s, so 30 m/s (58 kt) is clearly outside the normal noise.
        "max_speed_mismatch_mps": 30.0,
        # Climb or descent. Piston aircraft climb at 2.5 to 10 m/s (500 to
        # 2,000 ft/min), but single turboprops in this class (TBM, T-6 Texan)
        # reach 23 m/s (4,500 ft/min) on my data. 25 m/s is about 4,920 ft/min.
        "max_alt_rate_mps": 25.0,
        # Speed change. 4 m/s2 is about 7.8 kt per second, more than a small
        # aircraft does even on its takeoff roll.
        "max_accel_mps2": 4.0,
        # Distance from the straight line. Small aircraft wander (sightseeing,
        # training, avoiding weather), so the corridor is wide: 50 km (27 NM).
        "max_route_deviation_m": 50_000.0,
        # 0 means: judge the route at any altitude.
        "route_min_altitude_m": 0.0,
        # Ceiling of the class, used to spot aircraft that are not what the
        # database says (find_class_mismatches). Piston aircraft stay below
        # about 25,000 ft, turboprops in the light class top out around
        # 31,000 ft (TBM, PC-12), and a King Air 350 at 35,000 ft. So nothing
        # in this class cruises above 10,973 m (36,000 ft).
        "max_altitude_m": 10_973.0,
    },
    "rotorcraft": {
        # Fast helicopters cruise near 80 m/s (155 kt); with a 30 m/s (58 kt)
        # tailwind that is 110 m/s (214 kt).
        "max_speed_mps": 110.0,
        "max_speed_mismatch_mps": 30.0,      # 58 kt, same reasoning as light
        "max_alt_rate_mps": 15.0,            # about 2,950 ft/min
        # Helicopters can speed up and slow down quickly: 5 m/s2 (9.7 kt per second).
        "max_accel_mps2": 5.0,
        "max_route_deviation_m": 50_000.0,   # 27 NM, they do not fly airways
        "route_min_altitude_m": 0.0,         # any altitude
        # The highest helicopter ceilings are near 20,000 to 25,000 ft, and
        # most stay far below: 7,620 m is 25,000 ft.
        "max_altitude_m": 7_620.0,
    },
    "jet": {
        # Airliners cruise up to Mach 0.9, about 270 m/s (525 kt) true
        # airspeed. A strong jet stream adds up to 100 m/s (195 kt), so
        # 380 m/s (739 kt) over the ground is the most I should ever see.
        "max_speed_mps": 380.0,
        # Jets are faster, so the same timing error gives a bigger mismatch: 50 m/s (97 kt).
        "max_speed_mismatch_mps": 50.0,
        # Normal climbs and descents are 5 to 20 m/s (1,000 to 4,000 ft/min);
        # an emergency descent reaches about 40 m/s (7,870 ft/min).
        "max_alt_rate_mps": 40.0,
        # Airliners accelerate at 2 to 3 m/s2 on takeoff and less in flight;
        # 5 m/s2 is about 9.7 kt per second.
        "max_accel_mps2": 5.0,
        # A jet in cruise flies close to a direct route: 50 km (27 NM) either
        # side. (20 km flagged 17% of cruising jets on my data, which is too many.)
        "max_route_deviation_m": 50_000.0,
        # Only jets that stay above 6,000 m (about 19,700 ft) for the whole
        # trajectory are judged. Below that they are arriving or departing,
        # and controllers bend their paths to line them up for an airport.
        "route_min_altitude_m": 6_000.0,
        "max_altitude_m": float("inf"),      # no ceiling: jets are the highest class I have
    },
    # When I do not know the kind of aircraft I use the loosest limit of each
    # row above, so that I never flag an aircraft only because I could not
    # identify it.
    "unknown": {
        "max_speed_mps": 380.0,              # 739 kt (jet)
        "max_speed_mismatch_mps": 50.0,      # 97 kt (jet)
        "max_alt_rate_mps": 40.0,            # 7,870 ft/min (jet)
        "max_accel_mps2": 5.0,               # 9.7 kt per second
        "max_route_deviation_m": 50_000.0,   # 27 NM (light)
        "route_min_altitude_m": 0.0,         # any altitude
        "max_altitude_m": float("inf"),      # no ceiling
    },
}

# Where a class came from (aircraft_class_source) decides if I trust it enough
# to use its limits. Aircraft with one of these sources get the "unknown" limits:
#   callsign  "light" guessed from an N-number callsign. Business jets fly
#             under N-numbers too.
#   mismatch  the database class is contradicted by how the aircraft flies
#             (see find_class_mismatches).
UNTRUSTED_CLASS_SOURCES = ["callsign", "mismatch"]

# Limits that are the same for every aircraft.
CHECK_DT_RANGE_S = (5, 60)        # speed and change checks only use reports 5 to 60 s apart:
                                  # shorter gaps suffer from 1 s time rounding, longer ones hide turns
# Between two reports I only know the straight line from one position to the
# next. An aircraft that turns in between flies along a curve, and a curve is
# always LONGER than the straight line between its ends. So in a turn the
# implied speed (straight line / time) comes out LOWER than the real speed:
# for a half circle the straight line is only 64% of the path. A turn can
# never make the implied speed too HIGH. That is why, when the heading changed
# by more than this many degrees, I ignore a negative speed mismatch (implied
# below reported) but still check a positive one.
TURN_GUARD_DEG = 45.0
IMPOSSIBLE_SPEED_MPS = 1000.0     # 1,944 kt, about Mach 3: faster than any aircraft that carries ADS-B
MIN_JUMP_M = 5_000.0              # 2.7 NM: a jump must be at least this far, so time rounding cannot cause it
MIN_JUMPS_FOR_DUPLICATE = 3       # A, B, A, B is three jumps: away, back, and away again
ROUTE_MIN_DURATION_S = 600        # a trajectory must last 10 minutes ...
ROUTE_MIN_LENGTH_M = 50_000.0     # ... and cover 50 km (27 NM) start to end before I judge its route

ARTIFACT_MEDIAN_MPS = 20.0     # a batch is unreliable if its median speed mismatch is further from 0 than this (39 kt)
ARTIFACT_DT_RANGE_S = (5, 60)  # only reports with a time gap in this range are used (works for 10 s and 30 s data)
ARTIFACT_MIN_AIRCRAFT = 10     # a batch with fewer usable aircraft than this cannot be judged

# Checks that compare a report with the one before it. Only these can be
# caused by a bad batch (positions older than their timestamps), so only
# their flags can be labeled as artifacts. A reported speed, a route or an
# alternating track is not changed by a delivery that is a few seconds late.
STEP_BASED_CHECKS = {"speed_implied", "speed_mismatch", "position_jump", "climb_rate", "acceleration"}

# The columns of a flags table. The ML detector will use the same ones.
FLAG_COLUMNS = ["time", "batch_time", "icao24", "trajectory_id", "aircraft_class",
                "detector", "check", "value", "threshold", "is_artifact"]


# ---------------------------------------------------------------------------
# The data check: batches I cannot trust
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers shared by the aircraft checks
# ---------------------------------------------------------------------------

def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Sort the table and make sure it has the columns the checks need.

    Class mismatches are NOT decided here. That needs every report I have of
    an aircraft, and a check only sees the table it is given (often one hour).
    See mark_class_mismatches and scripts/mark_mismatches.py.
    """
    if "batch_artifact" not in df.columns:
        df = mark_artifacts(df)
    return df.sort_values(["trajectory_id", "time"]).reset_index(drop=True)


def _limit(df: pd.DataFrame, name: str, thresholds: dict) -> pd.Series:
    """The threshold called `name` for every row, chosen by the aircraft's class."""
    # a class I do not trust (a callsign guess, or a database class that the
    # aircraft's own behavior contradicts) is judged with the unknown limits
    untrusted = df["aircraft_class_source"].isin(UNTRUSTED_CLASS_SOURCES)
    cls = df["aircraft_class"].where(~untrusted, "unknown")
    return cls.map({c: limits[name] for c, limits in thresholds.items()}).astype("float64")


def _flags(df: pd.DataFrame, mask: pd.Series, check: str, value: pd.Series, threshold) -> pd.DataFrame:
    """Build the flags table for the rows where mask is True."""
    out = df.loc[mask, ["time", "batch_time", "icao24", "trajectory_id", "aircraft_class"]].copy()
    out["detector"] = "rules"
    out["check"] = check
    out["value"] = value[mask].astype("float64")           # what I measured
    if isinstance(threshold, pd.Series):
        threshold = threshold[mask]
    out["threshold"] = threshold                           # the limit it broke
    # True = blame the data, not the aircraft (only possible for step-based checks)
    out["is_artifact"] = df.loc[mask, "batch_artifact"] if check in STEP_BASED_CHECKS else False
    return out[FLAG_COLUMNS]


def _in_dt_window(df: pd.DataFrame, dt_range_s: tuple) -> pd.Series:
    return df["dt_s"].between(*dt_range_s)


def _airborne_step(df: pd.DataFrame) -> pd.Series:
    """True where this report AND the one before it were sent in the air.

    On the ground the reported speed cannot be trusted. After landing an
    aircraft keeps sending its position, but the speed field often stays
    frozen at the last value it had in the air (for example 113 kt while the
    aircraft is parked). Its positions then say "not moving" and its speed
    says "113 kt", which looks like a huge speed mismatch but is only a stale
    value. The speed and change checks compare two reports, so both must be
    airborne. Needs the table sorted by trajectory and time (see _prepare).
    """
    # the first report of a trajectory has no previous one: count it as "on the ground"
    previous_on_ground = df.groupby("trajectory_id")["on_ground"].shift(fill_value=True)
    return ~df["on_ground"] & ~previous_on_ground


def _find_jumps(df: pd.DataFrame):
    """Find impossible position jumps, and tell single jumps from alternating tracks.

    A jump is a step that would need more than IMPOSSIBLE_SPEED_MPS over more
    than MIN_JUMP_M. Returns two True/False columns:
      is_jump       this row is a jump away from the previous row
      alternating   this row's trajectory jumps back and forth between two
                    separate tracks (the A, B, A, B pattern of two aircraft
                    sending the same icao24)
    """
    is_jump = (df["implied_speed_mps"] > IMPOSSIBLE_SPEED_MPS) & (df["distance_m"] > MIN_JUMP_M)
    by_trajectory = is_jump.groupby(df["trajectory_id"])
    n_jumps = by_trajectory.transform("sum")
    # Every jump switches to "the other aircraft". Counting the jumps so far
    # and keeping the remainder of the division by 2 gives each row a side:
    # 0, 0, 1, 1, 0, 0, 1 ... (side 0 = track A, side 1 = track B)
    side = by_trajectory.cumsum() % 2

    # Only trajectories with enough jumps can be an A, B, A, B pattern. For
    # those I check that each side, taken on its own, is a track an aircraft
    # could really fly (no impossible step from one side-A point to the next).
    cand = df[n_jumps >= MIN_JUMPS_FOR_DUPLICATE].assign(side=side)
    g = cand.groupby(["trajectory_id", "side"])
    step_m = haversine_m(g["lat"].shift(), g["lon"].shift(), cand["lat"], cand["lon"])
    step_s = g["time"].diff()
    step_speed = step_m / step_s.where(step_s > 0)
    broken = (step_speed > IMPOSSIBLE_SPEED_MPS) & (step_m > MIN_JUMP_M)
    not_two_tracks = set(cand.loc[broken, "trajectory_id"])

    alternating = (n_jumps >= MIN_JUMPS_FOR_DUPLICATE) & ~df["trajectory_id"].isin(not_two_tracks)
    return is_jump, alternating


def _bearing_rad(lat1, lon1, lat2, lon2):
    """Direction from point 1 to point 2 in radians (0 = north, clockwise)."""
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return np.arctan2(y, x)


# ---------------------------------------------------------------------------
# The four aircraft checks
# ---------------------------------------------------------------------------

def check_speed(df: pd.DataFrame, thresholds: dict = THRESHOLDS,
                dt_range_s: tuple = CHECK_DT_RANGE_S,
                turn_guard_deg: float = TURN_GUARD_DEG) -> pd.DataFrame:
    """Speed check: is the aircraft faster than its kind can fly, or does its
    reported speed disagree with how far it actually moved?

    Four check names:
      speed_reported   the speed the aircraft SENDS is above the class limit
      speed_implied    the speed its POSITIONS imply is above the class limit
      speed_mismatch   implied and reported speed differ by more than the limit
      position_jump    one impossible jump (more than Mach 3 over more than 5 km)
                       that is not part of an alternating A, B, A, B pattern
    The first three only use reports 5 to 60 s apart, and only steps where
    the aircraft was airborne at both ends (see _airborne_step). A position
    jump is impossible at any time gap, so it is not limited to that window.

    speed_mismatch has one exception: if the aircraft turned by more than
    turn_guard_deg between the two reports, an implied speed BELOW the
    reported speed is expected (see TURN_GUARD_DEG) and is not flagged. An
    implied speed ABOVE the reported speed is still flagged, turn or not.
    """
    df = _prepare(df)
    is_jump, alternating = _find_jumps(df)
    # a jump row is reported once, as a jump, and not again as "too fast"
    normal = _in_dt_window(df, dt_range_s) & ~is_jump & _airborne_step(df)
    max_speed = _limit(df, "max_speed_mps", thresholds)
    max_mismatch = _limit(df, "max_speed_mismatch_mps", thresholds)
    # how much the heading changed between the previous report and this one
    heading_change = (df["turn_rate_dps"] * df["dt_s"]).abs()
    explained_by_turn = (heading_change > turn_guard_deg) & (df["speed_mismatch_mps"] < 0)
    return pd.concat([
        _flags(df, normal & (df["velocity_mps"] > max_speed), "speed_reported", df["velocity_mps"], max_speed),
        _flags(df, normal & (df["implied_speed_mps"] > max_speed), "speed_implied", df["implied_speed_mps"], max_speed),
        _flags(df, normal & ~explained_by_turn & (df["speed_mismatch_mps"].abs() > max_mismatch), "speed_mismatch",
               df["speed_mismatch_mps"], max_mismatch),
        _flags(df, is_jump & ~alternating, "position_jump", df["implied_speed_mps"], IMPOSSIBLE_SPEED_MPS),
    ], ignore_index=True)


def check_duplicate_icao(df: pd.DataFrame) -> pd.DataFrame:
    """Duplicate ICAO check: two aircraft sending the same icao24 at the same time.

    Each address should belong to one aircraft. If two transmitters use the
    same one, their reports get mixed into one track that jumps from aircraft
    A to aircraft B and back: A, B, A, B. I flag a trajectory when it has at
    least three impossible jumps AND the A reports and the B reports each form
    a track that can really be flown. A single impossible jump is not enough:
    that is reported as position_jump by the speed check.
    """
    df = _prepare(df)
    is_jump, alternating = _find_jumps(df)
    return _flags(df, is_jump & alternating, "duplicate_icao", df["implied_speed_mps"], IMPOSSIBLE_SPEED_MPS)


def check_changes(df: pd.DataFrame, thresholds: dict = THRESHOLDS,
                  dt_range_s: tuple = CHECK_DT_RANGE_S) -> pd.DataFrame:
    """Altitude and speed change check: does the aircraft climb, descend, speed
    up or slow down faster than its kind can?

    Two check names:
      climb_rate     change in barometric altitude per second (up or down)
      acceleration   change in reported speed per second (up or down)
    Only reports 5 to 60 s apart are used, and only steps where the aircraft
    was airborne at both ends (see _airborne_step). Jump rows are left out:
    they compare two positions that do not belong together, so their rates
    mean nothing.
    """
    df = _prepare(df)
    is_jump, _ = _find_jumps(df)
    normal = _in_dt_window(df, dt_range_s) & ~is_jump & _airborne_step(df)
    max_alt_rate = _limit(df, "max_alt_rate_mps", thresholds)
    max_accel = _limit(df, "max_accel_mps2", thresholds)
    return pd.concat([
        _flags(df, normal & (df["alt_rate_mps"].abs() > max_alt_rate), "climb_rate", df["alt_rate_mps"], max_alt_rate),
        _flags(df, normal & (df["accel_mps2"].abs() > max_accel), "acceleration", df["accel_mps2"], max_accel),
    ], ignore_index=True)


def check_route_deviation(df: pd.DataFrame, thresholds: dict = THRESHOLDS,
                          min_duration_s: float = ROUTE_MIN_DURATION_S,
                          min_length_m: float = ROUTE_MIN_LENGTH_M) -> pd.DataFrame:
    """Route deviation check: does the aircraft stray far from the direct route?

    I have no flight plans, so the "expected route" is the straight line (a
    great circle) from the first to the last report of the trajectory. A
    report is flagged when it is further from that line than the class limit.

    This needs a COMPLETED trajectory, because the line is only known once I
    have the last point. It cannot run on a single live snapshot. The backend
    (feature 10) will apply it to the trajectories it has accumulated.

    A trajectory is only judged if it lasts at least 10 minutes and its start
    and end are at least 50 km apart. That leaves out training flights and
    sightseeing loops that come back to where they started, for which a
    straight line is not the expected route at all. Jets are only judged if
    they stay in cruise (above 6,000 m) the whole time, because arrivals and
    departures do not fly straight. Trajectories with an impossible jump are
    left out too, since their positions cannot be trusted.
    """
    df = _prepare(df)
    is_jump, _ = _find_jumps(df)
    g = df.groupby("trajectory_id")
    # first and last point of each trajectory, repeated on every one of its rows
    lat0, lon0, t0 = g["lat"].transform("first"), g["lon"].transform("first"), g["time"].transform("first")
    lat1, lon1, t1 = g["lat"].transform("last"), g["lon"].transform("last"), g["time"].transform("last")
    has_jump = is_jump.groupby(df["trajectory_id"]).transform("any")
    # the lowest altitude of the trajectory must be above the class minimum
    # (6,000 m for jets, 0 = no condition for the other classes)
    min_altitude = _limit(df, "route_min_altitude_m", thresholds)
    high_enough = (min_altitude == 0) | (g["baro_altitude_m"].transform("min") >= min_altitude)
    judged = ((t1 - t0 >= min_duration_s)
              & (haversine_m(lat0, lon0, lat1, lon1) >= min_length_m)
              & high_enough
              & ~has_jump)

    # Cross-track distance: how far a point is from the line start -> end.
    # d is the distance start -> point as an angle, and the two bearings tell
    # me how far off the line's direction the point lies.
    d = haversine_m(lat0, lon0, df["lat"], df["lon"]) / EARTH_RADIUS_M
    off_direction = _bearing_rad(lat0, lon0, df["lat"], df["lon"]) - _bearing_rad(lat0, lon0, lat1, lon1)
    deviation_m = np.abs(np.arcsin(np.sin(d) * np.sin(off_direction))) * EARTH_RADIUS_M

    limit = _limit(df, "max_route_deviation_m", thresholds)
    return _flags(df, judged & (deviation_m > limit), "route_deviation", deviation_m, limit)


def run_all_checks(df: pd.DataFrame, thresholds: dict = THRESHOLDS) -> pd.DataFrame:
    """Run the four aircraft checks and return all flags in one table.

    df is a feature table that also has aircraft_class and
    aircraft_class_source (see scripts/build_features.py). Step-based flags
    from an unreliable batch are kept, with is_artifact set to True, so I can
    count them separately from real aircraft anomalies.
    """
    df = _prepare(df)
    flags = pd.concat([
        check_speed(df, thresholds),
        check_duplicate_icao(df),
        check_changes(df, thresholds),
        check_route_deviation(df, thresholds),
    ], ignore_index=True)
    return flags.sort_values(["time", "icao24", "check"]).reset_index(drop=True)


def find_class_mismatches(df: pd.DataFrame, thresholds: dict = THRESHOLDS) -> pd.DataFrame:
    """Find aircraft that do not behave like the class they are registered as.

    Example: an icao24 that the aircraft database lists as a small helicopter
    but that cruises at 217 m/s (422 kt) at 43,000 ft. One odd report could be
    a glitch, so I look at how the aircraft TYPICALLY flies. It is a mismatch
    if either of these is true:
      speed     the median of its reported airborne speeds is above the speed
                limit of its registered class (max_speed_mps)
      altitude  it spends real time above the ceiling of its registered class
                (max_altitude_m). I use the 95th percentile of its altitudes
                and not the maximum, so one wrong altitude value is not enough.

    The usual reason is an outdated database entry. In the United States the
    ICAO address is computed from the N-number, so the address follows the
    registration and not the airframe. When an N-number is given to another
    aircraft, the address goes with it, and an old database entry for that
    address now points at a different aircraft.

    Give this function ALL the reports you have of each aircraft. I first ran
    it on one hour at a time and it missed aircraft: in a single hour an
    aircraft can be climbing or descending the whole time, so its median speed
    in that hour is below the limit even though over the whole dataset it is
    clearly too fast. One aircraft was a mismatch in one file and "normal" in
    another, and got speed flags in the second.

    Only classes that come from the database are checked (also the ones
    already marked "mismatch"), because the other sources are guesses.

    Returns one row per aircraft: icao24, callsign, aircraft_class (the
    registered one), reason, median_speed_mps, max_speed_mps, high_altitude_m
    (the 95th percentile), max_altitude_m, median_altitude_m, reports.
    """
    summary = _aircraft_summary(df[df["aircraft_class_source"].isin(["database", "mismatch"])], thresholds)
    return summary[summary["reason"] != ""].reset_index(drop=True)


def list_marked_mismatches(df: pd.DataFrame, thresholds: dict = THRESHOLDS) -> pd.DataFrame:
    """The aircraft that are MARKED as mismatches in df, with their numbers in df.

    Same columns as find_class_mismatches. The difference: this does not
    decide anything, it reports a decision that was already made (maybe on
    more data than df holds). reason is "" when df alone would not show it.
    """
    return _aircraft_summary(df[df["aircraft_class_source"] == "mismatch"], thresholds)


def _aircraft_summary(rows: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Typical speed and altitude of each aircraft, compared with its class limits."""
    rows = rows[~rows["on_ground"]]
    per_aircraft = rows.groupby("icao24").agg(
        callsign=("callsign", "first"),
        aircraft_class=("aircraft_class", "first"),
        median_speed_mps=("velocity_mps", "median"),
        median_altitude_m=("baro_altitude_m", "median"),
        high_altitude_m=("baro_altitude_m", lambda alt: alt.quantile(0.95)),
        reports=("time", "size"),
    ).reset_index()
    for name in ("max_speed_mps", "max_altitude_m"):
        per_aircraft[name] = per_aircraft["aircraft_class"].map(
            {c: limits[name] for c, limits in thresholds.items()})
    too_fast = per_aircraft["median_speed_mps"] > per_aircraft["max_speed_mps"]
    too_high = per_aircraft["high_altitude_m"] > per_aircraft["max_altitude_m"]
    per_aircraft["reason"] = np.select([too_fast & too_high, too_fast, too_high],
                                       ["speed and altitude", "speed", "altitude"], default="")
    return per_aircraft.sort_values("median_speed_mps", ascending=False).reset_index(drop=True)


def apply_class_mismatches(df: pd.DataFrame, mismatched_icao24) -> pd.DataFrame:
    """Write a mismatch decision into a table: one decision per aircraft.

    Aircraft in mismatched_icao24 that have a database class get the source
    "mismatch". Aircraft marked "mismatch" earlier that are NOT in the list go
    back to "database", so an older decision made on less data does not stay.

    The registered class stays in aircraft_class so I can still report it, but
    the checks then judge the aircraft with the unknown limits (see
    UNTRUSTED_CLASS_SOURCES). Without this, an airliner that the database
    calls a helicopter gets a "too fast" flag on every single report. The
    finding is one fact about the aircraft (the database is wrong about it),
    so it is reported once, in its own list, and not as hundreds of speed flags.
    """
    from_database = df["aircraft_class_source"].isin(["database", "mismatch"])
    listed = df["icao24"].isin(list(mismatched_icao24))
    source = df["aircraft_class_source"].copy()
    source[from_database & listed] = "mismatch"
    source[from_database & ~listed] = "database"
    return df.assign(aircraft_class_source=source)


def mark_class_mismatches(df: pd.DataFrame, thresholds: dict = THRESHOLDS) -> pd.DataFrame:
    """Find the class mismatches in df and mark them, in one step.

    Use this when df holds everything I know about its aircraft (all the live
    snapshots, for example). For the historical dataset, which is stored as
    one file per hour, scripts/mark_mismatches.py makes the decision over all
    the files together.
    """
    return apply_class_mismatches(df, find_class_mismatches(df, thresholds)["icao24"])
