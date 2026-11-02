"""Feature 9 helper: one picture per disagreeing flight, for my manual inspection.

    python scripts/plot_disagreements.py

It reads data/report/disagreements.csv (made by scripts/compare_detectors.py),
the labeled reports and the forest scores, and saves one PNG per flight in
data/report/disagreements/, named <flagged_by>_<rank>_<trajectory_id>.png.

Left: the ground track, with the forest's most anomalous report and every
report flagged by a report-level rule in different colors.
Right, over time: reported and implied speed, barometric altitude, reported
vertical rate, and the forest score with its threshold.
I plot in knots, feet and feet per minute because those are the units I use
as a pilot; the data itself stays in m/s and meters.
"""
import json

import matplotlib
matplotlib.use("Agg")  # no window: the pictures are only saved to files
import matplotlib.pyplot as plt
import pandas as pd

import _path  # noqa: F401  (lets me import adsb from here)
from adsb.compare import SPLIT
from adsb.config import LABELED_DIR, MODELS_DIR, REPORT_DIR

KT_PER_MPS = 1.944    # knots in one m/s
FT_PER_M = 3.281      # feet in one meter
FPM_PER_MPS = 196.85  # feet per minute in one m/s
OUT_DIR = REPORT_DIR / "disagreements"
COLUMNS = ["trajectory_id", "time", "lat", "lon", "velocity_mps", "implied_speed_mps",
           "baro_altitude_m", "vertical_rate_mps", "label", "checks"]
TOP_COLOR = "tab:red"      # the forest's most anomalous report
RULE_COLOR = "tab:orange"  # reports flagged by a report-level rule


def plot_flight(flight: pd.Series, rows: pd.DataFrame, threshold: float, path):
    """Draw one disagreeing flight and save it to path."""
    rows = rows.sort_values("time")
    t = pd.to_datetime(rows["time"], unit="s")
    rule = rows["label"] == "anomaly"
    top = rows["score"].idxmax() if rows["score"].notna().any() else None

    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(4, 2, width_ratios=[1, 1.4], hspace=0.35, wspace=0.2)

    # left: where the aircraft flew
    track = fig.add_subplot(grid[:, 0])
    track.plot(rows["lon"], rows["lat"], color="0.6", linewidth=1, zorder=1)
    track.scatter(rows["lon"].iloc[0], rows["lat"].iloc[0], marker="^", color="tab:green", s=60,
                  label="first report", zorder=3)
    track.scatter(rows.loc[rule, "lon"], rows.loc[rule, "lat"], color=RULE_COLOR, s=30,
                  label="rule flag", zorder=4)
    if top is not None:
        track.scatter(rows.loc[top, "lon"], rows.loc[top, "lat"], color=TOP_COLOR, marker="*", s=200,
                      label="highest forest score", zorder=5)
    track.set_xlabel("Longitude")
    track.set_ylabel("Latitude")
    track.set_aspect("equal", adjustable="datalim")
    track.grid(alpha=0.3)
    track.legend(loc="best", fontsize=8)

    # right: the same flight over time, one panel per quantity
    panels = [
        ("Speed (kt)", [("reported", rows["velocity_mps"] * KT_PER_MPS),
                        ("implied by positions", rows["implied_speed_mps"] * KT_PER_MPS)]),
        ("Baro altitude (ft)", [("", rows["baro_altitude_m"] * FT_PER_M)]),
        ("Vertical rate (ft/min)", [("", rows["vertical_rate_mps"] * FPM_PER_MPS)]),
        ("Forest score", [("", rows["score"])]),
    ]
    axes = []
    for i, (ylabel, series) in enumerate(panels):
        ax = fig.add_subplot(grid[i, 1], sharex=axes[0] if axes else None)
        for label, values in series:
            ax.plot(t, values, linewidth=1, label=label or None)
        # mark the same reports as on the map, in every panel
        first_values = series[0][1]
        ax.scatter(t[rule], first_values[rule], color=RULE_COLOR, s=12, zorder=4)
        if top is not None:
            ax.scatter(t[top], first_values[top], color=TOP_COLOR, marker="*", s=90, zorder=5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.3)
        if len(series) > 1:
            ax.legend(fontsize=7, loc="best")
        axes.append(ax)
    axes[-1].axhline(threshold, color="black", linestyle="--", linewidth=1, label=f"threshold {threshold:.4f}")
    axes[-1].legend(fontsize=7, loc="best")
    axes[-1].set_xlabel("Time (UTC)")
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    checks = flight["rule_checks"] if isinstance(flight["rule_checks"], str) and flight["rule_checks"] else "none"
    callsign = flight["callsign"] if isinstance(flight["callsign"], str) else "no callsign"
    fig.suptitle(f"{callsign} ({flight['icao24']})   class {flight['aircraft_class']} "
                 f"(source {flight['aircraft_class_source']})   {flight['flagged_by']}\n"
                 f"rule checks: {checks}   max forest score {flight['max_score']:.4f}, "
                 f"rank {flight['score_rank']}   {flight['trajectory_id']}", fontsize=11)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    odd = pd.read_csv(REPORT_DIR / "disagreements.csv", dtype={"icao24": str})
    threshold = json.loads((MODELS_DIR / "iforest_settings.json").read_text())["threshold"]
    labeled = pd.read_parquet(LABELED_DIR / "labeled_rows.parquet", columns=COLUMNS,
                              filters=[("trajectory_id", "in", list(odd["trajectory_id"]))])
    scores = pd.read_parquet(MODELS_DIR / "iforest_scores.parquet", columns=["trajectory_id", "time", "score"],
                             filters=[("split", "==", SPLIT)])
    # every report of the flight, with its score when the forest scored it
    rows = labeled.merge(scores, on=["trajectory_id", "time"], how="left")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    made = 0
    for _, flight in odd.iterrows():
        name = f"{flight['flagged_by'].replace(' ', '_')}_{int(flight['score_rank']):04d}_{flight['trajectory_id']}.png"
        plot_flight(flight, rows[rows["trajectory_id"] == flight["trajectory_id"]], threshold, OUT_DIR / name)
        made += 1
    print(f"Saved {made} pictures in {OUT_DIR}")


if __name__ == "__main__":
    main()
