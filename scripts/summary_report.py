"""Feature 16: the final numbers, tables and figure for my paper.

    python scripts/summary_report.py

It reads what features 8 and 9 saved (data/models/, data/report/) and writes
to data/report/:
    metrics.json             every number, in one place
    table_confusion.tex      per-flight confusion matrix on the test days
    table_per_class.tex      the same comparison per aircraft class
    table_flight_length.tex  flagged rate of each detector by flight length
    table_verdicts.tex       my manual verdicts on the disagreements, per detector
    score_distribution.png / .pdf  forest scores of rule-anomalous and
                             rule-normal flights, with the threshold
It retrains nothing and recomputes nothing: it only collects and formats.
All comparisons are against my rules' weak labels, not ground truth.
"""
import json

import matplotlib
matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _path  # noqa: F401  (lets me import adsb from here)
from adsb.config import MODELS_DIR, REPORT_DIR
from adsb.report import NOT_JUDGED, VERDICTS, latex_table, pct, verdict_counts

# figure colors: the first two slots of a colorblind-checked categorical palette
NORMAL_COLOR = "#2a78d6"     # blue: rule-normal flights
ANOMALOUS_COLOR = "#eb6834"  # orange: rule-anomalous flights
INK = "#0b0b0b"              # text
MUTED_INK = "#52514e"        # threshold line and secondary text
N_BINS = 40


def score_figure(flights: pd.DataFrame, threshold: float, path_stem):
    """Two panels sharing the score axis: rule-normal flights on top, rule-anomalous below.

    The two groups are very different in size (thousands against a few
    dozen), so I give each its own panel and its own count axis instead of
    putting both on one, where the small group would be invisible.
    """
    scored = flights[flights["scored"]]
    bins = np.linspace(scored["max_score"].min(), scored["max_score"].max(), N_BINS + 1)
    groups = [(False, "Flights the rules call normal", NORMAL_COLOR),
              (True, "Flights the rules flag (report-level checks)", ANOMALOUS_COLOR)]
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    for ax, (anomalous, title, color) in zip(axes, groups):
        values = scored.loc[scored["rules_anomalous"] == anomalous, "max_score"]
        ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.8)
        ax.axvline(threshold, color=MUTED_INK, linestyle="--", linewidth=1.2)
        above = int((values >= threshold).sum())
        ax.set_title(f"{title} (n = {len(values):,}; {above} at or above the threshold)",
                     fontsize=10, color=INK, loc="left")
        ax.set_ylabel("Flights", color=INK)
        ax.grid(axis="y", alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].text(threshold, axes[0].get_ylim()[1] * 0.95, f"  threshold {threshold:.4f}",
                 color=MUTED_INK, fontsize=9, va="top")
    axes[-1].set_xlabel("Flight score (highest isolation forest score of its reports)", color=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    settings = json.loads((MODELS_DIR / "iforest_settings.json").read_text())
    comparison = json.loads((REPORT_DIR / "comparison.json").read_text())
    flights = pd.read_parquet(REPORT_DIR / "flights_test.parquet")
    disagreements = pd.read_csv(REPORT_DIR / "disagreements.csv", dtype=str, keep_default_na=False)
    verdicts = verdict_counts(disagreements)
    c = comparison["confusion"]

    # ---- metrics.json: everything in one file
    metrics = {
        "note": "Comparison against the weak labels of my rule-based checks, not against ground truth.",
        "model": {key: settings[key] for key in [
            "features", "n_estimators", "n_estimators_grid", "max_samples", "random_state", "threshold",
            "threshold_criterion", "n_estimators_criterion", "overlaps", "budget_share", "budget_flights",
            "validation_scorable_flights", "training_days", "validation_days", "test_days",
            "reports_used", "reports_left_out_missing_input", "flights_per_split"]},
        "test_comparison": comparison,
        "verdicts": {
            "per_detector": verdicts.to_dict(orient="index"),
            "total": verdicts.sum().to_dict(),
            "definitions": {
                "genuine anomaly": "the aircraft's behavior is really unusual or potentially unsafe",
                "equipment fault": "the data is physically impossible (position jump, frozen value, altitude step)",
                "false alarm": "the flight is normal even if it is rare for the model",
            },
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=int) + "\n")

    # ---- LaTeX tables
    confusion = latex_table(
        ["", "Forest flagged", "Forest not flagged", "Total"],
        [["Rules anomalous", c["both"], c["rules_only"], c["both"] + c["rules_only"]],
         ["Rules normal", c["forest_only"], f"{c['neither']:,}", f"{c['forest_only'] + c['neither']:,}"],
         ["Total", c["both"] + c["forest_only"], f"{c['rules_only'] + c['neither']:,}",
          f"{comparison['scored']:,}"]])
    (REPORT_DIR / "table_confusion.tex").write_text(confusion)

    per_class = comparison["per_class"]
    class_rows = [[cls, f"{v['scored_flights']:,}", v["both"], v["rules_only"], v["forest_only"],
                   f"{v['neither']:,}", pct(v["precision"]), pct(v["recall"]), pct(v["rule_normal_flagged"])]
                  for cls, v in per_class.items()]
    class_rows.append(["all", f"{comparison['scored']:,}", c["both"], c["rules_only"], c["forest_only"],
                       f"{c['neither']:,}", pct(c["precision"]), pct(c["recall"]), pct(c["rule_normal_flagged"])])
    (REPORT_DIR / "table_per_class.tex").write_text(latex_table(
        ["Class", "Flights", "Both", "Rules only", "Forest only", "Neither",
         "Precision", "Recall", "Normal flagged"], class_rows))

    length_rows = [[g["group"], f"{g['min_reports']} to {g['max_reports']}", f"{g['flights']:,}",
                    pct(g["rules_flagged"]), pct(g["forest_flagged"])] for g in comparison["by_length"]]
    (REPORT_DIR / "table_flight_length.tex").write_text(latex_table(
        ["Quartile", "Scored reports", "Flights", "Rules flagged", "Forest flagged"], length_rows))

    verdict_header = ["Flagged by"] + [v.capitalize() for v in VERDICTS]
    verdict_columns = VERDICTS + ([NOT_JUDGED] if verdicts[NOT_JUDGED].sum() else [])
    if verdicts[NOT_JUDGED].sum():
        verdict_header.append("Not judged")
    verdict_rows = [[det.capitalize()] + [int(verdicts.loc[det, v]) for v in verdict_columns]
                    for det in verdicts.index]
    verdict_rows.append(["Total"] + [int(verdicts[v].sum()) for v in verdict_columns])
    (REPORT_DIR / "table_verdicts.tex").write_text(latex_table(verdict_header + ["Total"],
        [row + [sum(row[1:])] for row in verdict_rows]))

    # ---- the figure
    score_figure(flights, settings["threshold"], REPORT_DIR / "score_distribution")

    # ---- summary on screen
    print(f"Test days {', '.join(settings['test_days'])}: {comparison['flights']:,} flights, "
          f"{comparison['scored']:,} scored, {comparison['not_scored']} not scored.")
    print(f"Forest: {settings['n_estimators']} trees, threshold {settings['threshold']:.4f}.")
    print(f"Per flight: both {c['both']}, rules only {c['rules_only']}, forest only {c['forest_only']}, "
          f"neither {c['neither']:,}.")
    print(f"Against the weak labels: precision {pct(c['precision'])}, recall {pct(c['recall'])}, "
          f"rule-normal flights flagged {pct(c['rule_normal_flagged'])}.")
    print("\nMy verdicts on the disagreements:")
    print(verdicts.to_string())
    if verdicts[NOT_JUDGED].sum():
        print(f"WARNING: {int(verdicts[NOT_JUDGED].sum())} disagreements have no verdict yet.")
    print(f"\nSaved to {REPORT_DIR}: metrics.json, table_confusion.tex, table_per_class.tex, "
          "table_flight_length.tex, table_verdicts.tex, score_distribution.png and .pdf")


if __name__ == "__main__":
    main()
