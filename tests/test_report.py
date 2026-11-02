"""Tests for feature 16. Run them from the repo root with: python -m pytest"""
import pandas as pd
import pytest

from adsb.report import NOT_JUDGED, latex_escape, latex_table, pct, verdict_counts


def disagreements(verdicts):
    flagged_by = ["forest only"] * 3 + ["rules only"] * (len(verdicts) - 3)
    return pd.DataFrame({"flagged_by": flagged_by, "my_verdict": verdicts})


def test_verdicts_are_counted_per_detector():
    table = verdict_counts(disagreements(["false alarm", "Genuine anomaly ", "false alarm",
                                          "equipment fault", "equipment fault"]))
    assert table.loc["forest only", "false alarm"] == 2
    assert table.loc["forest only", "genuine anomaly"] == 1   # case and spaces do not matter
    assert table.loc["rules only", "equipment fault"] == 2
    assert table.loc["rules only", "genuine anomaly"] == 0
    assert table.to_numpy().sum() == 5


def test_empty_verdict_is_counted_as_not_judged():
    table = verdict_counts(disagreements(["false alarm", "", "false alarm", ""]))
    assert table.loc["forest only", NOT_JUDGED] == 1
    assert table.loc["rules only", NOT_JUDGED] == 1


def test_misspelled_verdict_stops_the_report():
    with pytest.raises(ValueError):
        verdict_counts(disagreements(["false alarm", "flase alarm", "false alarm", "false alarm"]))


def test_latex_table_is_plain_tabular_with_hline():
    tex = latex_table(["Class", "Recall"], [["light", "34.5%"], ["jet", "0.0%"]])
    assert tex.startswith("\\begin{tabular}{lr}\n\\hline\n")
    assert tex.count("\\hline") == 3
    assert "toprule" not in tex and "midrule" not in tex   # no booktabs
    assert "light & 34.5\\% \\\\" in tex                    # % escaped, row ended
    assert tex.endswith("\\end{tabular}\n")


def test_latex_escape_and_pct():
    assert latex_escape("rules_only & 50%") == "rules\\_only \\& 50\\%"
    assert pct(0.3226) == "32.3%"
    assert pct(float("nan")) == "n/a"
