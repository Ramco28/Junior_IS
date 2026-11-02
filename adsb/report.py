"""Helpers for the final summary report (feature 16).

Turning my results into what goes in my paper: verdict counts from my manual
inspection, and LaTeX table snippets. The snippets are plain tabular with
\\hline (no booktabs), so they compile in any LaTeX template without extra
packages.
"""
import pandas as pd

# the three verdicts I give a disagreeing flight (definitions in my notes)
VERDICTS = ["genuine anomaly", "equipment fault", "false alarm"]
NOT_JUDGED = "not judged"
DETECTORS = ["forest only", "rules only"]


def verdict_counts(disagreements: pd.DataFrame) -> pd.DataFrame:
    """Count my verdicts per detector: one row per "flagged_by", one column per verdict.

    An empty my_verdict is counted as "not judged", so a half-finished file is
    visible in the numbers instead of silently shrinking them. Any other text
    is a typo and stops the report: a misspelled verdict must not be dropped.
    """
    verdict = disagreements["my_verdict"].fillna("").astype(str).str.strip().str.lower()
    unknown = sorted(set(verdict) - set(VERDICTS) - {""})
    if unknown:
        raise ValueError(f"Unknown verdicts in my_verdict: {unknown}. Use one of {VERDICTS}.")
    verdict = verdict.replace("", NOT_JUDGED)
    table = pd.crosstab(disagreements["flagged_by"], verdict)
    return table.reindex(index=DETECTORS, columns=VERDICTS + [NOT_JUDGED], fill_value=0)


def latex_escape(text) -> str:
    """Make a value safe inside a LaTeX table cell (& % _ # $ have special meanings)."""
    text = str(text)
    for char in ["\\", "&", "%", "$", "#", "_", "{", "}"]:
        text = text.replace(char, "\\" + char if char != "\\" else "\\textbackslash{}")
    return text


def latex_table(header: list, rows: list, align: str = None) -> str:
    """A plain tabular: one \\hline above and below the header, one at the end.

    header is a list of column titles, rows a list of lists of cell values
    (already formatted as text). align is the LaTeX column spec; by default
    the first column is left-aligned and the others right-aligned.
    """
    align = align or "l" + "r" * (len(header) - 1)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\hline",
             " & ".join(latex_escape(h) for h in header) + " \\\\", "\\hline"]
    for row in rows:
        lines.append(" & ".join(latex_escape(cell) for cell in row) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def pct(x, digits: int = 1) -> str:
    """A share as a percentage for a table, or n/a when it could not be computed."""
    return "n/a" if pd.isna(x) else f"{100 * x:.{digits}f}%"
