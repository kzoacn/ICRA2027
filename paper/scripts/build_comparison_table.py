#!/usr/bin/env python3
"""Combine cited literature means with measured ANCHOR episode outcomes."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def tex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                    "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(char, char) for char in value)


def render_table():
    published = json.loads((PAPER / "data/published_comparisons.json").read_text())
    assert published["column_order"] == ["Spatial", "Object", "Goal", "Long", "Average"]
    data = (PAPER / "data/episodes.jsonl").read_bytes()
    provenance = json.loads((PAPER / "data/provenance.json").read_text())
    assert hashlib.sha256(data).hexdigest() == provenance["episodes_projection_sha256"]
    assert provenance["complete"]
    episodes = [json.loads(line) for line in data.splitlines()]
    assert len(episodes) == len({r["episode_id"] for r in episodes}) == 400
    coverage = collections.Counter((r["suite"], r["task_id"]) for r in episodes)
    assert set(coverage) == {(suite, task) for suite in SUITES for task in range(10)}
    assert set(coverage.values()) == {10}
    assert all(type(r["success"]) is bool for r in episodes)
    lines = [
        "% Generated from published_comparisons.json and episodes.jsonl; do not edit.",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lp{.36\textwidth}rrrrr@{}}",
        r"\toprule",
        r"Method & Sensing and action-policy setup & Spatial & Object & Goal & Long & Average \\",
        r"\midrule",
        r"\multicolumn{7}{@{}l}{\emph{Published results (reported means)}} \\",
    ]
    for row in published["rows"]:
        assert row["source"] in published["sources"]
        assert len(row["values"]) == 5 and all(0 <= x <= 100 for x in row["values"])
        # Published averages can differ slightly from the mean of rounded columns.
        assert abs(sum(row["values"][:4]) / 4 - row["values"][4]) <= 0.11
        means = " & ".join(f"{value:.1f}" for value in row["values"])
        label = tex_escape(row["method"]) + r"~\cite{" + row["source"] + "}"
        lines.append(f"{label} & {tex_escape(row['setup'])} & {means}" + r" \\")
    lines += [r"\midrule", r"\multicolumn{7}{@{}l}{\emph{Measured in this work}} \\"]
    rates = [100 * sum(r["success"] for r in episodes if r["suite"] == suite)
             / sum(r["suite"] == suite for r in episodes) for suite in SUITES]
    rates.append(100 * sum(r["success"] for r in episodes) / len(episodes))
    lines.append(r"\method{} & Two RGB-D views + state; asset priors, fixed controller & "
                 + " & ".join(f"{value:.1f}" for value in rates) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check that the saved table is current.")
    args = parser.parse_args()
    rendered = render_table()
    path = PAPER / "generated/comparison_table.tex"
    if args.check:
        assert path.read_text() == rendered, "Published comparison table is stale."
        print("Comparison table matches cited values and measured episodes.")
    else:
        path.write_text(rendered)
        print(f"Wrote {path.relative_to(PAPER)}")


if __name__ == "__main__":
    main()
