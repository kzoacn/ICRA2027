#!/usr/bin/env python3
"""Build cited policy comparisons and model-size tables from recorded evidence."""
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
    lines.append(r"\method{} & Two RGB-D views + state; asset priors, geometric skills & "
                 + " & ".join(f"{value:.1f}" for value in rates) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    return "\n".join(lines) + "\n"


def render_model_artifacts():
    published = json.loads((PAPER / "data/published_comparisons.json").read_text())
    audit = json.loads((PAPER / "data/model_audit.json").read_text())
    implementation = PAPER.parent / "route-b-v170-cloud"
    lock = json.loads((implementation / "resources.lock.json").read_text())["model"]
    assert audit["schema"] == 1 and audit["checkpoint"] == lock
    source = PAPER.parent / audit["implementation"]["file"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == audit["implementation"]["sha256"]
    count = audit["parameters"]
    assert count > 0 and audit["runtime_trainable_parameters"] == 0
    assert audit["parameter_elements_by_dtype"] == {"torch.float32": count}
    assert audit["parameter_bytes"] == 4 * count
    assert audit["weight_file"] == next(item for item in lock["files"]
                                      if item["path"] == "model.safetensors")
    sizes = {key: published["sources"][key]["model_size"]
             for key in ["smolvla", "openvla", "openvlaoft"]}
    numbers = {
        "NeuralParameters": f"{count:,}",
        "NeuralMillion": f"{count / 1e6:.0f}",
        "NeuralBillion": f"{count / 1e9:.3f}",
        "DetectorWeightMB": f"{audit['weight_file']['size'] / 1e6:.0f}",
        "SmolParameterRatio": f"{sizes['smolvla']['parameters'] / count:.1f}",
        "OpenVLAParameterRatio": f"{sizes['openvla']['parameters'] / count:.1f}",
    }
    macros = "% Generated from model_audit.json and published_comparisons.json; do not edit.\n"
    macros += "".join("\\newcommand{\\" + name + "}{" + value + "}\n"
                      for name, value in numbers.items())
    lines = [
        "% Generated from model_audit.json and published_comparisons.json; do not edit.",
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrc@{}}",
        r"\toprule",
        r"Method & Parameters & Policy training \\",
        r"\midrule",
    ]
    for label, key in [("SmolVLA", "smolvla"), ("OpenVLA", "openvla"),
                       ("OpenVLA-OFT", "openvlaoft")]:
        size = sizes[key]
        assert size["parameters"] > 0 and size["robot_demonstrations_for_action_policy"] is True
        lines.append(label + r"~\cite{" + key + "} & "
                     + f"{size['parameters'] / 1e9:g}B" + r" & Robot demos \\")
    lines += [r"\midrule", r"\method{} & \NeuralBillion{}B & None \\",
              r"\bottomrule", r"\end{tabular*}"]
    return {"model_table.tex": "\n".join(lines) + "\n", "model_numbers.tex": macros}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check that the saved table is current.")
    args = parser.parse_args()
    artifacts = {"comparison_table.tex": render_table(), **render_model_artifacts()}
    for name, rendered in artifacts.items():
        path = PAPER / "generated" / name
        if args.check:
            assert path.read_text() == rendered, f"Generated artifact is stale: {name}"
        else:
            path.write_text(rendered)
            print(f"Wrote {path.relative_to(PAPER)}")
    if args.check:
        print("Comparison tables match cited values, measured episodes, and the model audit.")


if __name__ == "__main__":
    main()
