#!/usr/bin/env python3
"""Regenerate every numerical table and result plot from the frozen snapshot."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
SUITES = [("libero_spatial", "Spatial"), ("libero_object", "Object"),
          ("libero_goal", "Goal"), ("libero_10", "Long")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})


def escape(text):
    table = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
             "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
    return "".join(table.get(c, c) for c in str(text))


def failure_group(row):
    if row["success"]:
        return "External success"
    if row["policy_status"] == "succeeded":
        return "Controller completion / external failure"
    if row["policy_status"] == "timeout":
        return "Global action budget"
    text = (row["failure"] or "").lower()
    if "timed out" in text:
        return "Phase timeout"
    if any(word in text for word in ("retention", "grasp", "lift")):
        return "Grasp / retention rejection"
    if any(word in text for word in ("verification", "sensor")):
        return "Sensor verification rejection"
    return "Other controller failure"


def main():
    data = (PAPER / "data/episodes.jsonl").read_bytes()
    meta = json.loads((PAPER / "data/provenance.json").read_text())
    task_refresh = meta.get("evaluation_kind") == "task_replacement"
    assert hashlib.sha256(data).hexdigest() == meta["episodes_projection_sha256"]
    rows = [json.loads(line) for line in data.splitlines()]
    assert len(rows) == len({r["episode_id"] for r in rows}) == meta["recorded_episodes"]
    if meta["complete"]:
        assert len(rows) == 400
        assert all(sum(r["suite"] == s and r["task_id"] == t for r in rows) == 10
                   for s, _ in SUITES for t in range(10))
    generated = PAPER / "generated"
    figures = PAPER / "figures"
    generated.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    n = len(rows)
    successes = sum(r["success"] for r in rows)
    completed_false = sum(r["policy_status"] == "succeeded" and not r["success"] for r in rows)
    stopped_true = sum(r["policy_status"] != "succeeded" and r["success"] for r in rows)
    fresh = [r for r in rows if not r["reused"]]
    summaries = []
    for suite, label in SUITES:
        group = [r for r in rows if r["suite"] == suite]
        times = [r["elapsed_s"] for r in group if task_refresh or not r["reused"]]
        summaries.append({"suite": suite, "label": label, "n": len(group),
            "successes": sum(r["success"] for r in group),
            "rate": 100 * sum(r["success"] for r in group) / len(group),
            "mean_steps": statistics.mean(r["steps"] for r in group),
            "median_elapsed_s": statistics.median(times),
            "all_success_tasks": sum(all(r["success"] for r in group if r["task_id"] == t)
                  and sum(r["task_id"] == t for r in group) == 10 for t in range(10))})
    failures = collections.Counter(failure_group(r) for r in rows if not r["success"])
    stats = {"episodes": n, "successes": successes, "rate": 100 * successes / n,
        "evaluation_kind": meta.get("evaluation_kind", "full_campaign"),
        "per_suite": summaries, "policy_external_disagreement": {
            "controller_completed_external_failed": completed_false,
            "external_success_without_controller_completion": stopped_true},
        "failure_groups": dict(failures),
        "new_episodes": len(fresh), "reused_episodes": n - len(fresh),
        "new_successes": sum(r["success"] for r in fresh),
        "reused_successes": sum(r["success"] for r in rows if r["reused"]),
        "new_median_elapsed_s": statistics.median(r["elapsed_s"] for r in fresh),
        "new_total_episode_elapsed_s": sum(r["elapsed_s"] for r in fresh)}
    (PAPER / "data/statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    macros = {
        "EpisodeCount": n, "SuccessCount": successes, "SuccessRate": f"{100*successes/n:.1f}",
        "FailureCount": n-successes, "ControllerFalsePositive": completed_false,
        "ControllerMissedSuccess": stopped_true, "DisagreementCount": completed_false+stopped_true,
        "DisagreementRate": f"{100*(completed_false+stopped_true)/n:.1f}",
        "FreshCount": len(fresh), "FreshSuccessCount": sum(r["success"] for r in fresh),
        "ReusedCount": n-len(fresh),
        "FreshSuccessRate": f"{100*sum(r['success'] for r in fresh)/len(fresh):.1f}",
        "FreshMedianSeconds": f"{stats['new_median_elapsed_s']:.1f}",
        "RunMinutes": f"{meta['wall_elapsed_s']/60:.1f}",
        "RuntimeExceptionCount": sum(r["policy_status"] == "exception" for r in rows),
        "PerfectTaskCount": sum(s["all_success_tasks"] for s in summaries),
        "BaselineSuccessCount": meta.get("task_replacement", {}).get("baseline_total_successes", successes),
    }
    for s in summaries:
        macros[s["label"] + "Rate"] = f"{s['rate']:.1f}"
        macros[s["label"] + "MeanSteps"] = f"{s['mean_steps']:.1f}"
    for name, suite, task in [("TopDrawerSuccess", "libero_goal", 3),
                              ("DrawerPickSuccess", "libero_spatial", 4),
                              ("BottomDrawerSuccess", "libero_10", 3),
                              ("MokaSuccess", "libero_10", 8),
                              ("MicrowaveSuccess", "libero_10", 9)]:
        macros[name] = sum(r["success"] for r in rows if r["suite"] == suite and r["task_id"] == task)
    (generated / "numbers.tex").write_text(
        "% Generated from data/episodes.jsonl; do not edit manually.\n"
        + r"\newif\ifRunComplete" + "\n"
        + (r"\RunCompletetrue" if meta["complete"] else r"\RunCompletefalse") + "\n"
        + r"\newif\ifAllFresh" + "\n"
        + (r"\AllFreshtrue" if len(fresh) == n else r"\AllFreshfalse") + "\n"
        + r"\newif\ifTaskRefresh" + "\n"
        + (r"\TaskRefreshtrue" if task_refresh else r"\TaskRefreshfalse") + "\n"
        + "".join(r"\newcommand{\%s}{%s}" % (key,value) + "\n" for key,value in macros.items()))
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Suite & Success & SR (\%) & Mean steps & Median s$^\dagger$ \\",
             r"\midrule"]
    for s in summaries:
        lines.append(f"{s['label']} & {s['successes']}/{s['n']} & {s['rate']:.1f} & "
                     f"{s['mean_steps']:.1f} & {s['median_elapsed_s']:.1f}" + r" \\")
    lines += [r"\midrule", f"Overall & {successes}/{n} & {100*successes/n:.1f} & "
              f"{statistics.mean(r['steps'] for r in rows):.1f} & "
              f"{statistics.median(r['elapsed_s'] for r in (rows if task_refresh else fresh)):.1f}" + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    (generated / "suite_table.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{tabular}{lrr}",r"\toprule",
        r"Controller terminal state & External success & External failure \\",
        r"\midrule"]
    for state in ["succeeded","failed","timeout","exception"]:
        group = [r for r in rows if r["policy_status"] == state]
        if group:
            lines.append(f"{state.capitalize()} & {sum(r['success'] for r in group)} & "
                         f"{sum(not r['success'] for r in group)}" + r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (generated / "status_table.tex").write_text("\n".join(lines)+"\n")
    lines = [r"\begin{tabular}{lr}",r"\toprule",r"Recorded outcome group & Episodes \\",
             r"\midrule"]
    for label, count in failures.most_common():
        lines.append(f"{escape(label)} & {count}" + r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (generated / "failure_table.tex").write_text("\n".join(lines)+"\n")
    with (PAPER / "data/per_task.csv").open("w") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["suite","task_id","episodes","successes","instruction"])
        for suite,_ in SUITES:
            for task in range(10):
                group=[r for r in rows if r["suite"]==suite and r["task_id"]==task]
                writer.writerow([suite,task,len(group),sum(r["success"] for r in group),
                                 group[0]["instruction"] if group else ""])
    values=np.full((4,10),np.nan)
    for i,(suite,_) in enumerate(SUITES):
        for t in range(10):
            group=[r for r in rows if r["suite"]==suite and r["task_id"]==t]
            if group: values[i,t]=sum(r["success"] for r in group)/len(group)
    fig,ax=plt.subplots(figsize=(6.8,1.95),layout="constrained")
    ax.imshow(values,cmap="Blues",vmin=0,vmax=1,aspect="auto")
    for i in range(4):
        for t in range(10):
            group=[r for r in rows if r["suite"]==SUITES[i][0] and r["task_id"]==t]
            if group:
                ax.text(t,i,f"{sum(r['success'] for r in group)}/{len(group)}",
                        ha="center",va="center",color="white" if values[i,t]>.65 else "#132d43")
    ax.set_xticks(range(10),[f"{i:02d}" for i in range(10)])
    ax.set_yticks(range(4),[label for _,label in SUITES])
    ax.set_xlabel("Official task index (within each suite)")
    ax.tick_params(length=0)
    fig.savefig(figures/"per_task.pdf",bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(stats,indent=2))


if __name__ == "__main__":
    main()
