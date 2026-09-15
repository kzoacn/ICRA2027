#!/usr/bin/env python3
"""Render recorded frames with the camera's vertical display orientation fixed."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_results import escape, failure_group
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.family": "DejaVu Sans"})

PAPER = Path(__file__).resolve().parents[1]


def phase_at(transitions, step):
    eligible = [entry for entry in transitions if entry["step"] <= step]
    return eligible[-1]["phase"] if eligible else "initial"


def phase_label(phase):
    return {"move_pregrasp": "Pregrasp", "move_preplace": "Preplace",
            "done": "Complete", "failed": "Stopped"}.get(phase, phase.replace("_", " ").capitalize())


def main():
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source-root", type=Path)
    inputs.add_argument("--batch-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PAPER,
                        help="Stage generated figures and metadata outside the paper if needed.")
    parser.add_argument("--success-task", default="object_task00")
    parser.add_argument("--failure-task", default="goal_task03")
    args = parser.parse_args()
    source = (args.batch_dir or args.source_root).resolve()
    records_root = (source / "shards" if args.batch_dir else
                    source / "outputs/smolvla_scale_400_parallel_20260913/shards")
    if args.batch_dir:
        status = json.loads((source / "status.json").read_text())
        validation = json.loads((source / "final-validation.json").read_text())
        manifest = json.loads((source / "manifest.json").read_text())
        assert status["state"] == "completed" and validation["coverage_passed"]
        assert not validation["issues"] and manifest["kind"] == "full_400"
    records_by_task = {
        path.parent.name: sorted((json.loads(line) for line in path.read_text().splitlines()),
                                 key=lambda row: row["key"]["episode_index"])
        for path in sorted(records_root.glob("*/episodes.jsonl"))
    }

    def select(preferred, success):
        # Keep the preferred task when possible, then use task/index order.
        # This chooses illustrations only; every episode remains in the statistics.
        order = [preferred] + [task for task in sorted(records_by_task) if task != preferred]
        for task in order:
            for row in records_by_task.get(task, []):
                if row["evaluator_success"] is success:
                    return task, row
        raise ValueError(f"No episode with external success={success} is available to illustrate.")

    selected = [select(args.success_task, True), select(args.failure_task, False)]
    output = args.output_dir.resolve()
    frame_dir = output / "figures/recorded"
    frame_dir.mkdir(parents=True, exist_ok=True)
    (output / "generated").mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(6.8, 3.85))
    provenance = []
    macros = {}
    suite_names = {"libero_spatial": "Spatial", "libero_object": "Object",
                   "libero_goal": "Goal", "libero_10": "Long"}
    for row_index, (task, row) in enumerate(selected):
        init = row["key"]["episode_index"]
        task_label = f"{suite_names[row['key']['suite']]} {row['key']['task_id']:02d}"
        outcome = ("external success" if row["evaluator_success"] else failure_group({
            "success": False, "policy_status": row["policy_status"], "failure": row["failure"]}).lower())
        label = f"{task_label} / init {init:02d}: {outcome}"
        config = json.loads((records_root / task / "summary.json").read_text())["run_config"]
        stride, budget = int(config["video_stride"]), int(config["max_steps"])
        transitions = row["route_trace"]["phase_transitions"]
        assert all(0 <= entry["step"] <= row["steps"] for entry in transitions)
        assert [entry["step"] for entry in transitions] == sorted(entry["step"] for entry in transitions)
        video = Path(row["video_paths"]["dual"])
        reader = imageio.get_reader(video)
        frames = reader.count_frames()
        # _run_b records the reset observation and every executed step;
        # DualViewVideoRecorder writes observation indices 0, stride, ... .
        assert frames == row["steps"] // stride + 1
        indices = np.linspace(0, frames - 1, 4).round().astype(int)
        display_frames = []
        for column, index in enumerate(indices):
            frame = reader.get_data(int(index))
            # The recorder stores the simulator camera's bottom-up raster.
            # Flip only the vertical axis; preserve left/right and every pixel.
            upright = np.ascontiguousarray(frame[::-1, :frame.shape[1] // 2])
            frame_path = frame_dir / f"{task}_init{init:02d}_frame{index:03d}.png"
            imageio.imwrite(frame_path, upright)
            action_step = int(index) * stride
            logged_phase = phase_at(transitions, action_step)
            display_frames.append({"index": int(index), "action_step": action_step,
                "logged_phase": logged_phase,
                "file": str(frame_path.relative_to(output)),
                "sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
                "pixel_sha256": hashlib.sha256(upright.tobytes()).hexdigest()})
            axes[row_index, column].imshow(upright)
            axes[row_index, column].set_title(f"t={action_step}: {phase_label(logged_phase)}", fontsize=7, pad=2)
            axes[row_index, column].axis("off")
        reader.close()
        provenance.append({"episode_id": row["episode_id"], "success": row["evaluator_success"],
            "policy_status": row["policy_status"], "instruction": row["instruction"],
            "steps": row["steps"],
            "video_stride": stride, "action_budget": budget,
            "phase_transitions": transitions,
            "frame_alignment": "Frame index times recording stride; last phase transition at or before that executed-action count.",
            "display_label": label,
            "source_record_sha256": hashlib.sha256(next(
                line for line in (records_root / task / "episodes.jsonl").read_bytes().splitlines()
                if json.loads(line)["episode_id"] == row["episode_id"])).hexdigest(),
            "selection_rule": "First matching external outcome in the preferred task, then sorted task/index order.",
            "video": str(video.relative_to(source)),
            "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "total_recorded_frames": frames, "selected_frames": indices.tolist(),
            "display_frames": display_frames,
            "transformation": "Left half of the original dual-view frame (agentview), then vertical-axis flip to correct display orientation; left/right preserved; no retouching."})
        fig.text(.01, .985 if row_index == 0 else .56, label, fontsize=8, weight="bold")
        prefix = "RolloutSuccess" if row_index == 0 else "RolloutFailure"
        macros.update({prefix + "Task": escape(task_label), prefix + "Init": f"{init:02d}",
                       prefix + "Instruction": escape(row["instruction"]),
                       prefix + "Steps": str(row["steps"])})
        if row_index == 0:
            macros.update(OverviewBeforeImage=display_frames[0]["file"],
                          OverviewAfterImage=display_frames[2]["file"])
        else:
            active = [entry for entry in transitions if entry["step"] < row["steps"]]
            final_active = active[-1] if active else {"step": 0, "phase": "initial"}
            start, stop = final_active["step"], row["steps"]
            phase_steps, unused = stop - start, budget - stop
            assert unused >= 0
            macros.update(RolloutFailurePhaseStart=str(start),
                          RolloutFailurePhaseSteps=str(phase_steps),
                          RolloutFailureUnusedSteps=str(unused))
            provenance[-1]["final_active_phase_interval"] = {
                "phase": final_active["phase"], "start": start, "stop": stop,
                "actions": phase_steps, "unused_actions": unused}
            timeline = fig.add_axes([.065, .073, .91, .05])
            for left, width, color, title in [
                    (0, start, "#d8e6f4", f"Earlier phases: {start}"),
                    (start, phase_steps, "#f3d9a4", f"{phase_label(final_active['phase'])}: {phase_steps}"),
                    (stop, unused, "#e9edf1", f"{unused} unused")]:
                if width:
                    timeline.barh(.5, width, left=left, height=.85, color=color)
                    timeline.text(left + width/2, .5, title, ha="center", va="center", fontsize=6.5, color="#203448")
            timeline.set_xlim(0, budget)
            timeline.set_ylim(0, 1)
            timeline.set_xticks(sorted({0, start, stop, budget}))
            timeline.tick_params(axis="x", labelsize=6.5, length=2)
            timeline.set_yticks([])
            for spine in timeline.spines.values():
                spine.set_visible(False)
            timeline.set_title(f"{task_label} controller trace (actions)", loc="left", fontsize=7, pad=4)
    fig.subplots_adjust(left=.01, right=.99, top=.925, bottom=.19, hspace=.22, wspace=.03)
    fig.savefig(output / "figures/rollouts.pdf", bbox_inches="tight")
    fig.savefig(output / "figures/rollouts-preview.png", dpi=150, bbox_inches="tight")
    (output / "data/figure_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "generated/rollout_metadata.tex").write_text(
        "% Generated from recorded rollout selections; do not edit.\n"
        + "".join(r"\newcommand{\%s}{%s}" % item + "\n" for item in macros.items()))
    print(json.dumps({"selected": [row["episode_id"] for _, row in selected],
                      "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
