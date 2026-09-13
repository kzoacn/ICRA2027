#!/usr/bin/env python3
"""Extract actual recorded frames; no synthetic or retouched observations."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.family": "DejaVu Sans"})

PAPER = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    selected = [("object_task00", 0, "Object 00 / init 00: external success"),
                ("goal_task03", 2, "Goal 03 / init 02: action-budget exhaustion")]
    fig, axes = plt.subplots(2, 4, figsize=(6.8, 3.35))
    provenance = []
    for row_index, (task, init, label) in enumerate(selected):
        records = source / "outputs/smolvla_scale_400_parallel_20260913/shards" / task / "episodes.jsonl"
        row = next(json.loads(line) for line in records.read_text().splitlines()
                   if json.loads(line)["key"]["episode_index"] == init)
        video = Path(row["video_paths"]["dual"])
        reader = imageio.get_reader(video)
        frames = reader.count_frames()
        indices = np.linspace(0, frames - 1, 4).round().astype(int)
        for column, index in enumerate(indices):
            frame = reader.get_data(int(index))
            axes[row_index, column].imshow(frame[:, :frame.shape[1] // 2])
            axes[row_index, column].set_title(f"Recorded frame {index}", fontsize=7, pad=2)
            axes[row_index, column].axis("off")
        reader.close()
        provenance.append({"episode_id": row["episode_id"], "success": row["evaluator_success"],
            "policy_status": row["policy_status"], "instruction": row["instruction"],
            "video": str(video.relative_to(source)),
            "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "total_recorded_frames": frames, "selected_frames": indices.tolist(),
            "transformation": "Left half of original dual-view frame (agentview); no retouching."})
        fig.text(.01, .985 if row_index == 0 else .493, label, fontsize=8, weight="bold")
    fig.subplots_adjust(left=.01, right=.99, top=.92, bottom=.02, hspace=.40, wspace=.03)
    fig.savefig(PAPER / "figures/rollouts.pdf", bbox_inches="tight")
    fig.savefig(PAPER / "figures/rollouts-preview.png", dpi=150, bbox_inches="tight")
    (PAPER / "data/figure_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
