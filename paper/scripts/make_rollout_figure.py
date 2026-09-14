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
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.family": "DejaVu Sans"})

PAPER = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source-root", type=Path)
    inputs.add_argument("--batch-dir", type=Path)
    args = parser.parse_args()
    source = (args.batch_dir or args.source_root).resolve()
    records_root = (source / "shards" if args.batch_dir else
                    source / "outputs/smolvla_scale_400_parallel_20260913/shards")
    selected = [("object_task00", 0, "Object 00 / init 00: external success"),
                ("goal_task03", 2, "Goal 03 / init 02: action-budget exhaustion")]
    frame_dir = PAPER / "figures/recorded"
    frame_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(6.8, 3.35))
    provenance = []
    for row_index, (task, init, label) in enumerate(selected):
        records = records_root / task / "episodes.jsonl"
        row = next(json.loads(line) for line in records.read_text().splitlines()
                   if json.loads(line)["key"]["episode_index"] == init)
        assert row["evaluator_success"] is (row_index == 0), "Update the caption when the illustrated outcome changes."
        if row_index == 1:
            assert row["policy_status"] == "timeout", "The illustrated failure must match its action-budget caption."
            preceding = [json.loads(line) for line in records.read_text().splitlines()
                         if json.loads(line)["key"]["episode_index"] < init]
            assert all(item["evaluator_success"] for item in preceding), "The selected failure must be first in index order."
        video = Path(row["video_paths"]["dual"])
        reader = imageio.get_reader(video)
        frames = reader.count_frames()
        indices = np.linspace(0, frames - 1, 4).round().astype(int)
        display_frames = []
        for column, index in enumerate(indices):
            frame = reader.get_data(int(index))
            # The recorder stores the simulator camera's bottom-up raster.
            # Flip only the vertical axis; preserve left/right and every pixel.
            upright = np.ascontiguousarray(frame[::-1, :frame.shape[1] // 2])
            frame_path = frame_dir / f"{task}_init{init:02d}_frame{index:03d}.png"
            imageio.imwrite(frame_path, upright)
            display_frames.append({"index": int(index),
                "file": str(frame_path.relative_to(PAPER)),
                "sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
                "pixel_sha256": hashlib.sha256(upright.tobytes()).hexdigest()})
            axes[row_index, column].imshow(upright)
            axes[row_index, column].set_title(f"Recorded frame {index}", fontsize=7, pad=2)
            axes[row_index, column].axis("off")
        reader.close()
        provenance.append({"episode_id": row["episode_id"], "success": row["evaluator_success"],
            "policy_status": row["policy_status"], "instruction": row["instruction"],
            "video": str(video.relative_to(source)),
            "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "total_recorded_frames": frames, "selected_frames": indices.tolist(),
            "display_frames": display_frames,
            "transformation": "Left half of the original dual-view frame (agentview), then vertical-axis flip to correct display orientation; left/right preserved; no retouching."})
        fig.text(.01, .985 if row_index == 0 else .493, label, fontsize=8, weight="bold")
    fig.subplots_adjust(left=.01, right=.99, top=.92, bottom=.02, hspace=.40, wspace=.03)
    fig.savefig(PAPER / "figures/rollouts.pdf", bbox_inches="tight")
    fig.savefig(PAPER / "figures/rollouts-preview.png", dpi=150, bbox_inches="tight")
    (PAPER / "data/figure_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
