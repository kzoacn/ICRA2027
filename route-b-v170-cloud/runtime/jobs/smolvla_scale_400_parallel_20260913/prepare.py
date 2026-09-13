#!/usr/bin/env python3
"""Freeze the 400-case schedule and verify reuse before starting workers."""
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

JOB = Path(__file__).resolve().parent
ROOT = JOB.parents[2]
OUT = ROOT / "outputs" / JOB.name
SUITES = {"spatial": "libero_spatial", "object": "libero_object",
          "goal": "libero_goal", "long": "libero_10"}


def build(spec):
    short, task_id = spec
    name = f"{short}_task{task_id:02d}"
    command = [str(ROOT / "run.sh"), "run", "--suite", short,
               "--task-ids", str(task_id), "--episodes-per-task", "10",
               "--init-start", "0", "--device", "cuda", "--seed", "7",
               "--run-name", name, "--output-dir", str(OUT / "shards")]
    run = subprocess.run(command + ["--dry-run"], cwd=ROOT, text=True,
                         capture_output=True, check=True, timeout=90)
    parsed = json.loads(run.stdout)
    directory = OUT / "shards" / name
    return {"id": name, "suite": SUITES[short], "task_id": task_id,
            "command": command, "config": parsed["config"],
            "episodes": parsed["episodes"], "reused": False,
            "jsonl": str(directory / "episodes.jsonl"),
            "summary": str(directory / "summary.json")}


def main():
    assert not (JOB / "manifest.json").exists(), "Do not overwrite an existing manifest"
    ast.parse((JOB / "runner.py").read_text())
    subprocess.run(["sha256sum", "--quiet", "-c", "SHA256SUMS.deployed"], cwd=ROOT, check=True)
    old = ROOT / "runtime/jobs/full_2000_20260913"
    state = json.loads((old / "status.json").read_text())
    assert state["state"] == "stopped_by_user"
    assert not Path(f"/proc/{state['campaign_pid']}").exists()
    # Interleave suites so the initial throughput sample includes long tasks.
    schedule = [(suite, task) for task in range(10)
                for suite in ["long", "object", "goal", "spatial"]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = list(pool.map(build, schedule))
    ids = [episode for task in tasks for episode in task["episodes"]]
    assert len(ids) == len(set(ids)) == 400
    for task in tasks:
        assert task["episodes"] == [f"b:{task['suite']}:task{task['task_id']:02d}:ep{i:04d}" for i in range(10)]
    reused = next(task for task in tasks if task["id"] == "spatial_task00")
    source_dir = ROOT / "outputs/full_2000_20260913_spatial"
    old_cfg = json.loads((source_dir / "summary.json").read_text())["run_config"]
    metadata = {"task_ids", "episodes_per_task", "total_episodes", "run_name", "output_dir", "resume"}
    differences = [key for key in set(old_cfg) | set(reused["config"])
                   if key not in metadata and old_cfg.get(key) != reused["config"].get(key)]
    assert not differences, differences
    lines = (source_dir / "episodes.jsonl").read_text().splitlines()
    selected = [line for line in lines if json.loads(line)["episode_id"] in set(reused["episodes"])]
    assert len(selected) == 10
    assert {json.loads(line)["episode_id"] for line in selected} == set(reused["episodes"])
    for line in selected:
        video = Path(json.loads(line)["video_paths"]["dual"])
        assert video.is_file() and video.stat().st_size > 0
    reuse_dir = OUT / "reused/spatial_task00"
    reuse_dir.mkdir(parents=True)
    snapshot = reuse_dir / "episodes.jsonl"
    snapshot.write_text("\n".join(selected) + "\n")
    source = {"source_run": "full_2000_20260913_spatial", "source_directory": str(source_dir),
              "original_config": old_cfg, "selected_episode_ids": reused["episodes"],
              "selection": "Official initial states 0-9, selected before the new run; no success filtering",
              "metadata_differences_only": sorted(metadata)}
    (reuse_dir / "provenance.json").write_text(json.dumps(source,ensure_ascii=False,indent=2)+"\n")
    reused.update(reused=True, jsonl=str(snapshot), summary=str(source_dir / "summary.json"),
                  original_source=str(source_dir / "episodes.jsonl"),
                  snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
    manifest = {
        "run_name": JOB.name, "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {"sampling_reference": "https://arxiv.org/html/2506.01844v1",
                     "scope": "SmolVLA sampling scale; existing Route B controller and environment settings",
                     "tasks": 40, "episodes_per_task": 10, "official_init_ids": list(range(10)),
                     "seed": 7, "expected_episodes": 400, "reused_episodes": 10,
                     "new_episodes": 390, "step_caps": {"spatial":220,"object":280,"goal":300,"long":520},
                     "record_video": True, "image_size":256, "video_fps":20, "video_stride":2,
                     "scoring":"external_sticky_any_success", "policy_unchanged":True},
        "source_tree_sha256": tasks[0]["config"]["source_tree_sha256"],
        "suites": list(SUITES.values()), "tasks": tasks,
    }
    for task in tasks:
        assert task["config"]["source_tree_sha256"] == manifest["source_tree_sha256"]
    (JOB / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    (OUT / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    (JOB / "control.json").write_text('{"workers": 4}\n')
    print(json.dumps({"validated_tasks":len(tasks),"unique_episodes":len(set(ids)),
                      "reused":10,"new":390,"initial_workers":4,
                      "source_tree_sha256":manifest["source_tree_sha256"]}))


if __name__ == "__main__":
    main()
