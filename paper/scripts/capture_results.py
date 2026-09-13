#!/usr/bin/env python3
"""Freeze an auditable, compact projection of the existing evaluation records."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PAPER = Path(__file__).resolve().parents[1]
RUN = "smolvla_scale_400_parallel_20260913"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = args.source_root.resolve()
    job = root / "runtime/jobs" / RUN
    manifest = json.loads((job / "manifest.json").read_text())
    status = json.loads((job / "status.json").read_text())
    complete = status["state"] == "completed"
    validation_path = job / "final-validation.json"
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else None
    if not args.allow_partial:
        assert complete, f"Campaign is still {status['state']}; no final results claimed."
        assert validation and validation["coverage_passed"] and not validation["issues"]
    subprocess.run(["sha256sum", "--quiet", "-c", "SHA256SUMS.deployed"],
                   cwd=root, check=True)
    rows, sources, seen = [], [], set()
    expected = set()
    for task in manifest["tasks"]:
        expected.update(task["episodes"])
        path = Path(task["jsonl"])
        if not path.exists():
            continue
        raw = path.read_bytes()
        sources.append({"path": str(path.relative_to(root)), "sha256": sha(raw),
                        "task": task["id"], "reused": task["reused"]})
        for line in raw.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            assert row["episode_id"] in task["episodes"]
            assert row["episode_id"] not in seen
            assert row["seed"] == 7
            assert 0 <= row["steps"] <= task["config"]["max_steps"]
            assert row["key"]["suite"] == task["suite"]
            assert row["key"]["task_id"] == task["task_id"]
            seen.add(row["episode_id"])
            trace = row["route_trace"]
            rows.append({
                "episode_id": row["episode_id"],
                "suite": row["key"]["suite"], "task_id": row["key"]["task_id"],
                "init_id": row["key"]["episode_index"],
                "instruction": row["instruction"], "seed": row["seed"],
                "success": row["evaluator_success"], "policy_status": row["policy_status"],
                "failure": row["failure"], "steps": row["steps"],
                "elapsed_s": row["elapsed_s"], "reused": task["reused"],
                "final_phase": trace.get("phase"),
                "grasp_attempts": len(trace.get("grasp_target_attempts", [])),
                "grasp_checks": len(trace.get("grasp_verifications", [])),
                "rejected_grasp_checks": sum(not x.get("accepted", False)
                    for x in trace.get("grasp_verifications", [])),
                "placement_attempts": len(trace.get("placement_target_attempts", [])),
                "scoring": trace["evaluator_isolation"]["scoring"],
                "source_record_sha256": sha(line),
            })
    assert len(expected) == 400
    if complete:
        assert seen == expected and len(rows) == 400
        assert sum(r["success"] for r in rows) == status["overall"]["successes"]
        assert sum(r["reused"] for r in rows) == 10
    rows.sort(key=lambda r: (r["suite"], r["task_id"], r["init_id"]))
    out = PAPER / "data"
    out.mkdir(exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    (out / "episodes.jsonl").write_text(payload)
    dump(out / "provenance.json", {
        "run_name": RUN, "state": status["state"], "complete": complete,
        "status_updated_at": status["updated_at"],
        "recorded_episodes": len(rows), "expected_episodes": 400,
        "protocol": manifest["protocol"],
        "controller_source_sha256": manifest["source_tree_sha256"],
        "source_manifest_sha256": sha((job / "manifest.json").read_bytes()),
        "episodes_projection_sha256": sha(payload.encode()),
        "source_files": sources,
        "projection_note": "Numerical fields are copied from original records; trace counts are derived. "
            "Each source_record_sha256 hashes its original JSONL line without the newline. "
            "Original traces and videos remain on the evaluation server.",
        "validation": validation,
        "wall_elapsed_s": status.get("elapsed_wall_s"),
    })
    print(f"Captured {len(rows)}/400 episodes; complete={complete}")


if __name__ == "__main__":
    main()
