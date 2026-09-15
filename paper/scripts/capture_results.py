#!/usr/bin/env python3
"""Freeze an auditable, compact projection of the existing evaluation records."""
import argparse
import gzip
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-root", type=Path, help="Original deployed campaign root")
    source.add_argument("--batch-dir", type=Path, help="Immutable fresh full_400 campaign")
    parser.add_argument("--output-dir", type=Path, default=PAPER / "data")
    parser.add_argument("--environment", type=Path,
                        help="Measured runtime environment to preserve with the imported campaign.")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = args.source_root.resolve() if args.source_root else args.batch_dir.resolve()
    job = root / "runtime/jobs" / RUN if args.source_root else root
    manifest = json.loads((job / "manifest.json").read_text())
    if args.batch_dir:
        assert manifest.get("kind") == "full_400", "Development batches cannot update the paper result."
        assert manifest["protocol"]["new_episodes"] == 400
        assert manifest["protocol"]["reused_episodes"] == 0
        snapshot = Path(manifest["source_snapshot"]) / manifest.get("source_package", "libero_system")
        frozen = json.loads((job / "source_files.json").read_text())
        current = {str(p.relative_to(snapshot)): sha(p.read_bytes())
                   for p in sorted(snapshot.rglob("*"))
                   if p.is_file() and "__pycache__" not in p.parts}
        assert current == frozen, "Frozen controller source changed after evaluation."
    status = json.loads((job / "status.json").read_text())
    complete = status["state"] == "completed"
    validation_path = job / "final-validation.json"
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else None
    if not args.allow_partial:
        assert complete, f"Campaign is still {status['state']}; no final results claimed."
        assert validation and validation["coverage_passed"] and not validation["issues"]
    if args.source_root:
        subprocess.run(["sha256sum", "--quiet", "-c", "SHA256SUMS.deployed"],
                       cwd=root, check=True)
    rows, sources, seen, originals = [], [], set(), []
    expected = set()
    for task in manifest["tasks"]:
        assert task["config"]["source_tree_sha256"] == manifest["source_tree_sha256"]
        expected.update(task["episodes"])
        path = Path(task["jsonl"])
        if not path.exists():
            continue
        raw = path.read_bytes()
        originals.append(raw)
        if args.batch_dir:
            summary = json.loads(Path(task["summary"]).read_text())
            assert summary["run_config"] == task["config"]
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
                "source_campaign": manifest.get("run_name", RUN),
                "controller_source_sha256": manifest["source_tree_sha256"],
            })
    assert len(expected) == 400
    if complete:
        assert seen == expected and len(rows) == 400
        assert sum(r["success"] for r in rows) == status["overall"]["successes"]
        assert sum(r["reused"] for r in rows) == manifest["protocol"]["reused_episodes"]
    rows.sort(key=lambda r: (r["suite"], r["task_id"], r["init_id"]))
    out = args.output_dir.resolve()
    archive = None
    archive_data = None
    if args.batch_dir and complete:
        original_data = b"".join(originals)
        assert sha(original_data) == validation["episodes_jsonl_sha256"]
        archive_data = gzip.compress(original_data, mtime=0)
        archive = {"file": "raw_episodes.jsonl.gz", "sha256": sha(archive_data),
                   "uncompressed_sha256": sha(original_data), "records": len(rows)}
    out.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    (out / "episodes.jsonl").write_text(payload)
    if archive is not None:
        (out / archive["file"]).write_bytes(archive_data)
    environment = {}
    if args.environment:
        environment_data = args.environment.read_bytes()
        runtime = json.loads(environment_data)
        assert runtime["controller_source_sha256"] == manifest["source_tree_sha256"]
        assert runtime["campaign_label"] == manifest["run_name"]
        (out / "environment.json").write_bytes(environment_data)
        environment = {"environment_file": "environment.json",
                       "environment_sha256": sha(environment_data)}
    dump(out / "provenance.json", {
        "run_name": manifest.get("run_name", RUN), "state": status["state"], "complete": complete,
        "evaluation_kind": "full_campaign",
        "status_updated_at": status["updated_at"],
        "recorded_episodes": len(rows), "expected_episodes": 400,
        "protocol": manifest["protocol"],
        "controller_source_sha256": manifest["source_tree_sha256"],
        "source_manifest_sha256": sha((job / "manifest.json").read_bytes()),
        "episodes_projection_sha256": sha(payload.encode()),
        "source_files": sources,
        **({"raw_episode_archive": archive} if archive else {}),
        **environment,
        "projection_note": "Numerical fields are copied from original records; trace counts are derived. "
            "Each source_record_sha256 hashes its original JSONL line without the newline. "
            + ("Complete original records, including geometry traces, are included in the compressed archive; "
               "videos remain in the source campaign directory." if archive else
               "Original traces and videos remain in the source campaign directory."),
        "validation": validation,
        "wall_elapsed_s": status.get("elapsed_wall_s"),
    })
    print(f"Captured {len(rows)}/400 episodes; complete={complete}")


if __name__ == "__main__":
    main()
