#!/usr/bin/env python3
"""Run fixed-protocol tasks against an immutable copy of the candidate source."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
SUITES = {"spatial": "libero_spatial", "object": "libero_object",
          "goal": "libero_goal", "long": "libero_10"}
CAPS = {"spatial": 220, "object": 280, "goal": 300, "long": 520}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files(root):
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def task_rows(task):
    path = Path(task["jsonl"])
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    # The worker may currently be appending its last line.
    return [json.loads(line) for line in lines if line.endswith(b"\n") and line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--deployed-root", type=Path,
                        default=Path("/root/autodl-tmp/route-b-v170-cloud"))
    parser.add_argument("--tasks", nargs="+", default=["goal:3", "spatial:4", "long:3"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    assert args.label and all(c.isalnum() or c in "-_" for c in args.label)
    assert 1 <= args.workers <= 8 and 1 <= args.episodes <= 10
    assert 0 <= args.start and args.start + args.episodes <= 10
    if args.all:
        assert args.episodes == 10 and args.start == 0
    selection = ([f"{s}:{t}" for s in SUITES for t in range(10)]
                 if args.all else args.tasks)
    assert len(selection) == len(set(selection))
    parsed = [(s, int(t)) for s, t in (entry.split(":") for entry in selection)]
    assert all(s in SUITES and 0 <= t < 10 for s, t in parsed)
    batch = REPO / "runtime/route_b_90" / args.label
    batch.mkdir(parents=True, exist_ok=False)
    snapshot = batch / "source"
    snapshot.mkdir()
    source = REPO / "route-b-v170-cloud"
    shutil.copytree(source / "libero_system", snapshot / "libero_system",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("run.sh", "cloud.py", "resources.lock.json"):
        shutil.copy2(source / name, snapshot / name)
    (snapshot / ".venv").symlink_to(args.deployed_root / ".venv", target_is_directory=True)
    (snapshot / "resources").symlink_to(args.deployed_root / "resources", target_is_directory=True)
    (batch / "shards").mkdir()
    (batch / "logs").mkdir()
    frozen = source_files(snapshot / "libero_system")
    save(batch / "source_files.json", frozen)
    tasks = []
    for suite, task_id in parsed:
        name = f"{suite}_task{task_id:02d}"
        command = [str(snapshot / "run.sh"), "run", "--suite", suite,
                   "--task-ids", str(task_id), "--episodes-per-task", str(args.episodes),
                   "--init-start", str(args.start), "--device", "cuda", "--seed", "7",
                   "--run-name", name, "--output-dir", str(batch / "shards")]
        tasks.append({"id": name, "suite": SUITES[suite], "task_id": task_id,
                      "command": command, "reused": False,
                      "episodes": [f"b:{SUITES[suite]}:task{task_id:02d}:ep{i:04d}"
                                   for i in range(args.start, args.start + args.episodes)],
                      "jsonl": str(batch / "shards" / name / "episodes.jsonl"),
                      "summary": str(batch / "shards" / name / "summary.json")})
    probe = subprocess.run(tasks[0]["command"] + ["--dry-run"], capture_output=True, text=True)
    (batch / "logs/dry-run.log").write_text(probe.stdout + probe.stderr)
    assert probe.returncode == 0, probe.stdout + probe.stderr
    first = json.loads(probe.stdout[probe.stdout.index("{"):])["config"]
    for task, (suite, _) in zip(tasks, parsed):
        task["config"] = {**first, "suite": task["suite"], "task_ids": [task["task_id"]],
                          "run_name": task["id"], "max_steps": CAPS[suite]}
    manifest = {"run_name": args.label, "created_at": now(),
                "kind": "full_400" if args.all else "development",
                "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                    cwd=REPO, text=True).strip(),
                "source_tree_sha256": first["source_tree_sha256"],
                "source_snapshot": str(snapshot), "tasks": tasks,
                "protocol": {"tasks": len(tasks), "episodes_per_task": args.episodes,
                             "official_init_ids": list(range(args.start, args.start + args.episodes)),
                             "seed": 7, "step_caps": CAPS,
                             "scoring": "external_sticky_any_success",
                             "expected_episodes": len(tasks) * args.episodes,
                             "new_episodes": len(tasks) * args.episodes, "reused_episodes": 0}}
    save(batch / "manifest.json", manifest)
    save(batch / "control.json", {"workers": args.workers})
    pending, active, done, failures = list(tasks), {}, [], []
    started = time.monotonic()
    started_at = now()
    (batch / "runner.pid").write_text(str(__import__("os").getpid()) + "\n")
    print(json.dumps({"event": "start", "batch": str(batch),
                      "source_tree_sha256": first["source_tree_sha256"]}), flush=True)
    try:
        while pending or active:
            workers = int(json.loads((batch / "control.json").read_text())["workers"])
            assert 1 <= workers <= 8
            for name, (task, process, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[name]
                rows = task_rows(task)
                issues = []
                if code != 0:
                    issues.append(f"exit code {code}")
                else:
                    summary = json.loads(Path(task["summary"]).read_text())
                    if summary["run_config"] != task["config"]:
                        issues.append("run configuration mismatch")
                    if {r["episode_id"] for r in rows} != set(task["episodes"]):
                        issues.append("episode coverage mismatch")
                    if len(rows) != args.episodes:
                        issues.append("episode count mismatch")
                    for row in rows:
                        if row["seed"] != 7 or not 0 <= row["steps"] <= task["config"]["max_steps"]:
                            issues.append("seed or step budget mismatch")
                        if row["route_trace"]["evaluator_isolation"]["scoring"] != "external_sticky_any_success":
                            issues.append("scoring mismatch")
                        if row["policy_status"] == "exception":
                            issues.append("runtime exception")
                        if not all(Path(p).is_file() for p in row["video_paths"].values()):
                            issues.append("missing video")
                if issues:
                    failures.append({"task": name, "issues": issues})
                done.append(name)
                print(json.dumps({"event": "finished", "task": name,
                                  "successes": sum(r["evaluator_success"] for r in rows),
                                  "episodes": len(rows), "issues": issues}), flush=True)
            while pending and len(active) < workers:
                task = pending.pop(0)
                log = (batch / "logs" / (task["id"] + ".log")).open("w")
                process = subprocess.Popen(task["command"], cwd=snapshot,
                                           stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                active[task["id"]] = (task, process, log)
            rows = [row for task in tasks for row in task_rows(task)]
            status = {"state": "running" if pending or active else ("failed" if failures else "completed"),
                      "updated_at": now(), "started_at": started_at,
                      "elapsed_wall_s": time.monotonic() - started,
                      "completed_episodes": len(rows), "active": {k: v[1].pid for k, v in active.items()},
                      "completed_tasks": done, "pending_tasks": [t["id"] for t in pending],
                      "failures": failures,
                      "overall": {"episodes": len(rows), "successes": sum(r["evaluator_success"] for r in rows)},
                      "per_task": {t["id"]: {"episodes": len(task_rows(t)),
                                            "successes": sum(r["evaluator_success"] for r in task_rows(t))}
                                   for t in tasks}}
            save(batch / "status.json", status)
            if pending or active:
                time.sleep(5)
        assert source_files(snapshot / "libero_system") == frozen, "Frozen source changed."
        merged = b"".join(Path(t["jsonl"]).read_bytes() for t in tasks)
        (batch / "episodes.jsonl").write_bytes(merged)
        save(batch / "final-validation.json", {
            "coverage_passed": not failures, "issues": failures, "episodes": len(rows),
            "source_tree_sha256": first["source_tree_sha256"],
            "episodes_jsonl_sha256": hashlib.sha256(merged).hexdigest(),
            "validated_at": now()})
        print(json.dumps(status), flush=True)
        assert not failures, failures
    except KeyboardInterrupt:
        status_path = batch / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        status.update(state="interrupted", updated_at=now(), active={})
        save(status_path, status)
        print(json.dumps({"event": "interrupted", "batch": str(batch)}), flush=True)
        raise SystemExit(130)
    finally:
        for task, process, log in active.values():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            log.close()


if __name__ == "__main__":
    main()
