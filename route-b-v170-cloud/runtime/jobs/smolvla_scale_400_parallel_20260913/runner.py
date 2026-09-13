#!/usr/bin/env python3
"""Schedule isolated task processes; retain their original evaluation records."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

JOB = Path(__file__).resolve().parent
ROOT = JOB.parents[2]
OUT = ROOT / "outputs" / JOB.name
CACHE = {}
STOP = False


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records(task, *, strict=False):
    path = Path(task["jsonl"])
    if not path.exists():
        return []
    stamp = (path.stat().st_mtime_ns, path.stat().st_size, strict)
    if CACHE.get(str(path), (None,))[0] == stamp:
        return CACHE[str(path)][1]
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    result = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if strict or index != len(lines) - 1 or line.endswith(b"\n"):
                raise
    ids = [row["episode_id"] for row in result]
    if len(ids) != len(set(ids)) or not set(ids) <= set(task["episodes"]):
        raise ValueError(f"{task['id']}: duplicate or unexpected episode")
    for row in result:
        key = row["key"]
        expected_id = f"b:{task['suite']}:task{task['task_id']:02d}:ep{key['episode_index']:04d}"
        if (row["episode_id"] != expected_id or key["route"] != "b"
                or key["suite"] != task["suite"] or key["task_id"] != task["task_id"]
                or row["seed"] != 7 or not 0 <= row["steps"] <= task["config"]["max_steps"]):
            raise ValueError(f"{task['id']}: record differs from planned configuration")
    CACHE[str(path)] = (stamp, result)
    return result


def metrics(rows):
    count = len(rows)
    successes = sum(row["evaluator_success"] for row in rows)
    return {"episodes": count, "successes": successes,
            "success_rate": successes / count if count else None,
            "steps": sum(row["steps"] for row in rows),
            "elapsed_s": round(sum(row["elapsed_s"] for row in rows), 3),
            "exceptions": sum(row["policy_status"] == "exception" for row in rows)}


def collect(manifest, active, task_states, started, workers, events):
    all_rows, tasks, suites = [], [], []
    fresh_rows = []
    for task in manifest["tasks"]:
        rows = records(task)
        all_rows.extend(rows)
        if not task["reused"]:
            fresh_rows.extend(rows)
        tasks.append({"id": task["id"], "suite": task["suite"], "task_id": task["task_id"],
                      "expected": 10, "reused": task["reused"],
                      "state": task_states[task["id"]], **metrics(rows)})
    for suite in manifest["suites"]:
        rows = [row for row in all_rows if row["key"]["suite"] == suite]
        suites.append({"suite": suite, "expected": 100, **metrics(rows)})
    overall = metrics(all_rows)
    elapsed = time.monotonic() - started
    rate = len(fresh_rows) / elapsed if elapsed > 0 else 0
    result = {
        "run_name": JOB.name, "updated_at": now(), "state": "running",
        "runner_pid": os.getpid(), "expected_episodes": 400,
        "completed_episodes": len(all_rows), "reused_episodes": 10,
        "new_completed_episodes": len(fresh_rows), "overall": overall,
        "configured_workers": workers, "active_workers": len(active),
        "active": [{"task": key, "pid": item["process"].pid,
                    "started_at": item["started_at"], "log": item["log"]}
                   for key, item in active.items()],
        "elapsed_wall_s": round(elapsed, 3),
        "observed_new_episodes_per_hour": round(rate * 3600, 3),
        "naive_remaining_s": round((400 - len(all_rows)) / rate) if rate else None,
        "eta_note": "Observed throughput only; task mix and changed parallelism can alter it.",
        "per_suite": suites, "per_task": tasks, "parallelism_events": events,
    }
    save(JOB / "status.json", result)
    save(OUT / "summary.json", {**result, "protocol": manifest["protocol"],
                               "source_tree_sha256": manifest["source_tree_sha256"]})
    return result, all_rows


def validate_task(task, *, complete):
    rows = records(task, strict=True)
    if complete and set(row["episode_id"] for row in rows) != set(task["episodes"]):
        raise ValueError(f"{task['id']}: incomplete coverage")
    if task["reused"]:
        if sha(task["jsonl"]) != task["snapshot_sha256"]:
            raise ValueError("Reused snapshot changed")
    else:
        summary = json.loads(Path(task["summary"]).read_text())
        actual = summary["run_config"]
        differences = [key for key in set(actual) | set(task["config"])
                       if key != "resume" and actual.get(key) != task["config"].get(key)]
        if differences:
            raise ValueError(f"{task['id']}: configuration mismatch {differences}")
        if (summary["overall"]["episodes"] != len(rows)
                or summary["overall"]["successes"] != sum(row["evaluator_success"] for row in rows)):
            raise ValueError(f"{task['id']}: summary differs from records")
    for row in rows:
        video = Path(row.get("video_paths", {}).get("dual", ""))
        if not video.is_file() or video.stat().st_size == 0:
            raise ValueError(f"{row['episode_id']}: missing or empty video")
    return rows


def stop_requested(signum, frame):
    global STOP
    STOP = True


def main():
    lock = (JOB / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((JOB / "manifest.json").read_text())
    resume = "--resume" in sys.argv[1:]
    subprocess.run(["sha256sum", "--quiet", "-c", "SHA256SUMS.deployed"], cwd=ROOT, check=True)
    all_ids = [episode for task in manifest["tasks"] for episode in task["episodes"]]
    assert len(all_ids) == len(set(all_ids)) == 400
    assert len(manifest["tasks"]) == 40
    signal.signal(signal.SIGINT, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)
    (JOB / "runner.pid").write_text(str(os.getpid()) + "\n")
    active, states, queue, events, failures = {}, {}, [], [], []
    for task in manifest["tasks"]:
        if task["reused"]:
            validate_task(task, complete=True)
            states[task["id"]] = "reused"
            continue
        exists = Path(task["summary"]).exists() or Path(task["jsonl"]).exists()
        if exists and not resume:
            raise RuntimeError("Existing shard artifacts require --resume")
        if exists:
            validate_task(task, complete=False)
        if len(records(task)) == 10:
            validate_task(task, complete=True)
            states[task["id"]] = "completed"
        else:
            states[task["id"]] = "queued"
            queue.append(task)
    started = time.monotonic()
    started_at = now()
    workers = None
    try:
        while queue or active:
            control = json.loads((JOB / "control.json").read_text())
            desired = int(control["workers"])
            if not 1 <= desired <= 8:
                raise ValueError("workers must be between 1 and 8")
            if desired != workers:
                workers = desired
                events.append({"at": now(), "workers": workers,
                               "elapsed_s": round(time.monotonic() - started, 3)})
            if STOP:
                break
            for key, item in list(active.items()):
                code = item["process"].poll()
                if code is None:
                    continue
                item["handle"].close()
                task = item["task"]
                del active[key]
                try:
                    if code != 0:
                        raise RuntimeError(f"process exited {code}")
                    validate_task(task, complete=True)
                    states[key] = "completed"
                except Exception as exc:
                    states[key] = "failed"
                    failures.append({"task": key, "error": str(exc), "exit_code": code})
                print(json.dumps({"event": "worker_finished", "at": now(), "task": key,
                                  "state": states[key], "exit_code": code}), flush=True)
            while queue and len(active) < workers:
                task = queue.pop(0)
                command = list(task["command"])
                if Path(task["summary"]).exists():
                    command.append("--resume")
                log = JOB / "logs" / f"{task['id']}.log"
                log.parent.mkdir(exist_ok=True)
                handle = log.open("a", buffering=1)
                handle.write(f"\nWorker started {now()}\n")
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                         stdout=handle, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                active[task["id"]] = {"task": task, "process": child, "handle": handle,
                                      "started_at": now(), "log": str(log)}
                states[task["id"]] = "running"
                print(json.dumps({"event": "worker_started", "at": now(),
                                  "task": task["id"], "pid": child.pid}), flush=True)
            current, rows = collect(manifest, active, states, started, workers, events)
            current["started_at"] = started_at
            current["failures"] = failures
            save(JOB / "status.json", current)
            with (JOB / "progress.jsonl").open("a") as handle:
                handle.write(json.dumps({key: current[key] for key in
                    ["updated_at", "elapsed_wall_s", "configured_workers", "active_workers",
                     "completed_episodes", "new_completed_episodes", "per_suite"]}) + "\n")
            if queue or active:
                time.sleep(10)
    finally:
        if active:
            for item in active.values():
                if item["process"].poll() is None:
                    item["process"].send_signal(signal.SIGINT)
            deadline = time.monotonic() + 30
            for item in active.values():
                try:
                    item["process"].wait(timeout=max(0.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    item["process"].terminate()
                    item["process"].wait(timeout=10)
                item["handle"].close()
                states[item["task"]["id"]] = "stopped"
            active.clear()
    final, rows = collect(manifest, active, states, started, workers, events)
    if STOP:
        final.update(state="stopped_by_user", finished_at=now())
        save(JOB / "status.json", final)
        save(OUT / "summary.json", final)
        return 130
    issues = list(failures)
    for task in manifest["tasks"]:
        try:
            validate_task(task, complete=True)
        except Exception as exc:
            issues.append({"task": task["id"], "error": str(exc)})
    if set(row["episode_id"] for row in rows) != set(all_ids) or len(rows) != 400:
        issues.append({"error": "campaign coverage differs from planned 400 episodes"})
    integrity = subprocess.run(["sha256sum", "--quiet", "-c", "SHA256SUMS.deployed"], cwd=ROOT)
    if integrity.returncode:
        issues.append({"error": "deployed source changed"})
    temporary = OUT / "episodes.jsonl.tmp"
    with temporary.open("w") as handle:
        for row in sorted(rows, key=lambda row: row["episode_id"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(OUT / "episodes.jsonl")
    state = "failed" if issues else ("completed_with_exceptions" if final["overall"]["exceptions"] else "completed")
    validation = {"validated_at": now(), "coverage_passed": not issues,
                  "issues": issues, "episodes": len(rows),
                  "episodes_jsonl_sha256": sha(OUT / "episodes.jsonl"),
                  "source_tree_sha256": manifest["source_tree_sha256"]}
    save(JOB / "final-validation.json", validation)
    final.update(state=state, finished_at=now(), validation=validation,
                 protocol=manifest["protocol"], started_at=started_at,
                 source_tree_sha256=manifest["source_tree_sha256"])
    save(JOB / "status.json", final)
    save(OUT / "summary.json", final)
    return 0 if state == "completed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        error = {"state": "runner_error", "at": now(), "error": f"{type(exc).__name__}: {exc}"}
        save(JOB / "runner-error.json", error)
        if (JOB / "status.json").exists():
            status = json.loads((JOB / "status.json").read_text())
            status.update(error)
            save(JOB / "status.json", status)
        raise
