#!/usr/bin/env python3
"""Run the uploaded campaign in a detached process and record progress/coverage."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

JOB = Path(__file__).resolve().parent
ROOT = JOB.parents[2]
NAME = JOB.name
PREFLIGHT = json.loads((JOB / "preflight.json").read_text())
LOG = ROOT / "logs" / f"{NAME}.log"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(name, value):
    path = JOB / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def progress():
    suites = []
    for item in PREFLIGHT:
        cfg = item["config"]
        path = ROOT / "outputs" / cfg["run_name"] / "summary.json"
        overall = {"episodes": 0, "successes": 0, "elapsed_s": 0}
        if path.exists():
            overall = json.loads(path.read_text())["overall"]
        suites.append({"suite": cfg["suite"], "expected": 500,
                       "summary": str(path), **overall})
    completed = sum(item["episodes"] for item in suites)
    successes = sum(item["successes"] for item in suites)
    return {"completed_episodes": completed, "expected_episodes": 2000,
            "successes": successes,
            "success_rate_so_far": successes / completed if completed else None,
            "per_suite": suites}


def validate():
    suites, all_rows, issues = [], [], []
    for entry in PREFLIGHT:
        cfg = entry["config"]
        directory = ROOT / "outputs" / cfg["run_name"]
        rows = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines() if line.strip()]
        identifiers = [row["episode_id"] for row in rows]
        expected = set(entry["episodes"])
        if len(rows) != 500 or len(set(identifiers)) != 500 or set(identifiers) != expected:
            issues.append(f"{cfg['suite']}: episode coverage mismatch")
        summary = json.loads((directory / "summary.json").read_text())
        actual_cfg = summary["run_config"]
        for key, value in cfg.items():
            if key != "resume" and actual_cfg.get(key) != value:
                issues.append(f"{cfg['suite']}: configuration changed: {key}")
        successes = sum(row["evaluator_success"] for row in rows)
        exceptions = sum(row["policy_status"] == "exception" for row in rows)
        if summary["overall"]["episodes"] != len(rows) or summary["overall"]["successes"] != successes:
            issues.append(f"{cfg['suite']}: summary does not match episode records")
        missing_videos = [row["episode_id"] for row in rows
                          if not row.get("video_paths", {}).get("dual")
                          or not Path(row["video_paths"]["dual"]).is_file()
                          or Path(row["video_paths"]["dual"]).stat().st_size == 0]
        if missing_videos:
            issues.append(f"{cfg['suite']}: {len(missing_videos)} missing/empty videos")
        suites.append({"suite": cfg["suite"], "episodes": len(rows), "successes": successes,
                       "success_rate": successes / len(rows) if rows else None,
                       "exceptions": exceptions, "missing_videos": missing_videos})
        all_rows.extend(rows)
    campaign_path = ROOT / "outputs" / f"{NAME}_campaign.json"
    campaign = json.loads(campaign_path.read_text())
    successes = sum(row["evaluator_success"] for row in all_rows)
    if campaign["episodes"] != 2000 or campaign["successes"] != successes:
        issues.append("campaign aggregate does not match episode records")
    integrity = subprocess.run(["sha256sum", "--quiet", "-c", "SHA256SUMS.deployed"], cwd=ROOT)
    if integrity.returncode:
        issues.append("deployed source checksum changed")
    return {"validated_at": now(), "coverage_passed": not issues,
            "issues": issues, "episodes": len(all_rows), "successes": successes,
            "success_rate": successes / len(all_rows) if all_rows else None,
            "exceptions": sum(row["policy_status"] == "exception" for row in all_rows),
            "per_suite": suites, "campaign_summary": str(campaign_path)}


def main():
    lock = (JOB / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    resume = "--resume" in sys.argv[1:]
    command = [str(ROOT / "run.sh"), "campaign", "--episodes-per-task", "50",
               "--device", "cuda", "--run-name", NAME]
    if resume:
        command.append("--resume")
    subprocess.run(["sha256sum", "--quiet", "-c", "SHA256SUMS.deployed"], cwd=ROOT, check=True)
    started = now()
    state = {"run_name": NAME, "started_at": started, "runner_pid": os.getpid(),
             "command": command, "log": str(LOG), "resumed": resume}
    (JOB / "runner.pid").write_text(str(os.getpid()) + "\n")
    with LOG.open("a", buffering=1) as handle:
        handle.write(f"\nCampaign attempt started: {started}; resume={resume}\n")
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=handle, stderr=subprocess.STDOUT)
        state["campaign_pid"] = child.pid
        (JOB / "campaign.pid").write_text(str(child.pid) + "\n")
        while child.poll() is None:
            save("status.json", {**state, "state": "running", "updated_at": now(), **progress()})
            time.sleep(15)
        code = child.returncode
        final = {**state, "updated_at": now(), "finished_at": now(),
                 "exit_code": code, "state": "failed", **progress()}
        if code == 0:
            try:
                validation = validate()
                save("final-validation.json", validation)
                final["validation"] = validation
                if validation["coverage_passed"]:
                    final["state"] = "completed_with_exceptions" if validation["exceptions"] else "completed"
            except Exception as exc:
                final["validation_error"] = f"{type(exc).__name__}: {exc}"
        save("status.json", final)
        handle.write(f"Campaign attempt finished: {final['finished_at']}; state={final['state']}; exit_code={code}\n")
    return 0 if final["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
