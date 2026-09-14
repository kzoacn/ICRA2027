#!/usr/bin/env python3
"""Index Goal 03 development attempts, retaining each frozen candidate separately."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    batches = []
    for folder in sorted((ROOT / "runtime/route_b_90").glob("goal03_*")):
        manifest_file = folder / "manifest.json"
        if not manifest_file.exists():
            continue
        manifest = json.loads(manifest_file.read_text())
        status_file = folder / "status.json"
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
        records = []
        for task in manifest["tasks"]:
            path = Path(task["jsonl"])
            if not path.exists():
                continue
            for line in path.read_bytes().splitlines(keepends=True):
                if not line.endswith(b"\n"):
                    continue
                row = json.loads(line)
                records.append({
                    "episode_id": row["episode_id"],
                    "init_id": row["key"]["episode_index"],
                    "success": row["evaluator_success"],
                    "policy_status": row["policy_status"],
                    "steps": row["steps"], "failure": row["failure"],
                    "source_record_sha256": hashlib.sha256(line.rstrip(b"\n")).hexdigest(),
                })
        validation = folder / "final-validation.json"
        batches.append({
            "batch": manifest["run_name"], "state": status.get("state", "unknown"),
            "source_tree_sha256": manifest["source_tree_sha256"],
            "protocol": manifest["protocol"], "episodes": len(records),
            "successes": sum(row["success"] for row in records), "records": records,
            "validation": json.loads(validation.read_text()) if validation.exists() else None,
        })
    output = {
        "scope": "Development and complete targeted retests; each candidate retains its own outcomes. No new full-suite campaign is included.",
        "batches": batches,
    }
    (Path(__file__).parent / "development_record.json").write_text(
        json.dumps(output, indent=2) + "\n")
    print(json.dumps([{key: batch[key] for key in ("batch", "state", "episodes", "successes")}
                      for batch in batches], indent=2))


if __name__ == "__main__":
    main()
