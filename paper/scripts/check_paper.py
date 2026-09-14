#!/usr/bin/env python3
"""Check the final local artifact, its anonymous PDF, and the measured dataset."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

PAPER = Path(__file__).resolve().parents[1]


def main():
    meta = json.loads((PAPER / "data/provenance.json").read_text())
    data = (PAPER / "data/episodes.jsonl").read_bytes()
    rows = [json.loads(line) for line in data.splitlines()]
    assert meta["complete"], "A partial snapshot cannot pass final-paper validation."
    assert meta["validation"]["coverage_passed"]
    assert not meta["validation"]["issues"]
    assert hashlib.sha256(data).hexdigest() == meta["episodes_projection_sha256"]
    assert len(rows) == len({r["episode_id"] for r in rows}) == 400
    if archive := meta.get("raw_episode_archive"):
        compressed = (PAPER / "data" / archive["file"]).read_bytes()
        assert hashlib.sha256(compressed).hexdigest() == archive["sha256"]
        original = gzip.decompress(compressed)
        assert hashlib.sha256(original).hexdigest() == archive["uncompressed_sha256"]
        assert archive["uncompressed_sha256"] == meta["validation"]["episodes_jsonl_sha256"]
        lines = original.splitlines()
        actual = {json.loads(line)["episode_id"]:(json.loads(line), line) for line in lines}
        assert len(lines) == len(actual) == archive["records"] == 400
        for row in rows:
            source, line = actual[row["episode_id"]]
            assert hashlib.sha256(line).hexdigest() == row["source_record_sha256"]
            assert source["evaluator_success"] is row["success"]
            for key in ["instruction", "seed", "policy_status", "failure", "steps", "elapsed_s"]:
                assert source[key] == row[key]
            assert source["key"]["suite"] == row["suite"]
            assert source["key"]["task_id"] == row["task_id"]
            assert source["key"]["episode_index"] == row["init_id"]
    groups = collections.defaultdict(set)
    for row in rows:
        groups[row["suite"], row["task_id"]].add(row["init_id"])
        assert type(row["success"]) is bool and row["seed"] == 7
        assert row["scoring"] == "external_sticky_any_success"
    assert len(groups) == 40 and all(v == set(range(10)) for v in groups.values())
    assert sum(r["reused"] for r in rows) == meta["protocol"]["reused_episodes"]
    task_refresh = meta.get("evaluation_kind") == "task_replacement"
    if task_refresh:
        assert meta["controller_source_sha256"] is None
        replacements = meta.get("task_replacements") or [meta["task_replacement"]]
        replacement_by_task = {(item["suite"], item["task_id"]): item for item in replacements}
        assert len(replacement_by_task) == len(replacements)
        reference_dir = PAPER / "data/full400_reference"
        reference_meta = json.loads((reference_dir / "provenance.json").read_text())
        reference_data = (reference_dir / "episodes.jsonl").read_bytes()
        assert hashlib.sha256(reference_data).hexdigest() == reference_meta["episodes_projection_sha256"]
        reference_rows = {r["episode_id"]: r for r in map(json.loads, reference_data.splitlines())}
        refreshed, retained = collections.defaultdict(list), []
        for row in rows:
            key = (row["suite"], row["task_id"])
            if key in replacement_by_task:
                replacement = replacement_by_task[key]
                assert not row["reused"]
                assert row["controller_source_sha256"] == replacement["batch_manifest"]["source_tree_sha256"]
                assert row["source_campaign"] == replacement["batch_manifest"]["run_name"]
                refreshed[key].append(row)
            else:
                assert row["reused"]
                assert row["controller_source_sha256"] == reference_meta["controller_source_sha256"]
                assert row["source_record_sha256"] == reference_rows[row["episode_id"]]["source_record_sha256"]
                retained.append(row)
        assert len(retained) == 400 - 10*len(replacements)
        for key, replacement in replacement_by_task.items():
            group = refreshed[key]
            assert len(group) == replacement["episodes"] == 10
            assert sum(row["success"] for row in group) == replacement["successes"]
            refreshed_raw = b"".join(actual[row["episode_id"]][1] + b"\n"
                                      for row in sorted(group, key=lambda row: row["init_id"]))
            assert hashlib.sha256(refreshed_raw).hexdigest() == replacement["batch_validation"]["episodes_jsonl_sha256"]
            assert replacement["batch_validation"]["coverage_passed"]
            assert not replacement["batch_validation"]["issues"]
            assert replacement["baseline_successes"] == sum(row["success"] for row in reference_rows.values()
                if (row["suite"], row["task_id"]) == key)
        expected_sources = collections.Counter()
        for item in meta["controller_sources"]:
            expected_sources[item["source_tree_sha256"]] += item["included_episodes"]
        assert collections.Counter(row["controller_source_sha256"] for row in rows) == expected_sources
    caps = {"libero_spatial":220, "libero_object":280, "libero_goal":300, "libero_10":520}
    assert all(0 <= r["steps"] <= caps[r["suite"]] for r in rows)
    numbers = (PAPER / "generated/numbers.tex").read_text()
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]+)\}", numbers))
    # Check every task count used in the prose against the frozen records.
    for name, suite, task in [("TopDrawerSuccess","libero_goal",3),
                              ("DrawerPickSuccess","libero_spatial",4),
                              ("BottomDrawerSuccess","libero_10",3),
                              ("MokaSuccess","libero_10",8),
                              ("MicrowaveSuccess","libero_10",9)]:
        expected = sum(r["success"] for r in rows if r["suite"] == suite and r["task_id"] == task)
        assert int(macros[name]) == expected
        assert "\\" + name in (PAPER / "main.tex").read_text()
    assert int(macros["SuccessCount"]) == sum(r["success"] for r in rows)
    assert int(macros["ReusedCount"]) == sum(r["reused"] for r in rows)
    assert (r"\AllFreshtrue" in numbers) == (not any(r["reused"] for r in rows))
    assert (r"\TaskRefreshtrue" in numbers) == task_refresh
    tex = (PAPER / "main.tex").read_text()
    subprocess.run(["python3", str(PAPER / "scripts/build_comparison_table.py"), "--check"],
                   check=True)
    published_data = (PAPER / "data/published_comparisons.json").read_bytes()
    published = json.loads(published_data)
    model_data = (PAPER / "data/model_audit.json").read_bytes()
    model_audit = json.loads(model_data)
    tex += "\n" + (PAPER / "generated/comparison_table.tex").read_text()
    tex += "\n" + (PAPER / "generated/model_table.tex").read_text()
    bib = (PAPER / "references.bib").read_text()
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}",tex) for key in group.split(",")}
    assert cited <= keys and len(cited) >= 10
    log = (PAPER / "main.log").read_text()
    assert "Overfull" not in log
    assert "undefined" not in log
    info = subprocess.check_output(["pdfinfo", str(PAPER / "main.pdf")], text=True)
    pages = int(re.search(r"Pages:\s+(\d+)",info).group(1))
    assert 1 <= pages <= 8
    assert re.search(r"Page size:\s+612 x 792 pts",info)
    assert re.search(r"Author:\s+Anonymous",info)
    assert re.search(r"Encrypted:\s+no",info)
    text = subprocess.check_output(["pdftotext", str(PAPER/"main.pdf"), "-"], text=True)
    assert "Anonymous Authors" in text
    assert not any(t in text for t in ["kzoacn","ICRA2027.git","/root/","??",
                                     "preliminary snapshot","remaining planned trials are pending"])
    assert "400" in text
    if task_refresh:
        assert str(meta["protocol"]["reused_episodes"]) in text
        versions = len({item["source_tree_sha256"] for item in meta["controller_sources"]})
        assert (f"{versions} controller versions" in text
                or versions == 2 and "two controller versions" in text)
        assert "All 400 episodes are newly executed" not in text
    fonts = subprocess.check_output(["pdffonts", str(PAPER/"main.pdf")], text=True)
    assert "Type 3" not in fonts
    for line in fonts.splitlines()[2:]:
        if line.strip():
            assert line.split()[-5] == "yes", f"Unembedded font: {line}"
    urls = subprocess.check_output(["pdfinfo","-url",str(PAPER/"main.pdf")],text=True)
    assert len([line for line in urls.splitlines()[1:] if line.strip()]) == 0, urls
    report = {"passed": True, "pages": pages, "episodes":len(rows),
        "successes":sum(r["success"] for r in rows), "tasks":len(groups),
        "controller_source_sha256":meta["controller_source_sha256"],
        "evaluation_kind":meta.get("evaluation_kind", "full_campaign"),
        "fresh_episodes":sum(not row["reused"] for row in rows),
        "historical_episodes":sum(row["reused"] for row in rows),
        "raw_record_archive_verified":bool(meta.get("raw_episode_archive")),
        "raw_records_verified":len(rows) if meta.get("raw_episode_archive") else 0,
        "references":len(cited), "author_metadata":"Anonymous",
        "published_comparison_rows":len(published["rows"]),
        "published_comparisons_sha256":hashlib.sha256(published_data).hexdigest(),
        "neural_parameters":model_audit["parameters"],
        "runtime_trainable_neural_parameters":model_audit["runtime_trainable_parameters"],
        "model_audit_sha256":hashlib.sha256(model_data).hexdigest(),
        "letter_paper":True, "all_fonts_embedded":True,"type3_fonts":False,
        "overfull_boxes":False,"undefined_references":False,
        "pdf_sha256":hashlib.sha256((PAPER/"main.pdf").read_bytes()).hexdigest(),
        "scope":"Local build, format and data checks; not PaperPlaza compliance certification or human scientific review."}
    (PAPER/"data/artifact_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
