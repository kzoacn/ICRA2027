#!/usr/bin/env python3
"""Check the final local artifact, its anonymous PDF, and the measured dataset."""
import collections
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
    groups = collections.defaultdict(set)
    for row in rows:
        groups[row["suite"], row["task_id"]].add(row["init_id"])
        assert type(row["success"]) is bool and row["seed"] == 7
        assert row["scoring"] == "external_sticky_any_success"
    assert len(groups) == 40 and all(v == set(range(10)) for v in groups.values())
    assert sum(r["reused"] for r in rows) == meta["protocol"]["reused_episodes"]
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
    tex = (PAPER / "main.tex").read_text()
    subprocess.run(["python3", str(PAPER / "scripts/build_comparison_table.py"), "--check"],
                   check=True)
    published_data = (PAPER / "data/published_comparisons.json").read_bytes()
    published = json.loads(published_data)
    tex += "\n" + (PAPER / "generated/comparison_table.tex").read_text()
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
    fonts = subprocess.check_output(["pdffonts", str(PAPER/"main.pdf")], text=True)
    assert "Type 3" not in fonts
    for line in fonts.splitlines()[2:]:
        if line.strip():
            assert line.split()[-5] == "yes", f"Unembedded font: {line}"
    urls = subprocess.check_output(["pdfinfo","-url",str(PAPER/"main.pdf")],text=True)
    assert len([line for line in urls.splitlines()[1:] if line.strip()]) == 0, urls
    report = {"passed": True, "pages": pages, "episodes":len(rows),
        "successes":sum(r["success"] for r in rows), "tasks":len(groups),
        "references":len(cited), "author_metadata":"Anonymous",
        "published_comparison_rows":len(published["rows"]),
        "published_comparisons_sha256":hashlib.sha256(published_data).hexdigest(),
        "letter_paper":True, "all_fonts_embedded":True,"type3_fonts":False,
        "overfull_boxes":False,"undefined_references":False,
        "pdf_sha256":hashlib.sha256((PAPER/"main.pdf").read_bytes()).hexdigest(),
        "scope":"Local build, format and data checks; not PaperPlaza compliance certification or human scientific review."}
    (PAPER/"data/artifact_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
