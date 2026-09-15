#!/usr/bin/env python3
"""Check repository layout, active source syntax, links and archived evidence."""
import ast
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from source_archive import read_audited_source, verify_evaluated_source

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ("src", "scripts", "tests")
DIRECTORIES = ("src/anchor", "scripts", "configs", "docker", "tests",
               "experiments", "docs", "archive")
LEGACY_NAME = re.compile(
    r"libero_system|Route[BC][A-Z]|ROUTE_[BC]\b|route_[bc]_|\broute[-_ ][bc]\b|\bRoute[- ][BC]\b"
)
LINK = re.compile(r"\[[^\]\n]*\]\((?:<([^>]+)>|([^\s)]+))(?:\s+\"[^\"]*\")?\)")


def documentation_links():
    documents = [ROOT / "README.md"]
    for name in ("docs", "experiments", "archive"):
        documents.extend(sorted((ROOT / name).rglob("*.md")))
    checked, local_runtime = 0, 0
    failures = []
    for document in documents:
        text = re.sub(r"(?ms)^\s*```.*?^\s*```[^\n]*$", "", document.read_text())
        text = re.sub(r"`[^`\n]*`", "", text)
        for match in LINK.finditer(text):
            target = unquote(match.group(1) or match.group(2))
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path or parsed.path.startswith("/"):
                continue
            path = (document.parent / parsed.path).resolve()
            # Frozen reports link to host-local logs and videos excluded from Git.
            if path.is_relative_to(ROOT / "runtime"):
                local_runtime += 1
                continue
            checked += 1
            if not path.exists():
                failures.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not failures, "Broken documentation links:\n" + "\n".join(failures)
    return {"documents": len(documents), "local_links": checked,
            "runtime_links_excluded": local_runtime}


def main():
    root_files = sorted(path.name for path in ROOT.iterdir()
                        if path.is_file() and not path.name.startswith("."))
    assert root_files == ["README.md"], root_files
    for name in DIRECTORIES:
        assert (ROOT / name).is_dir(), name
    assert not (ROOT / "route-b-v170-cloud").exists()

    sources = [path for name in ACTIVE for path in sorted((ROOT / name).rglob("*.py"))]
    for path in sources:
        ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
        if path.is_relative_to(ROOT / "src"):
            assert not LEGACY_NAME.search(path.read_text()), f"Legacy source name: {path}"

    for line in (ROOT / "docker/Dockerfile").read_text().splitlines():
        if line.startswith("COPY "):
            for source in line.split()[1:-1]:
                assert (ROOT / source).exists(), f"Missing Docker COPY source: {source}"

    evaluated_hash = verify_evaluated_source()
    audit = json.loads((ROOT / "experiments/libero400/model_audit.json").read_text())
    detector = audit["implementation"]
    source_hash = hashlib.sha256(read_audited_source(detector["file"])).hexdigest()
    assert source_hash == detector["sha256"], "Model audit source mismatch"
    result = {"root_files": root_files, "python_files": len(sources),
              **documentation_links(), "evaluated_source_sha256": evaluated_hash}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
