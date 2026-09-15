"""Resolve immutable evaluated source independently of the current repo layout."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "archive/evaluated-source/manifest.json"


@lru_cache(maxsize=1)
def evaluated_archive():
    manifest = json.loads(MANIFEST.read_text())
    path = MANIFEST.parent / manifest["archive"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    return manifest, path


def read_archived_file(relative_path):
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("An evaluated-source path must remain inside the archive.")
    _, archive = evaluated_archive()
    with zipfile.ZipFile(archive) as source:
        return source.read(path.as_posix())


def read_audited_source(relative_path):
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("An audited source path must be repository-relative.")
    current = ROOT / path
    return current.read_bytes() if current.is_file() else read_archived_file(path.as_posix())


def verify_evaluated_source():
    manifest, archive = evaluated_archive()
    prefix = manifest["package_prefix"]
    files = {}
    directories = {"."}
    with zipfile.ZipFile(archive) as source:
        for relative, expected in manifest["source_files"].items():
            payload = source.read(prefix + relative)
            assert hashlib.sha256(payload).hexdigest() == expected, relative
            files[relative] = payload
            directories.update(str(parent) for parent in PurePosixPath(relative).parents)
    entries = [("D:" + name, b"") for name in directories]
    entries += [("F:" + name, payload) for name, payload in files.items()]
    digest = hashlib.sha256(b"libero-source-tree-exact.v2\0")
    for name, payload in sorted(entries):
        encoded = name.encode()
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    actual = digest.hexdigest()
    assert actual == manifest["controller_source_sha256"], (actual, manifest["controller_source_sha256"])
    return actual


def migrate_text(text, destination, spec):
    text = text.replace("\r\n", "\n")
    for before, after in spec["global_replacements"]:
        if before in spec.get("identifier_prefix_replacements", []):
            # Preserve unrelated words such as route_campaign.
            text = re.sub(re.escape(before) + r"(?=_|\b)", after, text)
        else:
            text = text.replace(before, after)
    for before, after in spec["file_replacements"].get(destination, []):
        assert before in text, (destination, before)
        text = text.replace(before, after)
    return text


def verify_source_migration():
    evaluated_hash = verify_evaluated_source()
    manifest, _ = evaluated_archive()
    spec = json.loads((ROOT / "configs/naming_migration.json").read_text())
    prefix = manifest["package_prefix"]
    checked = []
    for relative in manifest["source_files"]:
        original = prefix + relative
        destination = spec["paths"][original]
        expected = migrate_text(read_archived_file(original).decode(), destination, spec)
        actual = (ROOT / destination).read_text()
        assert actual == expected, f"Source differs beyond recorded naming/path changes: {destination}"
        checked.append(destination)
    allowed = set(checked) | set(spec["new_entrypoint_files"])
    actual_files = {str(path.relative_to(ROOT)) for path in (ROOT / "src/anchor").rglob("*.py")}
    assert actual_files == allowed, (actual_files - allowed, allowed - actual_files)
    return {"evaluated_source_sha256": evaluated_hash, "mapped_source_files": len(checked)}


if __name__ == "__main__":
    print(json.dumps(verify_source_migration(), indent=2))
