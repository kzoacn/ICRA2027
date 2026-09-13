"""Pinned 130-task manifest for evaluator-owned LIBERO campaigns.

This module deliberately lives on the evaluation side of the policy boundary.
Only :func:`build_libero_130_manifest` imports LIBERO benchmark metadata; route
policies receive only the instruction already exposed by ``PolicyTask``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence

import yaml


MANIFEST_SCHEMA = "libero-130-task-manifest.v1"
CANONICAL_SUITES: tuple[tuple[str, int], ...] = (
    ("libero_spatial", 10),
    ("libero_object", 10),
    ("libero_goal", 10),
    ("libero_90", 90),
    ("libero_10", 10),
)
CANONICAL_TASK_COUNT = sum(count for _suite, count in CANONICAL_SUITES)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LIBERO_ASSET_SUBDIRECTORIES = (
    "articulated_objects",
    "stable_scanned_objects",
    "turbosquid_objects",
    "stable_hope_objects",
    "scenes",
)


class ManifestError(ValueError):
    """Raised when a campaign manifest is incomplete or internally invalid."""


def _configured_path(value: object, *, config_root: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_root / path
    return path


def _validate_libero_asset_root(path: Path, *, source: str) -> Path:
    """Return a canonical, complete LIBERO runtime asset directory."""

    try:
        root = path.resolve(strict=True)
    except OSError as exc:
        raise ManifestError(
            f"LIBERO asset root from {source} is missing: {path}"
        ) from exc
    if not root.is_dir():
        raise ManifestError(
            f"LIBERO asset root from {source} is not a directory: {root}"
        )
    missing = [
        name for name in _LIBERO_ASSET_SUBDIRECTORIES if not (root / name).is_dir()
    ]
    if missing:
        raise ManifestError(
            f"LIBERO asset root from {source} is incomplete: {root}; "
            f"missing={missing}"
        )
    return root


def resolve_libero_asset_root(
    *, config_path: str | Path | None = None
) -> Path:
    """Locate the canonical asset tree used by the installed LIBERO runtime.

    Recent LIBERO releases first use ``<package>/assets`` and otherwise use
    ``~/.cache/libero/assets``.  Older/config-driven installs expose an
    explicit ``assets`` entry.  Resolve in that order without importing
    LIBERO or initiating a download; a formal campaign must start only after
    setup has produced a complete local tree.  Dataset/demonstration paths
    are deliberately never considered.
    """

    config_root = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else Path(os.environ.get("LIBERO_CONFIG_PATH", "~/.libero"))
        .expanduser()
        .resolve()
    )
    config_file = config_root / "config.yaml"
    try:
        raw_config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestError(f"cannot read LIBERO config {config_file}: {exc}") from exc
    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, Mapping):
        raise ManifestError(f"LIBERO config must be a mapping: {config_file}")

    benchmark_root = _configured_path(
        raw_config.get("benchmark_root"), config_root=config_root
    )
    package_assets = benchmark_root / "assets" if benchmark_root is not None else None
    configured_assets = _configured_path(
        raw_config.get("assets"), config_root=config_root
    )
    downloaded_assets = Path.home() / ".cache" / "libero" / "assets"

    candidates = (
        (package_assets, "config benchmark_root/assets"),
        (configured_assets, "config assets"),
        (downloaded_assets, "LIBERO default asset cache"),
    )
    for candidate, source in candidates:
        if candidate is None or not candidate.exists():
            continue
        return _validate_libero_asset_root(candidate, source=source)
    rendered = [str(path) for path, _source in candidates if path is not None]
    raise ManifestError(
        "cannot locate a local LIBERO asset root; checked " + ", ".join(rendered)
    )


def task_key(suite: str, task_id: int) -> str:
    """Return the canonical key used across shards and routes."""

    return f"{suite}:task{task_id:02d}"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_asset(path: Path, root: Path, *, label: str) -> str:
    try:
        relative = path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise ManifestError(f"{label} is missing or outside its LIBERO asset root: {path}") from exc
    return relative.as_posix()


def _manifest_from_benchmarks(
    benchmark_factories: Mapping[str, Callable[[], Any]],
    *,
    bddl_root: Path,
    init_states_root: Path,
    init_state_loader: Callable[[Any, int], Sequence[Any]],
) -> dict[str, Any]:
    """Build a manifest from injected evaluator metadata sources.

    The injected form keeps validation tests independent of a MuJoCo or LIBERO
    installation while the public builder below uses the installed benchmark.
    """

    entries: list[dict[str, Any]] = []
    for suite_name, expected_count in CANONICAL_SUITES:
        try:
            suite = benchmark_factories[suite_name]()
        except KeyError as exc:
            raise ManifestError(f"installed LIBERO has no {suite_name!r} benchmark") from exc
        tasks = tuple(suite.tasks)
        if len(tasks) != expected_count:
            raise ManifestError(
                f"{suite_name} exposes {len(tasks)} tasks, expected {expected_count}"
            )
        for task_id in range(expected_count):
            task = suite.get_task(task_id)
            bddl_path = bddl_root / task.problem_folder / task.bddl_file
            init_path = (
                init_states_root
                / task.problem_folder
                / Path(task.init_states_file).name
            )
            bddl_relative = _relative_asset(bddl_path, bddl_root, label="BDDL file")
            init_relative = _relative_asset(
                init_path, init_states_root, label="init-state file"
            )
            try:
                init_count = len(init_state_loader(suite, task_id))
            except Exception as exc:
                raise ManifestError(
                    f"cannot load init states for {task_key(suite_name, task_id)}: {exc}"
                ) from exc
            if init_count < 1:
                raise ManifestError(
                    f"{task_key(suite_name, task_id)} has no initial states"
                )
            entries.append(
                {
                    "task_key": task_key(suite_name, task_id),
                    "suite": suite_name,
                    "task_id": task_id,
                    "task_name": str(task.name),
                    "instruction": str(task.language),
                    "bddl": {
                        "relative_path": bddl_relative,
                        "sha256": file_sha256(bddl_path),
                    },
                    "init_states": {
                        "relative_path": init_relative,
                        "sha256": file_sha256(init_path),
                        "count": init_count,
                    },
                }
            )
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "task_key_fields": ["suite", "task_id"],
        "task_count": len(entries),
        "suites": [
            {"name": suite_name, "task_count": count}
            for suite_name, count in CANONICAL_SUITES
        ],
        "tasks": entries,
    }
    validate_manifest(manifest)
    return manifest


def build_libero_130_manifest(
    *, config_path: str | Path | None = None
) -> dict[str, Any]:
    """Read installed LIBERO metadata and return the canonical 130-task manifest."""

    if config_path is not None:
        config = Path(config_path).expanduser().resolve()
        if not (config / "config.yaml").is_file():
            raise ManifestError(f"LIBERO config.yaml is missing under {config}")
        os.environ["LIBERO_CONFIG_PATH"] = str(config)

    # Evaluator-only lazy import: none of this metadata is visible to a route.
    from libero.libero import benchmark, get_libero_path

    return _manifest_from_benchmarks(
        benchmark.get_benchmark_dict(),
        bddl_root=Path(get_libero_path("bddl_files")),
        init_states_root=Path(get_libero_path("init_states")),
        init_state_loader=lambda suite, index: suite.get_task_init_states(index),
    )


def validate_installed_assets(
    payload: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
) -> None:
    """Reject metadata or asset drift from a previously pinned manifest.

    This evaluator-only check reloads the installed benchmark, BDDL files, and
    init-state tensors.  Comparing the complete deterministic payload covers
    task ordering, language, relative paths, file hashes, and init counts.
    """

    validate_manifest(payload)
    installed = build_libero_130_manifest(config_path=config_path)
    if installed == payload:
        return
    expected_rows = {
        str(row["task_key"]): row for row in validate_manifest(payload)
    }
    installed_rows = {
        str(row["task_key"]): row for row in validate_manifest(installed)
    }
    mismatches = [
        key
        for key in expected_rows
        if expected_rows.get(key) != installed_rows.get(key)
    ]
    raise ManifestError(
        "installed LIBERO metadata/assets do not match the pinned manifest; "
        f"mismatched_tasks={mismatches[:5]}, mismatch_count={len(mismatches)}"
    )


def _valid_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts


def validate_manifest(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Validate and return manifest task rows in canonical suite/task order."""

    if not isinstance(payload, Mapping):
        raise ManifestError("manifest top-level value must be an object")
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(f"manifest schema must be {MANIFEST_SCHEMA!r}")
    if payload.get("task_key_fields") != ["suite", "task_id"]:
        raise ManifestError("manifest task_key_fields must be ['suite', 'task_id']")
    if payload.get("task_count") != CANONICAL_TASK_COUNT:
        raise ManifestError(f"manifest task_count must be {CANONICAL_TASK_COUNT}")
    expected_suite_rows = [
        {"name": suite_name, "task_count": count}
        for suite_name, count in CANONICAL_SUITES
    ]
    if payload.get("suites") != expected_suite_rows:
        raise ManifestError("manifest suites do not match the canonical 10/10/10/90/10 split")

    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ManifestError("manifest tasks must be a list")
    expected_keys = [
        task_key(suite_name, task_id)
        for suite_name, count in CANONICAL_SUITES
        for task_id in range(count)
    ]
    if len(raw_tasks) != CANONICAL_TASK_COUNT:
        raise ManifestError(f"manifest must contain exactly {CANONICAL_TASK_COUNT} task rows")

    actual_keys: list[str] = []
    validated: list[Mapping[str, Any]] = []
    for index, raw in enumerate(raw_tasks):
        if not isinstance(raw, Mapping):
            raise ManifestError(f"manifest task row {index} must be an object")
        suite = raw.get("suite")
        task_id = raw.get("task_id")
        if not isinstance(suite, str):
            raise ManifestError(f"manifest task row {index} has an invalid suite")
        if not isinstance(task_id, int) or isinstance(task_id, bool) or task_id < 0:
            raise ManifestError(f"manifest task row {index} has an invalid task_id")
        expected_key = task_key(suite, task_id)
        if raw.get("task_key") != expected_key:
            raise ManifestError(f"manifest task row {index} has an invalid task_key")
        actual_keys.append(expected_key)
        for field in ("task_name", "instruction"):
            if not isinstance(raw.get(field), str) or not raw[field].strip():
                raise ManifestError(f"{expected_key} has an invalid {field}")
        for field in ("bddl", "init_states"):
            asset = raw.get(field)
            if not isinstance(asset, Mapping):
                raise ManifestError(f"{expected_key}.{field} must be an object")
            if not _valid_relative_path(asset.get("relative_path")):
                raise ManifestError(f"{expected_key}.{field}.relative_path must be relative")
            digest = asset.get("sha256")
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ManifestError(f"{expected_key}.{field}.sha256 is invalid")
        init_count = raw["init_states"].get("count")
        if not isinstance(init_count, int) or isinstance(init_count, bool) or init_count < 1:
            raise ManifestError(f"{expected_key}.init_states.count must be positive")
        validated.append(raw)

    if actual_keys != expected_keys:
        missing = sorted(set(expected_keys) - set(actual_keys))
        duplicate_count = len(actual_keys) - len(set(actual_keys))
        raise ManifestError(
            "manifest tasks are not the canonical ordered suite/task set; "
            f"missing={missing[:3]}, duplicate_count={duplicate_count}"
        )
    return tuple(validated)


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestError("manifest top-level value must be an object")
    validate_manifest(payload)
    return payload


def write_manifest(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically write a validated deterministic manifest."""

    validate_manifest(payload)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(destination)
    return destination
