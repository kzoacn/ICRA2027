"""Immutable execution plans for one formal 130-task route campaign."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import yaml

from .campaign_manifest import (
    CANONICAL_SUITES,
    file_sha256,
    load_manifest,
    resolve_libero_asset_root,
    task_key,
    validate_manifest,
)
from .formal_provenance import (
    POLICY_CONTRACT_SCHEMA,
    default_policy_contract_path,
    structured_sha256,
    validate_policy_contract,
)


PLAN_SCHEMA = "libero-route130-campaign-plan.v6"
ORCHESTRATION_SOURCE_SCHEMA = "libero-formal-orchestration-source.v2"
EPISODES_PER_TASK = 5
FORMAL_INIT_STATE_START = 20
FORMAL_EPISODE_INDICES = tuple(range(20, 25))
FORMAL_PER_ROUTE_MAX_WORKERS = 2
FORMAL_GLOBAL_MAX_WORKERS = 2
_SHA256 = re.compile(r"[0-9a-f]{64}")

# ``-I`` ignores inherited PYTHONPATH, user site-packages, and the current
# working directory.  This fixed bootstrap adds only the source-locked
# repository root back, after checking that the child really started there.
# Keeping the bootstrap in the immutable plan also makes the exact isolation
# mechanism part of every command commitment.
FORMAL_PYTHON_ISOLATION_ARGS = (
    "-I",
    "-S",
    "-B",
    "-X",
    "pycache_prefix=/dev/null",
    "-c",
)

FORMAL_PYTHON_BOOTSTRAP = """\
import importlib
import json
import os
import runpy
import sys
root = sys.argv.pop(1)
locked_paths = json.loads(sys.argv.pop(1))
if not os.path.isabs(root) or os.path.realpath(root) != root:
    raise SystemExit("non-canonical formal repository root")
if os.path.realpath(os.getcwd()) != root:
    raise SystemExit("formal child working directory mismatch")
if not isinstance(locked_paths, list) or not locked_paths:
    raise SystemExit("formal child has no locked runtime paths")
for item in locked_paths:
    if not isinstance(item, str) or not os.path.isabs(item) or os.path.realpath(item) != item:
        raise SystemExit("formal child runtime path is not canonical")
sys.path[:] = [root, *locked_paths]
sys.path_importer_cache.clear()
importlib.invalidate_caches()
sys.argv[0] = "libero_system.integration"
runpy.run_module("libero_system.integration", run_name="__main__")
"""

FORMAL_SCRIPT_BOOTSTRAP = """\
import importlib
import json
import os
import runpy
import sys
root = sys.argv.pop(1)
locked_paths = json.loads(sys.argv.pop(1))
module = sys.argv.pop(1)
if not os.path.isabs(root) or os.path.realpath(root) != root:
    raise SystemExit("non-canonical formal repository root")
if not isinstance(locked_paths, list) or not locked_paths:
    raise SystemExit("formal script has no locked runtime paths")
for item in locked_paths:
    if not isinstance(item, str) or not os.path.isabs(item) or os.path.realpath(item) != item:
        raise SystemExit("formal script runtime path is not canonical")
if module not in {
    "scripts.run_route130_campaign",
    "scripts.verify_route_campaign",
    "scripts.verify_route_run",
}:
    raise SystemExit("unapproved formal script module")
sys.path[:] = [root, *locked_paths]
sys.path_importer_cache.clear()
importlib.invalidate_caches()
sys.argv[0] = module
runpy.run_module(module, run_name="__main__")
"""


def formal_script_command(
    *,
    python_executable: str,
    bootstrap_sys_path: Sequence[str],
    module: str,
    arguments: Sequence[str] = (),
) -> list[str]:
    """Build the sole supported isolated command for a formal script."""

    return [
        python_executable,
        *FORMAL_PYTHON_ISOLATION_ARGS,
        FORMAL_SCRIPT_BOOTSTRAP,
        str(canonical_repository_root()),
        json.dumps(
            list(bootstrap_sys_path),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ),
        module,
        *arguments,
    ]


def canonical_repository_root() -> Path:
    """Return the sole repository root accepted by the formal protocol."""

    return Path(__file__).resolve().parents[2]


def _has_symlink_component(path: Path) -> bool:
    """Check all existing components without first resolving aliases."""

    absolute = path if path.is_absolute() else Path.cwd() / path
    parts = absolute.parts
    current = Path(parts[0])
    for part in parts[1:]:
        current = current / part
        try:
            if current.is_symlink():
                return True
        except OSError:
            return True
        if not current.exists():
            break
    return False


def require_canonical_path(
    value: object,
    *,
    label: str,
    must_exist: bool = False,
) -> Path:
    """Reject relative, lexical, or symlink aliases before returning a path."""

    if not isinstance(value, str) or not value:
        raise CampaignPlanError(f"{label} must be a non-empty path string")
    raw = Path(value)
    if not raw.is_absolute():
        raise CampaignPlanError(f"{label} must be absolute")
    try:
        resolved = raw.resolve(strict=must_exist)
    except OSError as exc:
        raise CampaignPlanError(f"{label} cannot be resolved: {exc}") from exc
    if str(raw) != str(resolved):
        raise CampaignPlanError(f"{label} must use its exact canonical path")
    if _has_symlink_component(raw):
        raise CampaignPlanError(f"{label} must not contain a symlink component")
    return resolved


def orchestration_source_sha256(
    *, source_tree_digest: str | None = None
) -> str:
    """Hash the policy tree commitment plus three formal entrypoint scripts.

    ``integration.config.source_tree_sha256`` intentionally fingerprints only
    the importable policy package.  A formal campaign additionally depends on
    the runner and both verifier entrypoints, so changing resume, scheduling,
    logging, or verification code must invalidate a previously frozen plan.
    """

    if source_tree_digest is None:
        # Local import avoids a module-initialization cycle: config imports
        # campaign constants while the plan builder also consumes config.
        from .config import source_tree_sha256

        source_tree_digest = source_tree_sha256()
    if not isinstance(source_tree_digest, str) or _SHA256.fullmatch(
        source_tree_digest
    ) is None:
        raise CampaignPlanError("source tree commitment for orchestration is invalid")
    repository = canonical_repository_root()
    scripts_root = repository / "scripts"
    paths: list[Path] = []
    for entry in sorted(scripts_root.iterdir(), key=lambda item: item.name):
        metadata = entry.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise CampaignPlanError(
                f"formal orchestration directory contains a symlink: {entry}"
            )
        if stat.S_ISDIR(metadata.st_mode):
            if entry.name == "__pycache__":
                continue
            raise CampaignPlanError(
                f"formal orchestration directory contains an unexpected directory: {entry}"
            )
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise CampaignPlanError(
                f"formal orchestration source is not a single-link regular file: {entry}"
            )
        if entry.suffix in {".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib"}:
            raise CampaignPlanError(
                f"formal orchestration directory contains executable shadow bytes: {entry}"
            )
        if entry.suffix not in {".py", ".sh"}:
            raise CampaignPlanError(
                f"formal orchestration directory contains an unexpected file: {entry}"
            )
        paths.append(entry)
    required_names = {
        "run_route130_campaign.py",
        "verify_route_campaign.py",
        "verify_route_run.py",
    }
    if not required_names.issubset({path.name for path in paths}):
        raise CampaignPlanError("formal orchestration entrypoint is missing")
    digest = hashlib.sha256(f"{ORCHESTRATION_SOURCE_SCHEMA}\0".encode("ascii"))
    digest.update(b"source_tree_sha256\0")
    digest.update(source_tree_digest.encode("ascii"))
    for path in sorted(paths, key=lambda item: item.relative_to(repository).as_posix()):
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or resolved != path:
            raise CampaignPlanError(f"formal orchestration source is not a file: {path}")
        relative = path.relative_to(repository).as_posix().encode("utf-8")
        payload = resolved.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def formal_execution_schedule_contract() -> dict[str, Any]:
    """Return the frozen two-route concurrency contract.

    Each route may run two shards concurrently, but the two route campaigns
    themselves must be sequential.  Therefore the observable global maximum
    is also two, not four.  Strict verification additionally requires the
    runner-produced execution-schedule intervals proving that non-overlap.
    """

    return {
        "per_route_max_workers": FORMAL_PER_ROUTE_MAX_WORKERS,
        "global_max_workers": FORMAL_GLOBAL_MAX_WORKERS,
        "route_order": ["b", "c"],
        "routes_must_not_overlap": True,
        "evidence": "runner_execution_schedule_intervals",
    }


def formal_campaign_environment(libero_config_path: str | Path) -> dict[str, str]:
    """Return the complete allowlisted environment for every shard child."""

    return {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": "/nonexistent",
        "TMPDIR": "/tmp",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "LIBERO_CONFIG_PATH": str(libero_config_path),
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "NUMBA_DISABLE_JIT": "1",
        "NUMBA_CACHE_DIR": "/tmp/libero-route130-numba",
        "MPLCONFIGDIR": "/tmp/libero-route130-matplotlib",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


class CampaignPlanError(ValueError):
    """Raised when a formal campaign plan is invalid or has drifted."""


def directory_tree_provenance(path: str | Path) -> dict[str, Any]:
    """Hash every relative file name and target byte in a directory tree.

    File and directory symlinks are followed.  In particular, Hugging Face
    snapshot symlinks hash the actual cached blob bytes rather than only the
    link text.  Ancestor inode tracking rejects symlink cycles.
    """

    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise CampaignPlanError(f"dependency tree is not a directory: {root}")
    digest = hashlib.sha256(b"libero-directory-tree.v1\0")
    file_count = 0
    total_bytes = 0

    def visit(current: Path, relative: Path, ancestors: frozenset[tuple[int, int]]) -> None:
        nonlocal file_count, total_bytes
        try:
            resolved = current.resolve(strict=True)
            stat = resolved.stat()
        except OSError as exc:
            raise CampaignPlanError(f"cannot resolve dependency entry {current}: {exc}") from exc
        identity = (int(stat.st_dev), int(stat.st_ino))
        if resolved.is_dir():
            if identity in ancestors:
                raise CampaignPlanError(f"symlink cycle in dependency tree at {current}")
            next_ancestors = ancestors | {identity}
            try:
                children = sorted(current.iterdir(), key=lambda item: item.name)
            except OSError as exc:
                raise CampaignPlanError(f"cannot list dependency directory {current}: {exc}") from exc
            for child in children:
                visit(child, relative / child.name, next_ancestors)
            return
        if not resolved.is_file():
            raise CampaignPlanError(f"unsupported dependency entry: {current}")
        name = relative.as_posix().encode("utf-8")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        size = int(stat.st_size)
        digest.update(size.to_bytes(8, "big"))
        try:
            with current.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise CampaignPlanError(f"cannot hash dependency file {current}: {exc}") from exc
        file_count += 1
        total_bytes += size

    visit(root, Path(), frozenset())
    return {
        "path": str(root),
        "tree_sha256": digest.hexdigest(),
        "file_count": file_count,
        "bytes": total_bytes,
        "symlink_policy": "follow-target-bytes",
    }


_FORMAL_RUNTIME_CACHE_DIRECTORIES = {
    "__pycache__",
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}


def formal_runtime_dependency_root_provenance(
    path: str | Path,
) -> dict[str, Any]:
    """Recompute an exact byte-level lock for one import root or binary.

    Unlike the historical lock this includes cache files, native extensions,
    directory names, and every other regular entry.  A new sourceless module,
    ABI extension, or directory symlink therefore cannot appear without
    changing (or invalidating) the commitment.
    """

    root = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256(b"libero-formal-runtime-dependency-tree.v1\0")
    file_count = 0
    total_bytes = 0
    if root.is_file():
        metadata = root.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise CampaignPlanError(f"formal runtime root is unsupported: {root}")
        candidates: list[tuple[str, Path | None]] = [(f"F:{root.name}", root)]
        kind = "file"
    elif root.is_dir():
        candidates = []
        for current, directories, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            for dirname in directories:
                candidate = current_path / dirname
                metadata = candidate.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    raise CampaignPlanError(
                        f"formal runtime tree contains a directory symlink: {candidate}"
                    )
                if not stat.S_ISDIR(metadata.st_mode):
                    raise CampaignPlanError(
                        f"formal runtime tree contains an invalid directory: {candidate}"
                    )
            directories[:] = sorted(directories)
            relative_directory = current_path.relative_to(root).as_posix()
            candidates.append((f"D:{relative_directory}", None))
            for filename in sorted(files):
                candidate = current_path / filename
                metadata = candidate.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(
                    metadata.st_mode
                ):
                    raise CampaignPlanError(
                        f"formal runtime tree contains an unsupported entry: {candidate}"
                    )
                candidates.append(
                    (f"F:{candidate.relative_to(root).as_posix()}", candidate)
                )
        kind = "tree"
    else:
        raise CampaignPlanError(f"formal runtime root is unsupported: {root}")
    for tagged_name, candidate in sorted(candidates, key=lambda item: item[0]):
        name = tagged_name.encode("utf-8")
        payload = b"" if candidate is None else candidate.read_bytes()
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        if candidate is not None:
            file_count += 1
            total_bytes += len(payload)
    if file_count < 1:
        raise CampaignPlanError(f"formal runtime root is empty: {root}")
    return {
        "path": str(root),
        "kind": kind,
        "tree_sha256": digest.hexdigest(),
        "file_count": file_count,
        "bytes": total_bytes,
        "include": "exact-directory-and-all-regular-file-bytes",
        "excluded_suffixes": [],
        "excluded_directories": [],
        "symlink_policy": "reject-all-descendant-symlinks",
    }


def python_stdlib_root_provenance(path: str | Path) -> dict[str, Any]:
    """Hash a complete stdlib root while excluding separately locked installs."""

    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise CampaignPlanError(f"Python stdlib root is not a directory: {root}")
    digest = hashlib.sha256(b"libero-python-stdlib-tree.v1\0")
    file_count = 0
    total_bytes = 0
    directory_count = 0
    excluded = {"site-packages", "dist-packages", "__pycache__"}
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in tuple(directories):
            candidate = current_path / name
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise CampaignPlanError(
                    f"Python stdlib contains a directory symlink: {candidate}"
                )
            if not stat.S_ISDIR(metadata.st_mode):
                raise CampaignPlanError(
                    f"Python stdlib contains an invalid directory: {candidate}"
                )
        directories[:] = sorted(name for name in directories if name not in excluded)
        relative_directory = current_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(b"D")
        digest.update(len(relative_directory).to_bytes(4, "big"))
        digest.update(relative_directory)
        directory_count += 1
        for name in sorted(filenames):
            candidate = current_path / name
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise CampaignPlanError(
                    f"Python stdlib contains an unsupported entry: {candidate}"
                )
            relative = candidate.relative_to(root).as_posix().encode("utf-8")
            payload = candidate.read_bytes()
            digest.update(b"F")
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
            file_count += 1
            total_bytes += len(payload)
    if file_count < 1:
        raise CampaignPlanError(f"Python stdlib root is empty: {root}")
    return {
        "path": str(root),
        "tree_sha256": digest.hexdigest(),
        "file_count": file_count,
        "directory_count": directory_count,
        "bytes": total_bytes,
        "excluded_directory_names": sorted(excluded),
        "symlink_policy": "reject-all-descendant-symlinks",
    }


def media_binary_provenance(role: str, path: str | Path) -> dict[str, Any]:
    """Hash one canonical formal-runtime executable and freeze its version."""

    binary = require_canonical_path(
        str(Path(path).expanduser()), label=f"{role} executable", must_exist=True
    )
    if not binary.is_file():
        raise CampaignPlanError(f"{role} is not a regular file: {binary}")
    version_argv = ["--version"] if role == "bwrap" else ["-version"]
    clean_environment = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    process = subprocess.run(
        [str(binary), *version_argv],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
        env=clean_environment,
    )
    if process.returncode:
        detail = process.stderr.strip() or str(process.returncode)
        raise CampaignPlanError(f"{role} version probe failed: {detail}")
    lines = (process.stdout or process.stderr).splitlines()
    if not lines or not lines[0].strip():
        raise CampaignPlanError(f"{role} returned no version line")
    closure_targets = [binary]
    script_interpreter: dict[str, Any] | None = None
    try:
        first_line = binary.open("rb").readline(4096)
    except OSError as exc:
        raise CampaignPlanError(f"cannot inspect {role} executable: {exc}") from exc
    if first_line.startswith(b"#!"):
        interpreter = first_line[2:].decode("utf-8", errors="strict").strip().split()[0]
        declared_interpreter = Path(interpreter)
        if not declared_interpreter.is_absolute():
            raise CampaignPlanError(f"{role} script interpreter must be absolute")
        interpreter_path = declared_interpreter.resolve(strict=True)
        script_interpreter = {
            "declared_path": interpreter,
            "resolved_path": str(interpreter_path),
            "sha256": file_sha256(interpreter_path),
            "bytes": int(interpreter_path.stat().st_size),
        }
        closure_targets.append(interpreter_path)
    libraries: dict[str, dict[str, Any]] = {}
    ldd_path = Path("/usr/bin/ldd")
    if not ldd_path.is_file():
        raise CampaignPlanError("/usr/bin/ldd is required for native closure locking")
    for target in closure_targets:
        ldd = subprocess.run(
            [str(ldd_path), str(target)],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
            env=clean_environment,
        )
        output = "\n".join((ldd.stdout, ldd.stderr))
        for line in output.splitlines():
            match = re.search(r"(?:=>\s+)?(/[^\s()]+)\s+\(", line)
            if match is None:
                continue
            library = Path(match.group(1)).resolve(strict=True)
            if not library.is_file():
                continue
            libraries[str(library)] = {
                "path": str(library),
                "sha256": file_sha256(library),
                "bytes": int(library.stat().st_size),
            }
        if ldd.returncode and "not a dynamic executable" not in output:
            raise CampaignPlanError(
                f"cannot inspect {role} native closure: {output.strip()}"
            )
    native_files = [libraries[key] for key in sorted(libraries)]
    closure_bound = {
        "probe": "clean-environment-ldd-resolved-files",
        "ldd_path": str(ldd_path),
        "ldd_sha256": file_sha256(ldd_path),
        "script_interpreter": script_interpreter,
        "files": native_files,
    }
    native_closure = {
        "schema": "libero-native-loader-closure.v1",
        **closure_bound,
        "sha256": structured_sha256(
            "libero-native-loader-closure.v1", closure_bound
        ),
    }
    return {
        "role": role,
        "path": str(binary),
        "sha256": file_sha256(binary),
        "bytes": int(binary.stat().st_size),
        "version_argv": version_argv,
        "version": lines[0].strip(),
        "native_closure": native_closure,
    }


def validate_formal_runtime_binding(binding: Mapping[str, Any]) -> None:
    """Re-hash the exact interpreter, import roots, and media executables."""

    python_path = require_canonical_path(
        binding.get("python_executable"),
        label="formal runtime Python executable",
        must_exist=True,
    )
    if file_sha256(python_path) != binding.get("python_executable_sha256"):
        raise CampaignPlanError("formal runtime Python executable bytes changed")
    python_runtime = binding.get("python_runtime")
    if not isinstance(python_runtime, Mapping):
        raise CampaignPlanError("formal Python runtime closure is missing")
    stdlib_roots = python_runtime.get("stdlib_roots")
    if not isinstance(stdlib_roots, list) or not stdlib_roots:
        raise CampaignPlanError("formal Python stdlib closure is invalid")
    for root in stdlib_roots:
        if not isinstance(root, Mapping) or python_stdlib_root_provenance(
            root.get("path", "")
        ) != root:
            raise CampaignPlanError("formal Python stdlib bytes changed")
    pyvenv_record = python_runtime.get("pyvenv_cfg")
    if pyvenv_record is not None:
        if (
            not isinstance(pyvenv_record, Mapping)
            or file_sha256(pyvenv_record.get("path", ""))
            != pyvenv_record.get("sha256")
        ):
            raise CampaignPlanError("formal Python pyvenv.cfg bytes changed")
    native_closure = python_runtime.get("native_closure")
    native_files = (
        native_closure.get("files")
        if isinstance(native_closure, Mapping)
        else None
    )
    if not isinstance(native_files, list) or not native_files:
        raise CampaignPlanError("formal Python native closure is invalid")
    for record in native_files:
        if (
            not isinstance(record, Mapping)
            or file_sha256(record.get("path", "")) != record.get("sha256")
        ):
            raise CampaignPlanError("formal Python native closure bytes changed")
    if native_closure.get("sha256") != structured_sha256(
        "libero-python-process-native-closure.v1", native_files
    ):
        raise CampaignPlanError("formal Python native closure commitment is invalid")
    runtime_bound = {
        "isolation_argv": python_runtime.get("isolation_argv"),
        "bootstrap_sys_path": python_runtime.get("bootstrap_sys_path"),
        "stdlib_roots": stdlib_roots,
        "pyvenv_cfg": pyvenv_record,
        "native_closure": native_closure,
    }
    if python_runtime.get("sha256") != structured_sha256(
        "libero-python-runtime-closure.v1", runtime_bound
    ):
        raise CampaignPlanError("formal Python runtime closure commitment is invalid")
    environment = binding.get("formal_runtime_environment")
    modules = environment.get("modules") if isinstance(environment, Mapping) else None
    if not isinstance(modules, list):
        raise CampaignPlanError("formal runtime environment binding is invalid")
    for module in modules:
        if not isinstance(module, Mapping) or not isinstance(module.get("roots"), list):
            raise CampaignPlanError("formal runtime module binding is invalid")
        for root in module["roots"]:
            if not isinstance(root, Mapping):
                raise CampaignPlanError("formal runtime root binding is invalid")
            current = formal_runtime_dependency_root_provenance(root.get("path", ""))
            if current != root:
                raise CampaignPlanError(
                    f"formal runtime dependency bytes changed: {module.get('name')}"
                )
    distributions = environment.get("distributions")
    bootstrap_sys_path = environment.get("bootstrap_sys_path")
    if not isinstance(distributions, list) or not isinstance(bootstrap_sys_path, list):
        raise CampaignPlanError("formal runtime distribution binding is invalid")
    for distribution in distributions:
        if not isinstance(distribution, Mapping):
            raise CampaignPlanError("formal runtime distribution record is invalid")
        record_path = distribution.get("record_path", "")
        if file_sha256(record_path) != distribution.get("record_sha256"):
            raise CampaignPlanError(
                f"formal distribution RECORD changed: {distribution.get('name')}"
            )
        recorded_files = distribution.get("recorded_files")
        if not isinstance(recorded_files, list) or distribution.get(
            "recorded_files_sha256"
        ) != structured_sha256(
            "libero-distribution-recorded-files.v1", recorded_files
        ):
            raise CampaignPlanError("formal distribution file commitment is invalid")
        for file_record in recorded_files:
            if (
                not isinstance(file_record, Mapping)
                or file_sha256(file_record.get("path", ""))
                != file_record.get("sha256")
            ):
                raise CampaignPlanError(
                    f"formal distribution bytes changed: {distribution.get('name')}"
                )
        companions = distribution.get("companion_libraries")
        if not isinstance(companions, list):
            raise CampaignPlanError("formal distribution companion lock is invalid")
        for root in companions:
            if not isinstance(root, Mapping) or formal_runtime_dependency_root_provenance(
                root.get("path", "")
            ) != root:
                raise CampaignPlanError(
                    f"formal distribution companion bytes changed: {distribution.get('name')}"
                )
    environment_bound = {
        "modules": modules,
        "distributions": distributions,
        "bootstrap_sys_path": bootstrap_sys_path,
    }
    if environment.get("sha256") != structured_sha256(
        "libero-formal-runtime-environment.v2", environment_bound
    ):
        raise CampaignPlanError("formal runtime environment commitment is invalid")
    media = binding.get("media_binaries")
    executables = media.get("executables") if isinstance(media, Mapping) else None
    if not isinstance(executables, list):
        raise CampaignPlanError("formal media-binary binding is invalid")
    for record in executables:
        if not isinstance(record, Mapping):
            raise CampaignPlanError("formal media-binary record is invalid")
        current = media_binary_provenance(
            str(record.get("role", "")), str(record.get("path", ""))
        )
        if current != record:
            raise CampaignPlanError(
                f"formal media executable changed: {record.get('role')}"
            )
    if media.get("sha256") != structured_sha256(
        "libero-formal-media-binaries.v3", executables
    ):
        raise CampaignPlanError("formal media-binary commitment is invalid")


def _python_provenance(executable: Path) -> dict[str, Any]:
    # Run the complete probe with the *planned* interpreter.  Package roots
    # are resolved there (important for editable installs), and MuJoCo's
    # OSMesa module is imported there solely to identify the actual mapped
    # graphics libraries; the planning interpreter never resolves them.
    query = r"""
import os
import sys

# ``-S`` deliberately suppresses site.py (and every .pth file).  Add only the
# canonical environment site-packages directories; the resulting exact path
# vector is recorded below and reused by every formal child bootstrap.
runtime_prefix = os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))
python_tag = f'python{sys.version_info.major}.{sys.version_info.minor}'
isolated_base_sys_path = [
    os.path.realpath(item) for item in sys.path
    if isinstance(item, str) and item and os.path.exists(item)
]
explicit_site_candidates = (
    os.path.join(runtime_prefix, 'lib', python_tag, 'site-packages'),
    os.path.join(runtime_prefix, 'Lib', 'site-packages'),
)
for explicit_site in explicit_site_candidates:
    if os.path.isdir(explicit_site):
        canonical_site = os.path.realpath(explicit_site)
        if canonical_site not in sys.path:
            sys.path.append(canonical_site)

import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess

def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')

def structured_hash(domain, value):
    digest = hashlib.sha256()
    digest.update(domain.encode('ascii'))
    digest.update(b'\0')
    digest.update(canonical(value))
    return digest.hexdigest()

def normalized_name(value):
    return re.sub(r'[-_.]+', '-', str(value).strip()).lower()

def stable_direct_url(raw):
    if not isinstance(raw, dict):
        return None
    result = {}
    if isinstance(raw.get('url'), str):
        result['url'] = raw['url']
    if isinstance(raw.get('subdirectory'), str):
        result['subdirectory'] = raw['subdirectory']
    archive = raw.get('archive_info')
    if isinstance(archive, dict):
        stable = {}
        if isinstance(archive.get('hash'), str):
            stable['hash'] = archive['hash']
        hashes = archive.get('hashes')
        if isinstance(hashes, dict) and all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in hashes.items()
        ):
            stable['hashes'] = dict(sorted(hashes.items()))
        result['archive_info'] = stable
    directory = raw.get('dir_info')
    if isinstance(directory, dict):
        result['dir_info'] = {
            'editable': directory.get('editable') is True,
        }
    vcs = raw.get('vcs_info')
    if isinstance(vcs, dict):
        stable = {}
        for key in ('vcs', 'commit_id', 'requested_revision'):
            if isinstance(vcs.get(key), str):
                stable[key] = vcs[key]
        result['vcs_info'] = stable
    return result or None

inventory = []
for distribution in metadata.distributions():
    name = distribution.metadata.get('Name')
    version = distribution.version
    if not isinstance(name, str) or not name.strip():
        raise RuntimeError('installed distribution has no stable Name metadata')
    if not isinstance(version, str) or not version:
        raise RuntimeError(f'installed distribution {name!r} has no stable version')
    direct_url = None
    raw_direct_url = distribution.read_text('direct_url.json')
    if raw_direct_url:
        direct_url = stable_direct_url(json.loads(raw_direct_url))
    inventory.append({
        'name': normalized_name(name),
        'version': version,
        'direct_url': direct_url,
    })
inventory.sort(key=lambda item: canonical(item))
inventory_record = {
    'schema': 'libero-installed-distribution-inventory.v1',
    'count': len(inventory),
    'sha256': structured_hash(
        'libero-installed-distribution-inventory.v1', inventory
    ),
    'entries': inventory,
}

excluded_directories = {
    '__pycache__',
    'assets',
    'bddl_files',
    'datasets',
    'init_files',
}
cache_directories = {
    '__pycache__',
    '.cache',
    '.git',
    '.mypy_cache',
    '.pytest_cache',
    '.ruff_cache',
}

def source_root_record(root):
    root = Path(root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f'runtime package root is not a directory: {root}')
    digest = hashlib.sha256(b'libero-runtime-python-tree.v1\0')
    file_count = 0
    total_bytes = 0
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(
            item for item in directories if item not in excluded_directories
        )
        current_path = Path(current)
        for filename in sorted(files):
            if not filename.endswith('.py'):
                continue
            path = current_path / filename
            relative = path.relative_to(root).as_posix().encode('utf-8')
            payload = path.read_bytes()
            digest.update(len(relative).to_bytes(4, 'big'))
            digest.update(relative)
            digest.update(len(payload).to_bytes(8, 'big'))
            digest.update(payload)
            file_count += 1
            total_bytes += len(payload)
    if file_count < 1:
        raise RuntimeError(f'runtime package has no Python source files: {root}')
    return {
        'path': str(root),
        'tree_sha256': digest.hexdigest(),
        'file_count': file_count,
        'bytes': total_bytes,
        'include': '**/*.py',
        'excluded_directories': sorted(excluded_directories),
        'symlink_policy': 'resolved-root-no-directory-follow',
    }

runtime_packages = []
for package_name in ('libero', 'robosuite'):
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        raise RuntimeError(f'required runtime package is not importable: {package_name}')
    locations = list(spec.submodule_search_locations or ())
    if not locations and spec.origin:
        locations = [str(Path(spec.origin).parent)]
    roots = [source_root_record(path) for path in sorted(set(locations))]
    if not roots:
        raise RuntimeError(f'cannot resolve runtime package root: {package_name}')
    runtime_packages.append({'name': package_name, 'roots': roots})
runtime_code = {
    'schema': 'libero-runtime-package-code.v1',
    'packages': runtime_packages,
    'sha256': structured_hash('libero-runtime-package-code.v1', runtime_packages),
}

def resource_root_record(root):
    root = Path(root).expanduser().resolve(strict=True)
    digest = hashlib.sha256(b'libero-robosuite-runtime-resources.v1\0')
    file_count = 0
    total_bytes = 0
    excluded_relative_directories = {'models/assets/demonstrations'}
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        relative_current = current_path.relative_to(root).as_posix()
        directories[:] = sorted(
            item
            for item in directories
            if item not in cache_directories
            and (
                f'{relative_current}/{item}'.lstrip('/')
                not in excluded_relative_directories
            )
        )
        for filename in sorted(files):
            if filename.endswith(('.py', '.pyc', '.pyo')):
                continue
            path = current_path / filename
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix().encode('utf-8')
            payload = path.read_bytes()
            digest.update(len(relative).to_bytes(4, 'big'))
            digest.update(relative)
            digest.update(len(payload).to_bytes(8, 'big'))
            digest.update(payload)
            file_count += 1
            total_bytes += len(payload)
    if file_count < 1:
        raise RuntimeError(f'robosuite has no runtime resource files: {root}')
    return {
        'path': str(root),
        'tree_sha256': digest.hexdigest(),
        'file_count': file_count,
        'bytes': total_bytes,
        'include': 'all-regular-files-except-python-and-caches',
        'excluded_suffixes': ['.py', '.pyc', '.pyo'],
        'excluded_directories': sorted(cache_directories),
        'excluded_relative_directories': sorted(excluded_relative_directories),
        'symlink_policy': 'resolved-root-no-directory-follow',
    }

robosuite_roots = next(
    package['roots'] for package in runtime_packages if package['name'] == 'robosuite'
)
robosuite_resources = [
    resource_root_record(root['path']) for root in robosuite_roots
]
runtime_resources = {
    'schema': 'libero-robosuite-runtime-resources.v1',
    'roots': robosuite_resources,
    'sha256': structured_hash(
        'libero-robosuite-runtime-resources.v1', robosuite_resources
    ),
}

def complete_runtime_root_record(root):
    root = Path(root).expanduser().resolve(strict=True)
    digest = hashlib.sha256(b'libero-mujoco-runtime-package.v1\0')
    file_count = 0
    total_bytes = 0
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(
            item for item in directories if item not in cache_directories
        )
        current_path = Path(current)
        for filename in sorted(files):
            if filename.endswith(('.pyc', '.pyo')):
                continue
            path = current_path / filename
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix().encode('utf-8')
            payload = path.read_bytes()
            digest.update(len(relative).to_bytes(4, 'big'))
            digest.update(relative)
            digest.update(len(payload).to_bytes(8, 'big'))
            digest.update(payload)
            file_count += 1
            total_bytes += len(payload)
    if file_count < 1:
        raise RuntimeError(f'MuJoCo package has no runtime files: {root}')
    return {
        'path': str(root),
        'tree_sha256': digest.hexdigest(),
        'file_count': file_count,
        'bytes': total_bytes,
        'include': 'all-regular-runtime-files',
        'excluded_suffixes': ['.pyc', '.pyo'],
        'excluded_directories': sorted(cache_directories),
        'symlink_policy': 'resolved-root-no-directory-follow',
    }

mujoco_spec = importlib.util.find_spec('mujoco')
if mujoco_spec is None:
    raise RuntimeError('required runtime package is not importable: mujoco')
mujoco_locations = list(mujoco_spec.submodule_search_locations or ())
if not mujoco_locations and mujoco_spec.origin:
    mujoco_locations = [str(Path(mujoco_spec.origin).parent)]
mujoco_roots = [
    complete_runtime_root_record(path)
    for path in sorted(set(mujoco_locations))
]
mujoco_runtime_package = {
    'schema': 'libero-mujoco-runtime-package.v1',
    'roots': mujoco_roots,
    'sha256': structured_hash('libero-mujoco-runtime-package.v1', mujoco_roots),
}

def formal_dependency_root_record(root):
    root = Path(root).expanduser().resolve(strict=True)
    digest = hashlib.sha256(b'libero-formal-runtime-dependency-tree.v1\0')
    file_count = 0
    total_bytes = 0
    if root.is_file():
        candidates = [(root, Path(root.name))]
        kind = 'file'
    elif root.is_dir():
        candidates = []
        for current, directories, files in os.walk(root, followlinks=False):
            directories[:] = sorted(
                item for item in directories if item not in cache_directories
            )
            current_path = Path(current)
            for filename in sorted(files):
                if filename.endswith(('.pyc', '.pyo')):
                    continue
                path = current_path / filename
                if path.is_file():
                    candidates.append((path, path.relative_to(root)))
        kind = 'tree'
    else:
        raise RuntimeError(f'formal runtime root is unsupported: {root}')
    for path, relative_path in candidates:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise RuntimeError(f'formal runtime entry is not a file: {path}')
        name = relative_path.as_posix().encode('utf-8')
        payload = resolved.read_bytes()
        digest.update(len(name).to_bytes(4, 'big'))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, 'big'))
        digest.update(payload)
        file_count += 1
        total_bytes += len(payload)
    if file_count < 1:
        raise RuntimeError(f'formal runtime root is empty: {root}')
    return {
        'path': str(root),
        'kind': kind,
        'tree_sha256': digest.hexdigest(),
        'file_count': file_count,
        'bytes': total_bytes,
        'include': 'all-regular-runtime-files',
        'excluded_suffixes': ['.pyc', '.pyo'],
        'excluded_directories': sorted(cache_directories),
        'symlink_policy': 'resolved-root-no-directory-follow',
    }

package_owners = metadata.packages_distributions()

def distribution_record(distribution_name):
    distribution = metadata.distribution(distribution_name)
    files = list(distribution.files or ())
    record_candidates = [
        item for item in files if str(item).replace('\\', '/').endswith('.dist-info/RECORD')
    ]
    if len(record_candidates) != 1:
        raise RuntimeError(
            f'distribution {distribution_name!r} must expose exactly one RECORD'
        )
    record_path = Path(distribution.locate_file(record_candidates[0])).resolve(strict=True)
    file_rows = []
    companion_roots = set()
    for relative in sorted(files, key=lambda item: str(item).replace('\\', '/')):
        relative_text = str(relative).replace('\\', '/')
        path = Path(distribution.locate_file(relative)).resolve(strict=True)
        if not path.is_file():
            raise RuntimeError(
                f'distribution {distribution_name!r} RECORD entry is not a file: {relative_text}'
            )
        payload = path.read_bytes()
        file_rows.append({
            'relative_path': relative_text,
            'path': str(path),
            'bytes': len(payload),
            'sha256': hashlib.sha256(payload).hexdigest(),
        })
        for parent in (path, *path.parents):
            if parent.name.endswith('.libs'):
                companion_roots.add(str(parent.resolve(strict=True)))
                break
    normalized = normalized_name(distribution.metadata.get('Name'))
    return {
        'name': normalized,
        'version': distribution.version,
        'record_path': str(record_path),
        'record_sha256': hashlib.sha256(record_path.read_bytes()).hexdigest(),
        'recorded_file_count': len(file_rows),
        'recorded_bytes': sum(item['bytes'] for item in file_rows),
        'recorded_files': file_rows,
        'recorded_files_sha256': structured_hash(
            'libero-distribution-recorded-files.v1', file_rows
        ),
        'companion_roots': sorted(companion_roots),
    }

seed_modules = (
    'transformers',
    'torch',
    'numpy',
    'cv2',
    'imageio',
    'imageio_ffmpeg',
    'scipy',
    'yaml',
)
for module_name in seed_modules:
    importlib.import_module(module_name)

loaded_owned_modules = sorted({
    name.split('.', 1)[0]
    for name in sys.modules
    if name.split('.', 1)[0] in package_owners
})
formal_module_names = list(seed_modules) + [
    name for name in loaded_owned_modules if name not in seed_modules
]
formal_modules = []
owner_names = set()
for module_name in formal_module_names:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise RuntimeError(
            f'required formal runtime module is not importable: {module_name}'
        )
    locations = list(spec.submodule_search_locations or ())
    if locations:
        root_paths = sorted(set(locations))
    elif spec.origin:
        root_paths = [spec.origin]
    else:
        raise RuntimeError(f'cannot resolve formal runtime module: {module_name}')
    roots = [formal_dependency_root_record(path) for path in root_paths]
    owners = sorted({normalized_name(name) for name in package_owners.get(module_name, ())})
    if not owners:
        raise RuntimeError(f'formal runtime module has no owning distribution: {module_name}')
    owner_names.update(owners)
    imported = importlib.import_module(module_name)
    module_version = getattr(imported, '__version__', None)
    formal_modules.append({
        'name': module_name,
        'origin': str(Path(spec.origin).resolve(strict=True)) if spec.origin else None,
        'reported_version': str(module_version) if module_version is not None else None,
        'owners': owners,
        'roots': roots,
    })
distribution_records = [
    distribution_record(name) for name in sorted(owner_names)
]
bootstrap_sys_path = []
for item in sys.path:
    if not isinstance(item, str) or not item:
        continue
    candidate = Path(item).expanduser()
    if not candidate.exists():
        continue
    canonical_path = str(candidate.resolve(strict=True))
    if canonical_path not in bootstrap_sys_path:
        bootstrap_sys_path.append(canonical_path)
formal_runtime_environment = {
    'schema': 'libero-formal-runtime-environment.v2',
    'modules': formal_modules,
    'distributions': distribution_records,
    'bootstrap_sys_path': bootstrap_sys_path,
    'sha256': structured_hash(
        'libero-formal-runtime-environment.v2', {
            'modules': formal_modules,
            'distributions': distribution_records,
            'bootstrap_sys_path': bootstrap_sys_path,
        }
    ),
}

def executable_record(role, value, version_argv):
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise RuntimeError(f'{role} is not a regular file: {path}')
    process = subprocess.run(
        [str(path), *version_argv],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if process.returncode:
        raise RuntimeError(
            f'{role} version probe failed: '
            f'{process.stderr.strip() or process.returncode}'
        )
    version_lines = (process.stdout or process.stderr).splitlines()
    if not version_lines or not version_lines[0].strip():
        raise RuntimeError(f'{role} returned no version line')
    payload = path.read_bytes()
    return {
        'role': role,
        'path': str(path),
        'sha256': hashlib.sha256(payload).hexdigest(),
        'bytes': len(payload),
        'version_argv': version_argv,
        'version': version_lines[0].strip(),
    }

ffprobe_path = shutil.which('ffprobe')
if ffprobe_path is None:
    raise RuntimeError('ffprobe is required by the formal runtime')
imageio_ffmpeg_module = importlib.import_module('imageio_ffmpeg')
imageio_ffmpeg_path = imageio_ffmpeg_module.get_ffmpeg_exe()
media_records = [
    executable_record('ffprobe', ffprobe_path, ['-version']),
    executable_record('imageio_ffmpeg', imageio_ffmpeg_path, ['-version']),
    executable_record('bwrap', '/usr/bin/bwrap', ['--version']),
]
media_binaries = {
    'schema': 'libero-formal-media-binaries.v2',
    'executables': media_records,
    'sha256': structured_hash(
        'libero-formal-media-binaries.v2', media_records
    ),
}

os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')
graphics_errors = []
try:
    importlib.import_module('mujoco')
    importlib.import_module('mujoco.osmesa')
except Exception as exc:
    graphics_errors.append(
        f'import:{type(exc).__name__}:{exc}'
    )

libraries_by_path = {}
maps_path = Path('/proc/self/maps')
map_lines = []
if not maps_path.is_file():
    graphics_errors.append('proc_self_maps_unavailable')
else:
    try:
        map_lines = maps_path.read_text(encoding='utf-8').splitlines()
    except OSError as exc:
        graphics_errors.append(f'proc_self_maps:{type(exc).__name__}:{exc}')
        map_lines = []
    for line in map_lines:
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or not fields[-1].startswith('/'):
            continue
        candidate = Path(fields[-1])
        try:
            path = candidate.resolve(strict=True)
        except OSError:
            continue
        basename = path.name.lower()
        if basename.startswith('libosmesa.so'):
            role = 'osmesa'
        elif basename.startswith(('libglapi.so', 'libgl.so', 'libopengl.so')):
            role = 'opengl_api'
        elif basename.startswith('libstdc++.so'):
            role = 'cxx_runtime'
        else:
            continue
        if str(path) in libraries_by_path:
            continue
        digest = hashlib.sha256()
        size = 0
        try:
            with path.open('rb') as stream:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size += len(chunk)
        except OSError as exc:
            graphics_errors.append(
                f'library_hash:{path}:{type(exc).__name__}:{exc}'
            )
            continue
        libraries_by_path[str(path)] = {
            'role': role,
            'path': str(path),
            'basename': path.name,
            'size': size,
            'sha256': digest.hexdigest(),
        }
graphics_libraries = sorted(
    libraries_by_path.values(), key=lambda item: (item['role'], item['path'])
)
present_roles = {item['role'] for item in graphics_libraries}
missing_roles = sorted({'osmesa', 'opengl_api', 'cxx_runtime'} - present_roles)
if missing_roles:
    graphics_errors.append(f'missing_roles:{",".join(missing_roles)}')
osmesa_dynamic_libraries = {
    'schema': 'libero-osmesa-dynamic-libraries.v1',
    'backend': 'osmesa',
    'probe': 'import-mujoco-osmesa-and-hash-proc-self-maps-realpaths',
    'status': 'complete' if not graphics_errors else 'incomplete',
    'libraries': graphics_libraries,
    'missing_roles': missing_roles,
    'errors': graphics_errors,
    'sha256': structured_hash(
        'libero-osmesa-dynamic-libraries.v1', graphics_libraries
    ),
}

mapped_runtime_files = {}
for line in map_lines:
    fields = line.split(maxsplit=5)
    if len(fields) < 6 or not fields[-1].startswith('/'):
        continue
    try:
        mapped = Path(fields[-1]).resolve(strict=True)
    except OSError:
        continue
    if not mapped.is_file() or str(mapped) in mapped_runtime_files:
        continue
    payload = mapped.read_bytes()
    mapped_runtime_files[str(mapped)] = {
        'path': str(mapped),
        'bytes': len(payload),
        'sha256': hashlib.sha256(payload).hexdigest(),
    }
mapped_rows = [mapped_runtime_files[path] for path in sorted(mapped_runtime_files)]
process_native_closure = {
    'schema': 'libero-python-process-native-closure.v1',
    'probe': 'post-import-proc-self-maps-resolved-regular-files',
    'files': mapped_rows,
    'sha256': structured_hash(
        'libero-python-process-native-closure.v1', mapped_rows
    ),
}

names = [
    'hf_libero', 'libero', 'mujoco', 'robosuite', 'torch', 'transformers',
    'numpy', 'cv2', 'imageio', 'imageio_ffmpeg',
]
versions = {}
for name in names:
    if name == 'cv2':
        versions[name] = str(importlib.import_module('cv2').__version__)
    else:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None

print(json.dumps({
    'version': platform.python_version(),
    'implementation': platform.python_implementation(),
    'reported_executable': sys.executable,
    'runtime_prefix': runtime_prefix,
    'isolated_base_sys_path': isolated_base_sys_path,
    'packages': versions,
    'distribution_inventory': inventory_record,
    'runtime_package_code': runtime_code,
    'robosuite_runtime_resources': runtime_resources,
    'mujoco_runtime_package': mujoco_runtime_package,
    'osmesa_dynamic_libraries': osmesa_dynamic_libraries,
    'formal_runtime_environment': formal_runtime_environment,
    'process_native_closure': process_native_closure,
    'media_binaries': media_binaries,
}, ensure_ascii=False, allow_nan=False, sort_keys=True))
"""
    probe_environment = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": "/nonexistent",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "NUMBA_DISABLE_JIT": "1",
        "NUMBA_CACHE_DIR": "/tmp/libero-route130-numba",
        "MPLCONFIGDIR": "/tmp/libero-route130-matplotlib",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    process = subprocess.run(
        [
            str(executable),
            "-I",
            "-S",
            "-B",
            "-X",
            "pycache_prefix=/dev/null",
            "-c",
            query,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=canonical_repository_root(),
        env=probe_environment,
    )
    if process.returncode:
        detail = process.stderr.strip() or f"exit code {process.returncode}"
        raise CampaignPlanError(f"cannot inspect Python dependency environment: {detail}")
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise CampaignPlanError("Python dependency probe returned invalid JSON") from exc
    runtime_environment = payload.get("formal_runtime_environment")
    if not isinstance(runtime_environment, dict):
        raise CampaignPlanError("Python dependency probe omitted runtime environment")
    modules = runtime_environment.get("modules")
    distributions = runtime_environment.get("distributions")
    bootstrap_sys_path = runtime_environment.get("bootstrap_sys_path")
    if (
        not isinstance(modules, list)
        or not isinstance(distributions, list)
        or not isinstance(bootstrap_sys_path, list)
    ):
        raise CampaignPlanError("Python dependency probe returned an invalid runtime lock")
    for module in modules:
        if not isinstance(module, dict) or not isinstance(module.get("roots"), list):
            raise CampaignPlanError("Python dependency probe returned an invalid module")
        module["roots"] = [
            formal_runtime_dependency_root_provenance(root["path"])
            for root in module["roots"]
        ]
    for distribution in distributions:
        if not isinstance(distribution, dict):
            raise CampaignPlanError("Python dependency probe returned an invalid distribution")
        companion_paths = distribution.pop("companion_roots", None)
        if not isinstance(companion_paths, list):
            raise CampaignPlanError("Python dependency probe omitted companion roots")
        distribution["companion_libraries"] = [
            formal_runtime_dependency_root_provenance(path)
            for path in companion_paths
        ]
    runtime_bound = {
        "modules": modules,
        "distributions": distributions,
        "bootstrap_sys_path": bootstrap_sys_path,
    }
    runtime_environment["sha256"] = structured_sha256(
        "libero-formal-runtime-environment.v2", runtime_bound
    )

    media = payload.get("media_binaries")
    media_records = media.get("executables") if isinstance(media, Mapping) else None
    if not isinstance(media_records, list):
        raise CampaignPlanError("Python dependency probe omitted media executables")
    rebound_media = [
        media_binary_provenance(str(record["role"]), str(record["path"]))
        for record in media_records
    ]
    payload["media_binaries"] = {
        "schema": "libero-formal-media-binaries.v3",
        "executables": rebound_media,
        "sha256": structured_sha256(
            "libero-formal-media-binaries.v3", rebound_media
        ),
    }

    base_paths = payload.pop("isolated_base_sys_path", None)
    if not isinstance(base_paths, list) or not base_paths:
        raise CampaignPlanError("Python dependency probe omitted isolated stdlib paths")
    directory_paths = sorted(
        {
            str(Path(item).resolve(strict=True))
            for item in base_paths
            if isinstance(item, str) and Path(item).is_dir()
        },
        key=lambda item: (len(Path(item).parts), item),
    )
    stdlib_paths: list[str] = []
    for item in directory_paths:
        candidate = Path(item)
        if any(
            candidate == Path(parent) or Path(parent) in candidate.parents
            for parent in stdlib_paths
        ):
            continue
        stdlib_paths.append(item)
    stdlib_roots = [python_stdlib_root_provenance(item) for item in stdlib_paths]
    runtime_prefix = Path(str(payload.pop("runtime_prefix"))).resolve(strict=True)
    pyvenv_path = runtime_prefix / "pyvenv.cfg"
    pyvenv_record = None
    if pyvenv_path.is_file():
        pyvenv_record = {
            "path": str(pyvenv_path),
            "sha256": file_sha256(pyvenv_path),
            "bytes": int(pyvenv_path.stat().st_size),
        }
    native_closure = payload.pop("process_native_closure", None)
    if not isinstance(native_closure, Mapping):
        raise CampaignPlanError("Python dependency probe omitted native process closure")
    python_runtime_bound = {
        "isolation_argv": [
            "-I",
            "-S",
            "-B",
            "-X",
            "pycache_prefix=/dev/null",
            "-c",
        ],
        "bootstrap_sys_path": bootstrap_sys_path,
        "stdlib_roots": stdlib_roots,
        "pyvenv_cfg": pyvenv_record,
        "native_closure": native_closure,
    }
    payload["python_runtime"] = {
        "schema": "libero-python-runtime-closure.v1",
        **python_runtime_bound,
        "sha256": structured_sha256(
            "libero-python-runtime-closure.v1", python_runtime_bound
        ),
    }
    return {
        "executable": str(executable),
        "executable_sha256": file_sha256(executable),
        **payload,
    }


def _configured_task_asset_root(
    value: object,
    *,
    config_root: Path,
    benchmark_root: Path | None,
    fallback_name: str,
    label: str,
) -> Path:
    if isinstance(value, str) and value.strip():
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_root / path
    elif benchmark_root is not None:
        path = benchmark_root / fallback_name
    else:
        raise CampaignPlanError(
            f"LIBERO config does not declare {label} or benchmark_root"
        )
    try:
        root = path.resolve(strict=True)
    except OSError as exc:
        raise CampaignPlanError(f"installed {label} root is missing: {path}") from exc
    if not root.is_dir():
        raise CampaignPlanError(f"installed {label} root is not a directory: {root}")
    return root


def installed_task_assets_provenance(
    manifest: Mapping[str, Any],
    *,
    config_path: str | Path,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Bind all 130 installed BDDL/init files to their manifest declarations."""

    rows = validate_manifest(manifest)
    config_root = Path(config_path).expanduser().resolve(strict=True)
    config_file = config_root / "config.yaml"
    try:
        raw_config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CampaignPlanError(f"cannot read LIBERO config {config_file}: {exc}") from exc
    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, Mapping):
        raise CampaignPlanError(f"LIBERO config must be a mapping: {config_file}")
    raw_benchmark_root = raw_config.get("benchmark_root")
    benchmark_root: Path | None = None
    if isinstance(raw_benchmark_root, str) and raw_benchmark_root.strip():
        benchmark_root = Path(raw_benchmark_root).expanduser()
        if not benchmark_root.is_absolute():
            benchmark_root = config_root / benchmark_root
    bddl_root = _configured_task_asset_root(
        raw_config.get("bddl_files"),
        config_root=config_root,
        benchmark_root=benchmark_root,
        fallback_name="bddl_files",
        label="BDDL files",
    )
    init_root = _configured_task_asset_root(
        raw_config.get("init_states"),
        config_root=config_root,
        benchmark_root=benchmark_root,
        fallback_name="init_files",
        label="init-state files",
    )

    def bind(kind: str, root: Path) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        total_bytes = 0
        for row in rows:
            declaration = row[kind]
            relative = str(declaration["relative_path"])
            path = root.joinpath(*PurePosixPath(relative).parts)
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError) as exc:
                raise CampaignPlanError(
                    f"installed {kind} file is missing or outside its root: {path}"
                ) from exc
            if not resolved.is_file():
                raise CampaignPlanError(
                    f"installed {kind} entry is not a file: {resolved}"
                )
            digest = file_sha256(resolved)
            if digest != declaration["sha256"]:
                raise CampaignPlanError(
                    f"installed {kind} file differs from manifest for "
                    f"{row['task_key']}: expected={declaration['sha256']}, "
                    f"current={digest}"
                )
            size = int(resolved.stat().st_size)
            total_bytes += size
            entries.append(
                {
                    "task_key": row["task_key"],
                    "relative_path": relative,
                    "sha256": digest,
                    "bytes": size,
                }
            )
        return {
            "root": str(root),
            "file_count": len(entries),
            "bytes": total_bytes,
            "entries_sha256": structured_sha256(
                f"libero-installed-{kind}-entries.v1", entries
            ),
            "symlink_policy": "resolved-file-must-remain-under-root",
        }

    bound = {
        "manifest_sha256": manifest_sha256,
        "bddl": bind("bddl", bddl_root),
        "init_states": bind("init_states", init_root),
    }
    return {
        "schema": "libero-installed-task-assets.v1",
        **bound,
        "aggregate_sha256": structured_sha256(
            "libero-installed-task-assets.v1", bound
        ),
    }


def route_c_process_dependency_provenance(
    *,
    asset_root: str | Path,
    perception_model: str | Path,
    python_executable: str | Path,
) -> dict[str, Any]:
    """Derive every immutable Route-C sandbox commitment from plan inputs."""

    from .formal_process_sandbox import (
        BWRAP_PATH,
        POLICY_SOURCE_FILES,
        SYSTEM_RUNTIME_FILES,
        _gallery_files,
        _model_repository,
        environment_sha256,
        fixed_child_environment,
        sha256_file,
        tree_manifest_sha256,
        validate_policy_runtime_clock_source,
    )

    repository = canonical_repository_root()
    assets = Path(asset_root).resolve(strict=True)
    model = _model_repository(Path(perception_model))
    python = Path(python_executable).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="libero-route-c-plan-") as temporary:
        projection = Path(temporary)
        policy_projection = projection / "policy"
        gallery_projection = projection / "gallery"
        for relative in POLICY_SOURCE_FILES:
            source = repository / relative
            metadata = source.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise CampaignPlanError(
                    f"Route C projected source is not an exact regular file: {source}"
                )
            destination = policy_projection / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        gallery_sources = _gallery_files(assets)
        for source in gallery_sources:
            metadata = source.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise CampaignPlanError(
                    f"Route C gallery source is not an exact regular file: {source}"
                )
            destination = gallery_projection / source.relative_to(assets)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        source_sha256, source_file_count = tree_manifest_sha256(policy_projection)
        gallery_sha256, gallery_file_count = tree_manifest_sha256(gallery_projection)
    model_sha256, model_file_count = tree_manifest_sha256(model)
    system_runtime_files = [
        {
            "destination": destination,
            "sha256": sha256_file(Path(destination).resolve(strict=True)),
            "read_only": True,
        }
        for destination in SYSTEM_RUNTIME_FILES
    ]
    identity_payloads = {
        "/etc/passwd": b"nobody:x:65534:65534:Route C policy:/tmp:/usr/sbin/nologin\n",
        "/etc/group": b"nogroup:x:65534:\n",
    }
    fixed_identity_files = [
        {
            "destination": destination,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "read_only": True,
        }
        for destination, payload in identity_payloads.items()
    ]
    environment = fixed_child_environment()
    bound = {
        "source_sha256": source_sha256,
        "source_file_count": source_file_count,
        "clock_source_gate_sha256": validate_policy_runtime_clock_source(
            repository
            / "libero_system/integration/route_c_policy_runtime.py"
        ),
        "gallery_source_root": str(assets),
        "gallery_sha256": gallery_sha256,
        "gallery_file_count": gallery_file_count,
        "model_source_root": str(model),
        "model_sha256": model_sha256,
        "model_file_count": model_file_count,
        "python_sha256": sha256_file(python),
        "bwrap_path": str(BWRAP_PATH),
        "bwrap_sha256": sha256_file(BWRAP_PATH),
        "environment_values": environment,
        "environment_sha256": environment_sha256(environment),
        "system_runtime_files": system_runtime_files,
        "fixed_identity_files": fixed_identity_files,
        "policy_runner": "production_route_c_policy_episode",
        "bundle_build_count": 1,
        "model_load_count": 1,
    }
    return {
        "schema": "libero-formal-route-c-plan-binding.v1",
        **bound,
        "sha256": structured_sha256(
            "libero-formal-route-c-plan-binding.v1", bound
        ),
    }


def collect_dependency_provenance(
    *,
    python_executable: str | Path,
    libero_config_path: str | Path,
    perception_model: str | Path,
    manifest: Mapping[str, Any],
    manifest_path: str | Path,
) -> dict[str, Any]:
    python = Path(python_executable).expanduser().resolve(strict=True)
    config_file = (
        Path(libero_config_path).expanduser().resolve(strict=True) / "config.yaml"
    )
    if not config_file.is_file():
        raise CampaignPlanError(f"LIBERO config file is missing: {config_file}")
    try:
        asset_root = resolve_libero_asset_root(config_path=config_file.parent)
    except ValueError as exc:
        raise CampaignPlanError(str(exc)) from exc
    return {
        "python": _python_provenance(python),
        "libero_config": {
            "path": str(config_file),
            "sha256": file_sha256(config_file),
        },
        # This is the XML / mesh / texture / material tree used to construct
        # MuJoCo scenes and the static perception gallery.  Keep it entirely
        # separate from demonstrations, BDDL files, and init-state tensors.
        "libero_assets": directory_tree_provenance(asset_root),
        "installed_task_assets": installed_task_assets_provenance(
            manifest,
            config_path=config_file.parent,
            manifest_sha256=file_sha256(manifest_path),
        ),
        "perception_model": directory_tree_provenance(perception_model),
        "route_c_process": route_c_process_dependency_provenance(
            asset_root=asset_root,
            perception_model=perception_model,
            python_executable=python,
        ),
    }


def validate_plan_dependencies(
    plan: Mapping[str, Any], *, deep_model_check: bool = True
) -> None:
    """Reject executable, config, LIBERO-asset, or model drift from a plan.

    LIBERO assets, all manifest-declared BDDL/init files, runtime Python
    source, and the installed-distribution inventory are checked between
    campaign shards. ``deep_model_check`` controls only the much larger frozen
    perception-model tree, used at campaign start and final verification.
    """

    validate_campaign_plan(plan)
    protocol = plan["protocol"]
    expected = plan["dependency_provenance"]
    python = Path(protocol["python_executable"]).resolve(strict=True)
    config_file = Path(protocol["libero_config_path"]).resolve(strict=True) / "config.yaml"
    manifest_file = Path(str(plan["manifest"]["path"])).resolve(strict=True)
    shallow_mismatches: list[str] = []
    if orchestration_source_sha256() != plan["orchestration_source_sha256"]:
        shallow_mismatches.append("formal orchestration source")
    if file_sha256(python) != expected["python"]["executable_sha256"]:
        shallow_mismatches.append("python executable")
    if file_sha256(config_file) != expected["libero_config"]["sha256"]:
        shallow_mismatches.append("LIBERO config.yaml")
    if file_sha256(manifest_file) != plan["manifest"]["sha256"]:
        shallow_mismatches.append("campaign manifest")
    policy_contract = plan["policy_contract"]
    try:
        validate_policy_contract(
            policy_contract["path"], policy_contract["sha256"]
        )
    except (OSError, ValueError) as exc:
        raise CampaignPlanError(str(exc)) from exc
    if shallow_mismatches:
        raise CampaignPlanError(
            f"campaign dependency drift: {', '.join(shallow_mismatches)}"
        )

    # Runtime Python code and installed-distribution inventory are cheap
    # enough to re-probe between every shard.  This catches an editable
    # LIBERO/robosuite checkout or pip/conda mutation without rehashing the
    # large model tree.
    current_python = _python_provenance(python)
    if current_python != expected["python"]:
        mismatches = [
            field
            for field in expected["python"]
            if expected["python"].get(field) != current_python.get(field)
        ]
        raise CampaignPlanError(
            "Python runtime dependency provenance changed: "
            f"{mismatches}"
        )

    expected_assets = expected["libero_assets"]
    try:
        asset_root = resolve_libero_asset_root(
            config_path=protocol["libero_config_path"]
        )
        current_assets = directory_tree_provenance(asset_root)
    except (OSError, ValueError) as exc:
        raise CampaignPlanError(f"LIBERO asset dependency drift: {exc}") from exc
    if current_assets != expected_assets:
        raise CampaignPlanError(
            "LIBERO asset dependency provenance changed: "
            f"planned={expected_assets.get('tree_sha256')}, "
            f"current={current_assets.get('tree_sha256')}"
        )
    manifest = load_manifest(manifest_file)
    current_task_assets = installed_task_assets_provenance(
        manifest,
        config_path=protocol["libero_config_path"],
        manifest_sha256=plan["manifest"]["sha256"],
    )
    if current_task_assets != expected["installed_task_assets"]:
        raise CampaignPlanError(
            "installed BDDL/init dependency provenance changed"
        )
    if not deep_model_check:
        return
    current = {
        "python": current_python,
        "libero_config": {
            "path": str(config_file),
            "sha256": file_sha256(config_file),
        },
        "libero_assets": current_assets,
        "installed_task_assets": current_task_assets,
        "perception_model": directory_tree_provenance(
            protocol["perception"]["model"]
        ),
        "route_c_process": route_c_process_dependency_provenance(
            asset_root=asset_root,
            perception_model=protocol["perception"]["model"],
            python_executable=python,
        ),
    }
    if current != expected:
        mismatches = [
            field for field in expected if expected.get(field) != current.get(field)
        ]
        raise CampaignPlanError(
            f"campaign dependency provenance changed: {mismatches}"
        )


def canonical_shards() -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return 13 non-overlapping shards covering the canonical 130 tasks."""

    shards: list[tuple[str, tuple[int, ...]]] = [
        ("libero_spatial", tuple(range(10))),
        ("libero_object", tuple(range(10))),
        ("libero_goal", tuple(range(10))),
    ]
    shards.extend(
        ("libero_90", tuple(range(start, start + 10)))
        for start in range(0, 90, 10)
    )
    shards.append(("libero_10", tuple(range(10))))
    return tuple(shards)


def _range_argument(task_ids: Sequence[int]) -> str:
    ids = tuple(task_ids)
    if ids == tuple(range(ids[0], ids[-1] + 1)):
        return f"{ids[0]}-{ids[-1]}"
    return ",".join(str(task_id) for task_id in ids)


def _shard_name(route: str, suite: str, task_ids: Sequence[int]) -> str:
    return f"route_{route}_{suite}_tasks{task_ids[0]:02d}_{task_ids[-1]:02d}"


def build_campaign_plan(
    *,
    route: str,
    campaign_directory: str | Path,
    manifest_path: str | Path,
    perception_model: str | Path,
    libero_config_path: str | Path,
    python_executable: str | Path,
    init_state_start: int,
    seed: int,
    image_size: int = 256,
    video_fps: int = 20,
    video_stride: int = 2,
    device: str = "cpu",
    policy_contract_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create a deterministic, source-locked plan without starting episodes."""

    if route not in {"b", "c"}:
        raise CampaignPlanError("route must be 'b' or 'c'")
    if init_state_start != FORMAL_INIT_STATE_START:
        raise CampaignPlanError(
            "formal init_state_start must be exactly 20 "
            "(episode indices 20,21,22,23,24)"
        )
    if image_size < 64 or video_fps < 1 or video_stride < 1:
        raise CampaignPlanError("invalid image/video protocol")
    if device != "cpu" and re.fullmatch(r"cuda(?::\d+)?", device) is None:
        raise CampaignPlanError("device must be cpu, cuda, or cuda:N")

    campaign_root = Path(campaign_directory).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    model = Path(perception_model).expanduser().resolve(strict=True)
    config_root = Path(libero_config_path).expanduser().resolve(strict=True)
    if not (config_root / "config.yaml").is_file():
        raise CampaignPlanError(f"LIBERO config.yaml is missing under {config_root}")
    python = Path(python_executable).expanduser().resolve(strict=True)
    contract_path = Path(
        policy_contract_path or default_policy_contract_path()
    ).expanduser().resolve(strict=True)
    contract = validate_policy_contract(contract_path)

    # Importing config is evaluator-owned.  Omitting --max-steps from each
    # command makes these values the actual config defaults, while pinning
    # them here makes a later default change a detectable plan/source drift.
    from .config import DEFAULT_MAX_STEPS, source_tree_sha256

    expected_suites = {suite for suite, _count in CANONICAL_SUITES}
    if set(DEFAULT_MAX_STEPS) != expected_suites:
        raise CampaignPlanError(
            "integration.config.DEFAULT_MAX_STEPS must cover exactly the five canonical suites"
        )
    manifest = load_manifest(manifest_file)
    required_init_count = init_state_start + EPISODES_PER_TASK
    insufficient_init_states = [
        str(row["task_key"])
        for row in manifest["tasks"]
        if int(row["init_states"]["count"]) < required_init_count
    ]
    if insufficient_init_states:
        raise CampaignPlanError(
            "formal init-state block is unavailable for "
            f"{insufficient_init_states[:5]}"
        )
    dependencies = collect_dependency_provenance(
        python_executable=python,
        libero_config_path=config_root,
        perception_model=model,
        manifest=manifest,
        manifest_path=manifest_file,
    )
    source_digest = source_tree_sha256()
    runs_root = campaign_root / "runs"
    shards: list[dict[str, Any]] = []
    for index, (suite, task_ids) in enumerate(canonical_shards()):
        run_name = _shard_name(route, suite, task_ids)
        run_directory = runs_root / run_name
        command = [
            str(python),
            *FORMAL_PYTHON_ISOLATION_ARGS,
            FORMAL_PYTHON_BOOTSTRAP,
            str(canonical_repository_root()),
            json.dumps(
                dependencies["python"]["python_runtime"]["bootstrap_sys_path"],
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ),
            "--route",
            route,
            "--suite",
            suite,
            "--task-ids",
            _range_argument(task_ids),
            "--episodes-per-task",
            str(EPISODES_PER_TASK),
            "--init-state-start",
            str(init_state_start),
            "--output-dir",
            str(runs_root),
            "--run-name",
            run_name,
            "--device",
            device,
            "--perception-backend",
            "grounding-dino",
            "--perception-model",
            str(model),
            "--libero-config-path",
            str(config_root),
            "--image-size",
            str(image_size),
            "--seed",
            str(seed),
            "--video",
            "--video-fps",
            str(video_fps),
            "--video-stride",
            str(video_stride),
            "--render-backend",
            "osmesa",
        ]
        shards.append(
            {
                "index": index,
                "shard_id": f"{suite}:tasks{task_ids[0]:02d}-{task_ids[-1]:02d}",
                "suite": suite,
                "task_ids": list(task_ids),
                "task_keys": [task_key(suite, task_id) for task_id in task_ids],
                "episodes": len(task_ids) * EPISODES_PER_TASK,
                "run_name": run_name,
                "run_directory": str(run_directory),
                "base_command": command,
            }
        )
    return {
        "schema": PLAN_SCHEMA,
        "route": route,
        "campaign_directory": str(campaign_root),
        "working_directory": str(canonical_repository_root()),
        "manifest": {
            "path": str(manifest_file),
            "sha256": file_sha256(manifest_file),
            "task_count": int(manifest["task_count"]),
            "minimum_init_state_count": min(
                int(row["init_states"]["count"])
                for row in manifest["tasks"]
            ),
        },
        "source_tree_sha256": source_digest,
        "orchestration_source_sha256": orchestration_source_sha256(
            source_tree_digest=source_digest
        ),
        "dependency_provenance": dependencies,
        "policy_contract": {
            "schema": contract["schema"],
            "path": str(contract_path),
            "sha256": file_sha256(contract_path),
        },
        "protocol": {
            "episodes_per_task": EPISODES_PER_TASK,
            "init_state_start": init_state_start,
            "episode_indices": list(
                range(init_state_start, init_state_start + EPISODES_PER_TASK)
            ),
            "base_seed": seed,
            "episode_seed_formula": "base_seed + task_id * 10000 + init_state_index",
            "image_size": image_size,
            "video": {
                "enabled": True,
                "views": ["agentview", "wrist"],
                "fps": video_fps,
                "stride": video_stride,
            },
            "render_backend": "osmesa",
            "perception": {
                "backend": "grounding-dino",
                "model": str(model),
            },
            "device": device,
            "python_executable": str(python),
            "libero_config_path": str(config_root),
            "formal_environment": formal_campaign_environment(config_root),
            "formal_script_entrypoints": {
                module: formal_script_command(
                    python_executable=str(python),
                    bootstrap_sys_path=dependencies["python"]["python_runtime"][
                        "bootstrap_sys_path"
                    ],
                    module=module,
                )
                for module in (
                    "scripts.run_route130_campaign",
                    "scripts.verify_route_campaign",
                    "scripts.verify_route_run",
                )
            },
            "max_steps": {
                suite: int(DEFAULT_MAX_STEPS[suite])
                for suite, _count in CANONICAL_SUITES
            },
            "execution_schedule": formal_execution_schedule_contract(),
        },
        "task_count": 130,
        "episode_count": 130 * EPISODES_PER_TASK,
        "shard_count": len(shards),
        "shards": shards,
    }


def validate_campaign_plan(payload: Mapping[str, Any]) -> None:
    """Validate the immutable structure and non-overlap of a route plan."""

    if not isinstance(payload, Mapping) or payload.get("schema") != PLAN_SCHEMA:
        raise CampaignPlanError(f"plan schema must be {PLAN_SCHEMA!r}")
    route = payload.get("route")
    if route not in {"b", "c"}:
        raise CampaignPlanError("plan route must be 'b' or 'c'")
    campaign_root = require_canonical_path(
        payload.get("campaign_directory"),
        label="plan campaign_directory",
    )
    working_directory = require_canonical_path(
        payload.get("working_directory"),
        label="plan working_directory",
        must_exist=True,
    )
    if working_directory != canonical_repository_root():
        raise CampaignPlanError(
            "plan working_directory must be the canonical repository root"
        )
    digest = payload.get("source_tree_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise CampaignPlanError("plan source_tree_sha256 is invalid")
    orchestration_digest = payload.get("orchestration_source_sha256")
    if (
        not isinstance(orchestration_digest, str)
        or _SHA256.fullmatch(orchestration_digest) is None
    ):
        raise CampaignPlanError("plan orchestration_source_sha256 is invalid")
    dependencies = payload.get("dependency_provenance")
    if not isinstance(dependencies, Mapping):
        raise CampaignPlanError("plan dependency_provenance must be an object")
    policy_contract = payload.get("policy_contract")
    if not isinstance(policy_contract, Mapping):
        raise CampaignPlanError("plan policy_contract must be an object")
    if policy_contract.get("schema") != POLICY_CONTRACT_SCHEMA:
        raise CampaignPlanError(
            f"plan policy contract schema must be {POLICY_CONTRACT_SCHEMA!r}"
        )
    if not isinstance(policy_contract.get("path"), str) or not policy_contract["path"]:
        raise CampaignPlanError("plan policy_contract.path is invalid")
    policy_digest = policy_contract.get("sha256")
    if not isinstance(policy_digest, str) or _SHA256.fullmatch(policy_digest) is None:
        raise CampaignPlanError("plan policy_contract.sha256 is invalid")
    python_dependency = dependencies.get("python")
    if not isinstance(python_dependency, Mapping):
        raise CampaignPlanError("plan Python provenance must be an object")
    if not isinstance(python_dependency.get("version"), str):
        raise CampaignPlanError("plan Python version is invalid")
    if not isinstance(python_dependency.get("implementation"), str):
        raise CampaignPlanError("plan Python implementation is invalid")
    if not isinstance(python_dependency.get("packages"), Mapping):
        raise CampaignPlanError("plan Python package versions are invalid")
    if set(python_dependency["packages"]) != {
        "hf_libero",
        "libero",
        "mujoco",
        "robosuite",
        "torch",
        "transformers",
        "numpy",
        "cv2",
        "imageio",
        "imageio_ffmpeg",
    }:
        raise CampaignPlanError("plan Python package version set is invalid")
    for required_distribution in (
        "hf_libero",
        "mujoco",
        "robosuite",
        "torch",
        "transformers",
        "numpy",
        "cv2",
        "imageio",
        "imageio_ffmpeg",
    ):
        version = python_dependency["packages"].get(required_distribution)
        if not isinstance(version, str) or not version:
            raise CampaignPlanError(
                f"required Python distribution is missing: {required_distribution}"
            )
    if not isinstance(python_dependency.get("reported_executable"), str):
        raise CampaignPlanError("plan Python reported executable is invalid")
    planned_python_path = require_canonical_path(
        python_dependency.get("executable"),
        label="plan Python executable",
        must_exist=True,
    )
    reported_python_path = require_canonical_path(
        python_dependency.get("reported_executable"),
        label="plan reported Python executable",
        must_exist=True,
    )
    if reported_python_path != planned_python_path:
        raise CampaignPlanError(
            "planned Python must report the same exact executable path"
        )
    inventory = python_dependency.get("distribution_inventory")
    if not isinstance(inventory, Mapping) or inventory.get("schema") != (
        "libero-installed-distribution-inventory.v1"
    ):
        raise CampaignPlanError("plan installed-distribution inventory is invalid")
    inventory_entries = inventory.get("entries")
    if not isinstance(inventory_entries, list) or not inventory_entries:
        raise CampaignPlanError("plan installed-distribution entries are invalid")
    for entry in inventory_entries:
        if (
            not isinstance(entry, Mapping)
            or not isinstance(entry.get("name"), str)
            or not entry["name"]
            or not isinstance(entry.get("version"), str)
            or not entry["version"]
            or (
                entry.get("direct_url") is not None
                and not isinstance(entry.get("direct_url"), Mapping)
            )
        ):
            raise CampaignPlanError(
                "plan installed-distribution entry is invalid"
            )
    if inventory.get("count") != len(inventory_entries):
        raise CampaignPlanError("plan installed-distribution count is invalid")
    inventory_digest = inventory.get("sha256")
    if (
        not isinstance(inventory_digest, str)
        or _SHA256.fullmatch(inventory_digest) is None
        or inventory_digest
        != structured_sha256(
            "libero-installed-distribution-inventory.v1", inventory_entries
        )
    ):
        raise CampaignPlanError("plan installed-distribution hash is invalid")
    runtime_code = python_dependency.get("runtime_package_code")
    if not isinstance(runtime_code, Mapping) or runtime_code.get("schema") != (
        "libero-runtime-package-code.v1"
    ):
        raise CampaignPlanError("plan runtime-package code provenance is invalid")
    runtime_packages = runtime_code.get("packages")
    if (
        not isinstance(runtime_packages, list)
        or [item.get("name") for item in runtime_packages if isinstance(item, Mapping)]
        != ["libero", "robosuite"]
    ):
        raise CampaignPlanError("plan runtime-package set is invalid")
    expected_exclusions = [
        "__pycache__",
        "assets",
        "bddl_files",
        "datasets",
        "init_files",
    ]
    for package in runtime_packages:
        roots = package.get("roots")
        if not isinstance(roots, list) or not roots:
            raise CampaignPlanError("plan runtime-package roots are invalid")
        for root in roots:
            if not isinstance(root, Mapping):
                raise CampaignPlanError("plan runtime-package root is invalid")
            if not isinstance(root.get("path"), str) or not root["path"]:
                raise CampaignPlanError("plan runtime-package root path is invalid")
            if root.get("include") != "**/*.py":
                raise CampaignPlanError("plan runtime-package include rule is invalid")
            if root.get("excluded_directories") != expected_exclusions:
                raise CampaignPlanError("plan runtime-package exclusions are invalid")
            if root.get("symlink_policy") != "resolved-root-no-directory-follow":
                raise CampaignPlanError("plan runtime-package symlink policy is invalid")
            tree_digest = root.get("tree_sha256")
            if not isinstance(tree_digest, str) or _SHA256.fullmatch(tree_digest) is None:
                raise CampaignPlanError("plan runtime-package tree hash is invalid")
            for field in ("file_count", "bytes"):
                value = root.get(field)
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    raise CampaignPlanError(
                        f"plan runtime-package {field} must be positive"
                    )
    runtime_digest = runtime_code.get("sha256")
    if (
        not isinstance(runtime_digest, str)
        or _SHA256.fullmatch(runtime_digest) is None
        or runtime_digest
        != structured_sha256(
            "libero-runtime-package-code.v1", runtime_packages
        )
    ):
        raise CampaignPlanError("plan runtime-package aggregate hash is invalid")
    resources = python_dependency.get("robosuite_runtime_resources")
    if not isinstance(resources, Mapping) or resources.get("schema") != (
        "libero-robosuite-runtime-resources.v1"
    ):
        raise CampaignPlanError("plan robosuite resource provenance is invalid")
    resource_roots = resources.get("roots")
    if not isinstance(resource_roots, list) or not resource_roots:
        raise CampaignPlanError("plan robosuite resource roots are invalid")
    for root in resource_roots:
        if not isinstance(root, Mapping):
            raise CampaignPlanError("plan robosuite resource root is invalid")
        if not isinstance(root.get("path"), str) or not root["path"]:
            raise CampaignPlanError("plan robosuite resource path is invalid")
        if root.get("include") != "all-regular-files-except-python-and-caches":
            raise CampaignPlanError("plan robosuite resource include rule is invalid")
        if root.get("excluded_suffixes") != [".py", ".pyc", ".pyo"]:
            raise CampaignPlanError("plan robosuite resource suffix exclusions are invalid")
        if root.get("excluded_directories") != [
            ".cache",
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "__pycache__",
        ]:
            raise CampaignPlanError(
                "plan robosuite resource directory exclusions are invalid"
            )
        if root.get("excluded_relative_directories") != [
            "models/assets/demonstrations"
        ]:
            raise CampaignPlanError(
                "plan robosuite resource dataset exclusions are invalid"
            )
        if root.get("symlink_policy") != "resolved-root-no-directory-follow":
            raise CampaignPlanError("plan robosuite resource symlink policy is invalid")
        tree_digest = root.get("tree_sha256")
        if not isinstance(tree_digest, str) or _SHA256.fullmatch(tree_digest) is None:
            raise CampaignPlanError("plan robosuite resource tree hash is invalid")
        for field in ("file_count", "bytes"):
            value = root.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise CampaignPlanError(
                    f"plan robosuite resource {field} must be positive"
                )
    resources_digest = resources.get("sha256")
    if (
        not isinstance(resources_digest, str)
        or _SHA256.fullmatch(resources_digest) is None
        or resources_digest
        != structured_sha256(
            "libero-robosuite-runtime-resources.v1", resource_roots
        )
    ):
        raise CampaignPlanError("plan robosuite resource aggregate hash is invalid")
    mujoco_runtime = python_dependency.get("mujoco_runtime_package")
    if not isinstance(mujoco_runtime, Mapping) or mujoco_runtime.get("schema") != (
        "libero-mujoco-runtime-package.v1"
    ):
        raise CampaignPlanError("plan MuJoCo runtime-package provenance is invalid")
    mujoco_roots = mujoco_runtime.get("roots")
    if not isinstance(mujoco_roots, list) or not mujoco_roots:
        raise CampaignPlanError("plan MuJoCo runtime-package roots are invalid")
    expected_cache_exclusions = [
        ".cache",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
    ]
    for root in mujoco_roots:
        if not isinstance(root, Mapping):
            raise CampaignPlanError("plan MuJoCo runtime-package root is invalid")
        if not isinstance(root.get("path"), str) or not root["path"]:
            raise CampaignPlanError("plan MuJoCo runtime-package path is invalid")
        if root.get("include") != "all-regular-runtime-files":
            raise CampaignPlanError("plan MuJoCo runtime-package include rule is invalid")
        if root.get("excluded_suffixes") != [".pyc", ".pyo"]:
            raise CampaignPlanError("plan MuJoCo runtime-package exclusions are invalid")
        if root.get("excluded_directories") != expected_cache_exclusions:
            raise CampaignPlanError(
                "plan MuJoCo runtime-package cache exclusions are invalid"
            )
        if root.get("symlink_policy") != "resolved-root-no-directory-follow":
            raise CampaignPlanError("plan MuJoCo runtime-package symlink policy is invalid")
        tree_digest = root.get("tree_sha256")
        if not isinstance(tree_digest, str) or _SHA256.fullmatch(tree_digest) is None:
            raise CampaignPlanError("plan MuJoCo runtime-package tree hash is invalid")
        for field in ("file_count", "bytes"):
            value = root.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise CampaignPlanError(
                    f"plan MuJoCo runtime-package {field} must be positive"
                )
    mujoco_digest = mujoco_runtime.get("sha256")
    if (
        not isinstance(mujoco_digest, str)
        or _SHA256.fullmatch(mujoco_digest) is None
        or mujoco_digest
        != structured_sha256("libero-mujoco-runtime-package.v1", mujoco_roots)
    ):
        raise CampaignPlanError("plan MuJoCo runtime-package aggregate hash is invalid")
    graphics = python_dependency.get("osmesa_dynamic_libraries")
    if not isinstance(graphics, Mapping) or graphics.get("schema") != (
        "libero-osmesa-dynamic-libraries.v1"
    ):
        raise CampaignPlanError("plan OSMesa dynamic-library provenance is invalid")
    if graphics.get("backend") != "osmesa" or graphics.get("probe") != (
        "import-mujoco-osmesa-and-hash-proc-self-maps-realpaths"
    ):
        raise CampaignPlanError("plan OSMesa dynamic-library probe is invalid")
    if (
        graphics.get("status") != "complete"
        or graphics.get("missing_roles") != []
        or graphics.get("errors") != []
    ):
        raise CampaignPlanError(
            "plan OSMesa dynamic-library probe is incomplete: "
            f"missing={graphics.get('missing_roles')}, errors={graphics.get('errors')}"
        )
    libraries = graphics.get("libraries")
    if not isinstance(libraries, list) or not libraries:
        raise CampaignPlanError("plan OSMesa dynamic-library records are invalid")
    roles: set[str] = set()
    library_paths: set[str] = set()
    for library in libraries:
        if not isinstance(library, Mapping):
            raise CampaignPlanError("plan OSMesa dynamic-library record is invalid")
        role = library.get("role")
        path = library.get("path")
        if role not in {"osmesa", "opengl_api", "cxx_runtime"}:
            raise CampaignPlanError("plan OSMesa dynamic-library role is invalid")
        if (
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or library.get("basename") != Path(path).name
            or path in library_paths
        ):
            raise CampaignPlanError("plan OSMesa dynamic-library path is invalid")
        size = library.get("size")
        digest = library.get("sha256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise CampaignPlanError("plan OSMesa dynamic-library size is invalid")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise CampaignPlanError("plan OSMesa dynamic-library hash is invalid")
        roles.add(role)
        library_paths.add(path)
    if roles != {"osmesa", "opengl_api", "cxx_runtime"}:
        raise CampaignPlanError("plan OSMesa dynamic-library role set is incomplete")
    graphics_digest = graphics.get("sha256")
    if (
        not isinstance(graphics_digest, str)
        or _SHA256.fullmatch(graphics_digest) is None
        or graphics_digest
        != structured_sha256("libero-osmesa-dynamic-libraries.v1", libraries)
    ):
        raise CampaignPlanError("plan OSMesa dynamic-library aggregate hash is invalid")
    executable_digest = python_dependency.get("executable_sha256")
    if not isinstance(executable_digest, str) or _SHA256.fullmatch(executable_digest) is None:
        raise CampaignPlanError("plan Python executable hash is invalid")
    python_runtime = python_dependency.get("python_runtime")
    if (
        not isinstance(python_runtime, Mapping)
        or python_runtime.get("schema") != "libero-python-runtime-closure.v1"
    ):
        raise CampaignPlanError("plan Python runtime closure is invalid")
    if python_runtime.get("isolation_argv") != list(FORMAL_PYTHON_ISOLATION_ARGS):
        raise CampaignPlanError("plan Python isolation argv is invalid")
    bootstrap_sys_path = python_runtime.get("bootstrap_sys_path")
    if not isinstance(bootstrap_sys_path, list) or not bootstrap_sys_path:
        raise CampaignPlanError("plan Python bootstrap sys.path is invalid")
    if len(bootstrap_sys_path) != len(set(bootstrap_sys_path)):
        raise CampaignPlanError("plan Python bootstrap sys.path contains duplicates")
    for path in bootstrap_sys_path:
        require_canonical_path(
            path, label="plan Python bootstrap path", must_exist=True
        )
    stdlib_roots = python_runtime.get("stdlib_roots")
    if not isinstance(stdlib_roots, list) or not stdlib_roots:
        raise CampaignPlanError("plan Python stdlib roots are invalid")
    for root in stdlib_roots:
        if (
            not isinstance(root, Mapping)
            or root.get("excluded_directory_names")
            != ["__pycache__", "dist-packages", "site-packages"]
            or root.get("symlink_policy") != "reject-all-descendant-symlinks"
            or not isinstance(root.get("tree_sha256"), str)
            or _SHA256.fullmatch(root["tree_sha256"]) is None
        ):
            raise CampaignPlanError("plan Python stdlib root record is invalid")
    pyvenv_record = python_runtime.get("pyvenv_cfg")
    if pyvenv_record is not None and (
        not isinstance(pyvenv_record, Mapping)
        or set(pyvenv_record) != {"path", "sha256", "bytes"}
        or not isinstance(pyvenv_record.get("sha256"), str)
        or _SHA256.fullmatch(pyvenv_record["sha256"]) is None
    ):
        raise CampaignPlanError("plan Python pyvenv.cfg record is invalid")
    native_closure = python_runtime.get("native_closure")
    if (
        not isinstance(native_closure, Mapping)
        or native_closure.get("schema")
        != "libero-python-process-native-closure.v1"
        or native_closure.get("probe")
        != "post-import-proc-self-maps-resolved-regular-files"
        or not isinstance(native_closure.get("files"), list)
        or not native_closure["files"]
        or native_closure.get("sha256")
        != structured_sha256(
            "libero-python-process-native-closure.v1",
            native_closure.get("files"),
        )
    ):
        raise CampaignPlanError("plan Python native-loader closure is invalid")
    runtime_bound = {
        "isolation_argv": python_runtime["isolation_argv"],
        "bootstrap_sys_path": bootstrap_sys_path,
        "stdlib_roots": stdlib_roots,
        "pyvenv_cfg": pyvenv_record,
        "native_closure": native_closure,
    }
    if python_runtime.get("sha256") != structured_sha256(
        "libero-python-runtime-closure.v1", runtime_bound
    ):
        raise CampaignPlanError("plan Python runtime aggregate hash is invalid")
    formal_environment = python_dependency.get("formal_runtime_environment")
    if (
        not isinstance(formal_environment, Mapping)
        or formal_environment.get("schema")
        != "libero-formal-runtime-environment.v2"
    ):
        raise CampaignPlanError("plan formal runtime environment is invalid")
    formal_modules = formal_environment.get("modules")
    required_formal_modules = [
        "transformers",
        "torch",
        "numpy",
        "cv2",
        "imageio",
        "imageio_ffmpeg",
        "scipy",
        "yaml",
    ]
    if (
        not isinstance(formal_modules, list)
        or [
            item.get("name") if isinstance(item, Mapping) else None
            for item in formal_modules[: len(required_formal_modules)]
        ] != required_formal_modules
        or len({item.get("name") for item in formal_modules if isinstance(item, Mapping)})
        != len(formal_modules)
    ):
        raise CampaignPlanError("plan formal runtime module set is invalid")
    for module in formal_modules:
        if (
            not isinstance(module.get("owners"), list)
            or not module["owners"]
            or not all(isinstance(owner, str) and owner for owner in module["owners"])
            or (
                module.get("origin") is not None
                and (
                    not isinstance(module.get("origin"), str)
                    or not Path(module["origin"]).is_absolute()
                )
            )
        ):
            raise CampaignPlanError("plan formal runtime module ownership is invalid")
        roots = module.get("roots")
        if not isinstance(roots, list) or not roots:
            raise CampaignPlanError("plan formal runtime module roots are invalid")
        for root in roots:
            if not isinstance(root, Mapping):
                raise CampaignPlanError("plan formal runtime root is invalid")
            if root.get("kind") not in {"file", "tree"}:
                raise CampaignPlanError("plan formal runtime root kind is invalid")
            if (
                not isinstance(root.get("path"), str)
                or not root["path"]
                or not Path(root["path"]).is_absolute()
            ):
                raise CampaignPlanError("plan formal runtime root path is invalid")
            require_canonical_path(
                root["path"],
                label="plan formal runtime root",
                must_exist=True,
            )
            if root.get("include") != "exact-directory-and-all-regular-file-bytes":
                raise CampaignPlanError("plan formal runtime include rule is invalid")
            if root.get("excluded_suffixes") != []:
                raise CampaignPlanError("plan formal runtime exclusions are invalid")
            if root.get("excluded_directories") != []:
                raise CampaignPlanError(
                    "plan formal runtime cache exclusions are invalid"
                )
            if root.get("symlink_policy") != "reject-all-descendant-symlinks":
                raise CampaignPlanError("plan formal runtime symlink policy is invalid")
            if (
                not isinstance(root.get("tree_sha256"), str)
                or _SHA256.fullmatch(root["tree_sha256"]) is None
            ):
                raise CampaignPlanError("plan formal runtime tree hash is invalid")
            for field in ("file_count", "bytes"):
                value = root.get(field)
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    raise CampaignPlanError(
                        f"plan formal runtime {field} must be positive"
                    )
    formal_distributions = formal_environment.get("distributions")
    environment_bootstrap_path = formal_environment.get("bootstrap_sys_path")
    if (
        not isinstance(formal_distributions, list)
        or not formal_distributions
        or environment_bootstrap_path != bootstrap_sys_path
    ):
        raise CampaignPlanError("plan formal distribution closure is invalid")
    distribution_names: set[str] = set()
    for distribution in formal_distributions:
        if not isinstance(distribution, Mapping):
            raise CampaignPlanError("plan formal distribution record is invalid")
        name = distribution.get("name")
        if not isinstance(name, str) or not name or name in distribution_names:
            raise CampaignPlanError("plan formal distribution name is invalid")
        distribution_names.add(name)
        if (
            not isinstance(distribution.get("version"), str)
            or not distribution["version"]
            or not isinstance(distribution.get("record_path"), str)
            or not Path(distribution["record_path"]).is_absolute()
            or not isinstance(distribution.get("record_sha256"), str)
            or _SHA256.fullmatch(distribution["record_sha256"]) is None
            or not isinstance(distribution.get("recorded_files_sha256"), str)
            or _SHA256.fullmatch(distribution["recorded_files_sha256"]) is None
            or not isinstance(distribution.get("recorded_file_count"), int)
            or distribution["recorded_file_count"] < 1
            or not isinstance(distribution.get("recorded_files"), list)
            or distribution["recorded_file_count"]
            != len(distribution["recorded_files"])
            or distribution.get("recorded_files_sha256")
            != structured_sha256(
                "libero-distribution-recorded-files.v1",
                distribution.get("recorded_files"),
            )
            or not isinstance(distribution.get("companion_libraries"), list)
        ):
            raise CampaignPlanError("plan formal distribution fields are invalid")
        for companion in distribution["companion_libraries"]:
            if (
                not isinstance(companion, Mapping)
                or companion.get("include")
                != "exact-directory-and-all-regular-file-bytes"
                or companion.get("symlink_policy")
                != "reject-all-descendant-symlinks"
            ):
                raise CampaignPlanError(
                    "plan formal distribution companion lock is invalid"
                )
    for module in formal_modules:
        if not set(module["owners"]).issubset(distribution_names):
            raise CampaignPlanError("plan formal module owner is not byte-locked")
    cv2_module = next(item for item in formal_modules if item["name"] == "cv2")
    if cv2_module.get("reported_version") != python_dependency["packages"]["cv2"]:
        raise CampaignPlanError("plan cv2 import version is not the recorded version")
    environment_bound = {
        "modules": formal_modules,
        "distributions": formal_distributions,
        "bootstrap_sys_path": environment_bootstrap_path,
    }
    if formal_environment.get("sha256") != structured_sha256(
        "libero-formal-runtime-environment.v2", environment_bound
    ):
        raise CampaignPlanError("plan formal runtime aggregate hash is invalid")
    media = python_dependency.get("media_binaries")
    if (
        not isinstance(media, Mapping)
        or media.get("schema") != "libero-formal-media-binaries.v3"
    ):
        raise CampaignPlanError("plan formal media-binary provenance is invalid")
    executables = media.get("executables")
    if (
        not isinstance(executables, list)
        or [
            item.get("role") if isinstance(item, Mapping) else None
            for item in executables
        ]
        != ["ffprobe", "imageio_ffmpeg", "bwrap"]
    ):
        raise CampaignPlanError("plan formal media-binary role set is invalid")
    for executable in executables:
        if set(executable) != {
            "role",
            "path",
            "sha256",
            "bytes",
            "version_argv",
            "version",
            "native_closure",
        }:
            raise CampaignPlanError("plan formal media-binary fields are invalid")
        if (
            not isinstance(executable.get("path"), str)
            or not Path(executable["path"]).is_absolute()
            or not isinstance(executable.get("version"), str)
            or not executable["version"]
            or executable.get("version_argv")
            != (["--version"] if executable.get("role") == "bwrap" else ["-version"])
            or not isinstance(executable.get("bytes"), int)
            or isinstance(executable.get("bytes"), bool)
            or executable["bytes"] < 1
            or not isinstance(executable.get("sha256"), str)
            or _SHA256.fullmatch(executable["sha256"]) is None
        ):
            raise CampaignPlanError("plan formal media-binary record is invalid")
        require_canonical_path(
            executable["path"],
            label=f"plan {executable['role']} executable",
            must_exist=True,
        )
        if executable["role"] == "bwrap" and executable["path"] != "/usr/bin/bwrap":
            raise CampaignPlanError(
                "plan bwrap executable must be exactly /usr/bin/bwrap"
            )
        closure = executable.get("native_closure")
        files = closure.get("files") if isinstance(closure, Mapping) else None
        if (
            not isinstance(closure, Mapping)
            or closure.get("schema") != "libero-native-loader-closure.v1"
            or closure.get("probe") != "clean-environment-ldd-resolved-files"
            or closure.get("ldd_path") != "/usr/bin/ldd"
            or not isinstance(files, list)
            or closure.get("sha256")
            != structured_sha256(
                "libero-native-loader-closure.v1",
                {
                    "probe": closure.get("probe"),
                    "ldd_path": closure.get("ldd_path"),
                    "ldd_sha256": closure.get("ldd_sha256"),
                    "script_interpreter": closure.get("script_interpreter"),
                    "files": files,
                },
            )
        ):
            raise CampaignPlanError("plan formal media native closure is invalid")
    if media.get("sha256") != structured_sha256(
        "libero-formal-media-binaries.v3", executables
    ):
        raise CampaignPlanError("plan formal media-binary aggregate hash is invalid")
    route_c_process = dependencies.get("route_c_process")
    if (
        not isinstance(route_c_process, Mapping)
        or route_c_process.get("schema")
        != "libero-formal-route-c-plan-binding.v1"
    ):
        raise CampaignPlanError("plan Route C process binding is invalid")
    route_c_bound = {
        key: route_c_process.get(key)
        for key in route_c_process
        if key not in {"schema", "sha256"}
    }
    if route_c_process.get("sha256") != structured_sha256(
        "libero-formal-route-c-plan-binding.v1", route_c_bound
    ):
        raise CampaignPlanError("plan Route C process binding hash is invalid")
    for field in (
        "source_sha256",
        "clock_source_gate_sha256",
        "gallery_sha256",
        "model_sha256",
        "python_sha256",
        "bwrap_sha256",
        "environment_sha256",
    ):
        value = route_c_process.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise CampaignPlanError(f"plan Route C {field} is invalid")
    for field in ("source_file_count", "gallery_file_count", "model_file_count"):
        value = route_c_process.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise CampaignPlanError(f"plan Route C {field} is invalid")
    if (
        route_c_process.get("python_sha256") != executable_digest
        or route_c_process.get("bwrap_path") != "/usr/bin/bwrap"
        or route_c_process.get("policy_runner")
        != "production_route_c_policy_episode"
        or route_c_process.get("bundle_build_count") != 1
        or route_c_process.get("model_load_count") != 1
        or not isinstance(route_c_process.get("system_runtime_files"), list)
        or not isinstance(route_c_process.get("fixed_identity_files"), list)
        or not isinstance(route_c_process.get("environment_values"), Mapping)
    ):
        raise CampaignPlanError("plan Route C process fixed values are invalid")
    config_dependency = dependencies.get("libero_config")
    if not isinstance(config_dependency, Mapping):
        raise CampaignPlanError("plan LIBERO config provenance must be an object")
    config_digest = config_dependency.get("sha256")
    if not isinstance(config_digest, str) or _SHA256.fullmatch(config_digest) is None:
        raise CampaignPlanError("plan LIBERO config hash is invalid")
    model_dependency = dependencies.get("perception_model")
    if not isinstance(model_dependency, Mapping):
        raise CampaignPlanError("plan perception model provenance must be an object")
    asset_dependency = dependencies.get("libero_assets")
    if not isinstance(asset_dependency, Mapping):
        raise CampaignPlanError("plan LIBERO asset provenance must be an object")
    for label, tree_dependency in (
        ("perception model", model_dependency),
        ("LIBERO asset", asset_dependency),
    ):
        tree_digest = tree_dependency.get("tree_sha256")
        if not isinstance(tree_digest, str) or _SHA256.fullmatch(tree_digest) is None:
            raise CampaignPlanError(f"plan {label} tree hash is invalid")
        if tree_dependency.get("symlink_policy") != "follow-target-bytes":
            raise CampaignPlanError(f"plan {label} symlink policy is invalid")
        if not isinstance(tree_dependency.get("path"), str) or not tree_dependency["path"]:
            raise CampaignPlanError(f"plan {label} path is invalid")
        for field in ("file_count", "bytes"):
            value = tree_dependency.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise CampaignPlanError(f"plan {label} {field} must be positive")
    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping):
        raise CampaignPlanError("plan manifest must be an object")
    manifest_digest = manifest.get("sha256")
    if not isinstance(manifest_digest, str) or _SHA256.fullmatch(manifest_digest) is None:
        raise CampaignPlanError("plan manifest.sha256 is invalid")
    if manifest.get("task_count") != 130:
        raise CampaignPlanError("plan manifest.task_count must be 130")
    if not isinstance(manifest.get("path"), str) or not manifest["path"]:
        raise CampaignPlanError("plan manifest.path is invalid")
    task_assets = dependencies.get("installed_task_assets")
    if not isinstance(task_assets, Mapping) or task_assets.get("schema") != (
        "libero-installed-task-assets.v1"
    ):
        raise CampaignPlanError("plan installed BDDL/init provenance is invalid")
    if task_assets.get("manifest_sha256") != manifest_digest:
        raise CampaignPlanError(
            "plan installed BDDL/init provenance is not bound to the manifest"
        )
    for kind in ("bddl", "init_states"):
        asset_group = task_assets.get(kind)
        if not isinstance(asset_group, Mapping):
            raise CampaignPlanError(
                f"plan installed {kind} provenance must be an object"
            )
        root = asset_group.get("root")
        if not isinstance(root, str) or not Path(root).is_absolute():
            raise CampaignPlanError(f"plan installed {kind} root is invalid")
        if asset_group.get("file_count") != 130:
            raise CampaignPlanError(
                f"plan installed {kind} provenance must cover 130 files"
            )
        size = asset_group.get("bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise CampaignPlanError(f"plan installed {kind} byte count is invalid")
        entries_digest = asset_group.get("entries_sha256")
        if (
            not isinstance(entries_digest, str)
            or _SHA256.fullmatch(entries_digest) is None
        ):
            raise CampaignPlanError(
                f"plan installed {kind} entries hash is invalid"
            )
        if asset_group.get("symlink_policy") != (
            "resolved-file-must-remain-under-root"
        ):
            raise CampaignPlanError(
                f"plan installed {kind} symlink policy is invalid"
            )
    task_asset_bound = {
        "manifest_sha256": task_assets["manifest_sha256"],
        "bddl": task_assets["bddl"],
        "init_states": task_assets["init_states"],
    }
    task_asset_digest = task_assets.get("aggregate_sha256")
    if (
        not isinstance(task_asset_digest, str)
        or _SHA256.fullmatch(task_asset_digest) is None
        or task_asset_digest
        != structured_sha256(
            "libero-installed-task-assets.v1", task_asset_bound
        )
    ):
        raise CampaignPlanError(
            "plan installed BDDL/init aggregate hash is invalid"
        )
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise CampaignPlanError("plan protocol must be an object")
    if protocol.get("episodes_per_task") != EPISODES_PER_TASK:
        raise CampaignPlanError("plan episodes_per_task must be 5")
    start = protocol.get("init_state_start")
    indices = protocol.get("episode_indices")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or indices != list(range(start, start + EPISODES_PER_TASK))
    ):
        raise CampaignPlanError("plan episode_indices do not match init_state_start")
    if start != FORMAL_INIT_STATE_START or indices != list(FORMAL_EPISODE_INDICES):
        raise CampaignPlanError(
            "formal plan must use exactly init indices 20,21,22,23,24"
        )
    execution_schedule = protocol.get("execution_schedule")
    required_execution_schedule = formal_execution_schedule_contract()
    if (
        not isinstance(execution_schedule, Mapping)
        or set(execution_schedule) != set(required_execution_schedule)
        or execution_schedule != required_execution_schedule
        or type(execution_schedule.get("per_route_max_workers")) is not int
        or type(execution_schedule.get("global_max_workers")) is not int
        or type(execution_schedule.get("routes_must_not_overlap")) is not bool
    ):
        raise CampaignPlanError(
            "plan execution_schedule must lock two workers per route and "
            "sequential Route B then Route C execution"
        )
    minimum_init_count = manifest.get("minimum_init_state_count")
    if (
        not isinstance(minimum_init_count, int)
        or isinstance(minimum_init_count, bool)
        or minimum_init_count < start + EPISODES_PER_TASK
    ):
        raise CampaignPlanError(
            "plan manifest does not provide every selected init-state index"
        )
    base_seed = protocol.get("base_seed")
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise CampaignPlanError("plan base_seed must be an integer")
    image_size = protocol.get("image_size")
    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size < 64:
        raise CampaignPlanError("plan image_size must be an integer >= 64")
    if protocol.get("render_backend") != "osmesa":
        raise CampaignPlanError("plan render_backend must be osmesa")
    perception = protocol.get("perception")
    if not isinstance(perception, Mapping) or perception.get("backend") != "grounding-dino":
        raise CampaignPlanError("plan perception backend must be grounding-dino")
    if not isinstance(perception.get("model"), str) or not perception["model"]:
        raise CampaignPlanError("plan perception model path is invalid")
    video = protocol.get("video")
    if not isinstance(video, Mapping) or video.get("enabled") is not True:
        raise CampaignPlanError("plan must enable dual-view video")
    if video.get("views") != ["agentview", "wrist"]:
        raise CampaignPlanError("plan video views must be agentview+wrist")
    for field in ("fps", "stride"):
        value = video.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise CampaignPlanError(f"plan video {field} must be a positive integer")
    for field in ("device", "python_executable", "libero_config_path"):
        if not isinstance(protocol.get(field), str) or not protocol[field]:
            raise CampaignPlanError(f"plan protocol {field} is invalid")
    if protocol.get("formal_environment") != formal_campaign_environment(
        protocol["libero_config_path"]
    ):
        raise CampaignPlanError("plan formal child environment is not the exact allowlist")
    expected_script_entrypoints = {
        module: formal_script_command(
            python_executable=protocol["python_executable"],
            bootstrap_sys_path=python_runtime["bootstrap_sys_path"],
            module=module,
        )
        for module in (
            "scripts.run_route130_campaign",
            "scripts.verify_route_campaign",
            "scripts.verify_route_run",
        )
    }
    if protocol.get("formal_script_entrypoints") != expected_script_entrypoints:
        raise CampaignPlanError("plan formal script entrypoints are invalid")
    if python_dependency.get("executable") != protocol["python_executable"]:
        raise CampaignPlanError("plan Python executable path disagrees with protocol")
    if config_dependency.get("path") != str(
        Path(protocol["libero_config_path"]) / "config.yaml"
    ):
        raise CampaignPlanError("plan LIBERO config path disagrees with protocol")
    if model_dependency.get("path") != perception["model"]:
        raise CampaignPlanError("plan perception model path disagrees with protocol")
    expected_max_steps = {suite for suite, _count in CANONICAL_SUITES}
    max_steps = protocol.get("max_steps")
    if not isinstance(max_steps, Mapping) or set(max_steps) != expected_max_steps:
        raise CampaignPlanError("plan max_steps must cover exactly the five suites")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in max_steps.values()
    ):
        raise CampaignPlanError("plan max_steps values must be positive integers")

    shards = payload.get("shards")
    if not isinstance(shards, list) or len(shards) != 13:
        raise CampaignPlanError("plan must contain exactly 13 shards")
    actual_specs: list[tuple[str, tuple[int, ...]]] = []
    run_directories: list[str] = []
    all_task_keys: list[str] = []
    for index, shard in enumerate(shards):
        if not isinstance(shard, Mapping) or shard.get("index") != index:
            raise CampaignPlanError(f"plan shard {index} is invalid")
        suite = shard.get("suite")
        raw_ids = shard.get("task_ids")
        if not isinstance(suite, str) or not isinstance(raw_ids, list) or not all(
            isinstance(task_id, int) and not isinstance(task_id, bool)
            for task_id in raw_ids
        ):
            raise CampaignPlanError(f"plan shard {index} has invalid suite/task_ids")
        ids = tuple(raw_ids)
        actual_specs.append((suite, ids))
        expected_keys = [task_key(suite, task_id) for task_id in ids]
        if shard.get("task_keys") != expected_keys:
            raise CampaignPlanError(f"plan shard {index} task_keys disagree with task_ids")
        if shard.get("episodes") != len(ids) * EPISODES_PER_TASK:
            raise CampaignPlanError(f"plan shard {index} episode count is invalid")
        run_directory = shard.get("run_directory")
        if not isinstance(run_directory, str) or not run_directory:
            raise CampaignPlanError(f"plan shard {index} run_directory is invalid")
        run_name = shard.get("run_name")
        if not isinstance(run_name, str) or not run_name:
            raise CampaignPlanError(f"plan shard {index} run_name is invalid")
        run_directories.append(run_directory)
        all_task_keys.extend(expected_keys)
        command = shard.get("base_command")
        if not isinstance(command, list) or not all(
            isinstance(argument, str) and argument for argument in command
        ):
            raise CampaignPlanError(f"plan shard {index} base_command is invalid")
        if "--resume" in command or "--max-steps" in command:
            raise CampaignPlanError(
                f"plan shard {index} must forbid resume and use config-default max_steps"
            )
        bootstrap_paths_json = json.dumps(
            python_dependency["python_runtime"]["bootstrap_sys_path"],
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        expected_entrypoint = [
            protocol["python_executable"],
            *FORMAL_PYTHON_ISOLATION_ARGS,
            FORMAL_PYTHON_BOOTSTRAP,
            str(canonical_repository_root()),
            bootstrap_paths_json,
        ]
        if command[: len(expected_entrypoint)] != expected_entrypoint:
            raise CampaignPlanError(f"plan shard {index} has an invalid Python entrypoint")

        def option(flag: str) -> str:
            if command.count(flag) != 1:
                raise CampaignPlanError(
                    f"plan shard {index} must contain {flag} exactly once"
                )
            position = command.index(flag)
            if position + 1 >= len(command) or command[position + 1].startswith("--"):
                raise CampaignPlanError(f"plan shard {index} has no value for {flag}")
            return command[position + 1]

        expected_options = {
            "--route": route,
            "--suite": suite,
            "--task-ids": _range_argument(ids),
            "--episodes-per-task": str(EPISODES_PER_TASK),
            "--init-state-start": str(protocol["init_state_start"]),
            "--run-name": run_name,
            "--device": protocol["device"],
            "--perception-backend": "grounding-dino",
            "--perception-model": protocol["perception"]["model"],
            "--libero-config-path": protocol["libero_config_path"],
            "--image-size": str(protocol["image_size"]),
            "--seed": str(protocol["base_seed"]),
            "--video-fps": str(protocol["video"]["fps"]),
            "--video-stride": str(protocol["video"]["stride"]),
            "--render-backend": "osmesa",
        }
        for flag, expected_value in expected_options.items():
            if option(flag) != expected_value:
                raise CampaignPlanError(
                    f"plan shard {index} {flag} disagrees with frozen protocol"
                )
        expected_command = [
            protocol["python_executable"],
            *FORMAL_PYTHON_ISOLATION_ARGS,
            FORMAL_PYTHON_BOOTSTRAP,
            str(canonical_repository_root()),
            bootstrap_paths_json,
            "--route",
            route,
            "--suite",
            suite,
            "--task-ids",
            _range_argument(ids),
            "--episodes-per-task",
            str(protocol["episodes_per_task"]),
            "--init-state-start",
            str(protocol["init_state_start"]),
            "--output-dir",
            str(Path(run_directory).parent),
            "--run-name",
            run_name,
            "--device",
            protocol["device"],
            "--perception-backend",
            "grounding-dino",
            "--perception-model",
            protocol["perception"]["model"],
            "--libero-config-path",
            protocol["libero_config_path"],
            "--image-size",
            str(protocol["image_size"]),
            "--seed",
            str(protocol["base_seed"]),
            "--video",
            "--video-fps",
            str(protocol["video"]["fps"]),
            "--video-stride",
            str(protocol["video"]["stride"]),
            "--render-backend",
            "osmesa",
        ]
        if command != expected_command:
            raise CampaignPlanError(
                f"plan shard {index} base_command must exactly match the frozen protocol"
            )
        if command.count("--video") != 1 or "--no-video" in command:
            raise CampaignPlanError(f"plan shard {index} must enable video exactly once")
        expected_run_directory = campaign_root / "runs" / str(run_name)
        actual_run_directory = require_canonical_path(
            run_directory,
            label=f"plan shard {index} run_directory",
        )
        if actual_run_directory != expected_run_directory:
            raise CampaignPlanError(
                f"plan shard {index} run_directory is not its exact canonical location"
            )
        output_directory = require_canonical_path(
            option("--output-dir"),
            label=f"plan shard {index} --output-dir",
        )
        if output_directory / option("--run-name") != actual_run_directory:
            raise CampaignPlanError(
                f"plan shard {index} command writes a different run directory"
            )
    if tuple(actual_specs) != canonical_shards():
        raise CampaignPlanError("plan shards do not match the canonical 13-way split")
    if len(run_directories) != len(set(run_directories)):
        raise CampaignPlanError("two plan shards share one run directory")
    if len(all_task_keys) != 130 or len(set(all_task_keys)) != 130:
        raise CampaignPlanError("plan shards do not cover 130 unique tasks")
    if payload.get("task_count") != 130 or payload.get("episode_count") != 650:
        raise CampaignPlanError("plan task/episode totals must be 130/650")
    if payload.get("shard_count") != 13:
        raise CampaignPlanError("plan shard_count must be 13")


def load_campaign_plan(path: str | Path) -> dict[str, Any]:
    raw_path = Path(path).expanduser()
    plan_path = require_canonical_path(
        str(raw_path), label="campaign plan path", must_exist=True
    )
    if plan_path.name != "campaign_plan.json":
        raise CampaignPlanError(
            "formal campaign plan must retain the exact campaign_plan.json name"
        )
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignPlanError(f"cannot read campaign plan {plan_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CampaignPlanError("campaign plan top-level value must be an object")
    validate_campaign_plan(payload)
    return payload


def ensure_campaign_plan(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Create a plan once, or reject any requested/source/manifest drift."""

    validate_campaign_plan(payload)
    destination = require_canonical_path(
        str(Path(path).expanduser()), label="campaign plan destination"
    )
    expected = Path(str(payload["campaign_directory"])) / "campaign_plan.json"
    if destination != expected:
        raise CampaignPlanError(
            f"campaign plan destination must be exactly {expected}"
        )
    if destination.exists():
        previous = load_campaign_plan(destination)
        if previous != payload:
            changed = [
                field
                for field in sorted(set(previous) | set(payload))
                if previous.get(field) != payload.get(field)
            ]
            raise CampaignPlanError(
                "existing campaign plan differs from the requested frozen plan; "
                f"changed_fields={changed}"
            )
        return destination
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
