"""Bubblewrap projection and runtime evidence for formal Route-C policy code."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Iterable


BWRAP_PATH = Path("/usr/bin/bwrap")
POLICY_SOURCE_ROOT = Path("/policy")
SANDBOX_ASSET_ROOT = Path("/home/kzoacn/.cache/libero/assets")
SANDBOX_RUNTIME_ROOT = Path("/runtime")
SANDBOX_HOSTNAME = "libero-route-c-policy"

POLICY_SOURCE_FILES = (
    "libero_system/__init__.py",
    "libero_system/common/__init__.py",
    "libero_system/common/camera_geometry.py",
    "libero_system/common/grasp_journal.py",
    "libero_system/common/observation.py",
    "libero_system/common/pan_affordance.py",
    "libero_system/common/policy.py",
    "libero_system/goal_skills/__init__.py",
    "libero_system/goal_skills/compiler.py",
    "libero_system/goal_skills/controller.py",
    "libero_system/goal_skills/detectors.py",
    "libero_system/goal_skills/schema.py",
    "libero_system/integration/__init__.py",
    "libero_system/integration/adapters.py",
    "libero_system/integration/cavity.py",
    "libero_system/integration/components.py",
    "libero_system/integration/formal_process_sandbox.py",
    "libero_system/integration/formal_process_transport.py",
    "libero_system/integration/formal_route_c_process.py",
    "libero_system/integration/policy_ipc.py",
    "libero_system/integration/policy_result.py",
    "libero_system/integration/route_c_policy_runtime.py",
    "libero_system/perception/__init__.py",
    "libero_system/perception/adapters.py",
    "libero_system/perception/fusion.py",
    "libero_system/perception/gallery.py",
    "libero_system/perception/geometry.py",
    "libero_system/perception/grounding_dino.py",
    "libero_system/perception/pipeline.py",
    "libero_system/perception/schema.py",
    "libero_system/route_b/__init__.py",
    "libero_system/route_b/controller.py",
    "libero_system/route_b/models.py",
    "libero_system/route_b/perception.py",
    "libero_system/route_b/task_compiler.py",
    "libero_system/route_c/__init__.py",
    "libero_system/route_c/compiler.py",
    "libero_system/route_c/controller.py",
    "libero_system/route_c/grasp.py",
    "libero_system/route_c/optimizer.py",
    "libero_system/route_c/perception.py",
    "libero_system/route_c/schema.py",
    "libero_system/route_c/sdf.py",
    "libero_system/route_c/sequential.py",
    "libero_system/route_c/task_planner.py",
)

EXCLUDED_POLICY_FILES = (
    "libero_system/common/env_adapter.py",
    "libero_system/common/runner.py",
    "libero_system/integration/campaign_manifest.py",
    "libero_system/integration/campaign_plan.py",
    "libero_system/integration/cli.py",
    "libero_system/integration/config.py",
    "libero_system/integration/evaluator.py",
    "libero_system/integration/formal_provenance.py",
    "libero_system/integration/results.py",
    "libero_system/integration/video.py",
    "libero_system/integration/video_audit.py",
)

MASKED_SITE_PACKAGES = (
    "bddl",
    "bddl-1.0.1.dist-info",
    "libero",
    "mujoco",
    "mujoco-3.3.2.dist-info",
    "robosuite",
    "robosuite-1.4.0.dist-info",
)

SYSTEM_RUNTIME_FILES = (
    "/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
    "/lib/x86_64-linux-gnu/libGL.so.1",
    "/lib/x86_64-linux-gnu/libGLX.so.0",
    "/lib/x86_64-linux-gnu/libGLdispatch.so.0",
    "/lib/x86_64-linux-gnu/libX11.so.6",
    "/lib/x86_64-linux-gnu/libXau.so.6",
    "/lib/x86_64-linux-gnu/libXdmcp.so.6",
    "/lib/x86_64-linux-gnu/libbsd.so.0",
    "/lib/x86_64-linux-gnu/libc.so.6",
    "/lib/x86_64-linux-gnu/libdl.so.2",
    "/lib/x86_64-linux-gnu/libglib-2.0.so.0",
    "/lib/x86_64-linux-gnu/libgthread-2.0.so.0",
    "/lib/x86_64-linux-gnu/libm.so.6",
    "/lib/x86_64-linux-gnu/libmd.so.0",
    "/lib/x86_64-linux-gnu/libpcre2-8.so.0",
    "/lib/x86_64-linux-gnu/libpthread.so.0",
    "/lib/x86_64-linux-gnu/librt.so.1",
    "/lib/x86_64-linux-gnu/libutil.so.1",
    "/lib/x86_64-linux-gnu/libxcb.so.1",
    "/lib64/ld-linux-x86-64.so.2",
)


class FormalProcessSandboxError(RuntimeError):
    """Raised when the fixed policy projection cannot be constructed."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    """Fields that must remain fixed while one regular file is consumed."""

    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_verified_regular(path: Path, expected: os.stat_result) -> tuple[int, int]:
    """Open exactly the lstat'ed single-link regular inode without following."""

    if not stat.S_ISREG(expected.st_mode) or expected.st_nlink != 1:
        raise FormalProcessSandboxError(
            f"projected model file must be a single-link regular file: {path}"
        )
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FormalProcessSandboxError(
            f"cannot safely open projected regular file: {path}: {exc}"
        ) from exc
    opened = os.fstat(descriptor)
    if _stat_identity(opened) != _stat_identity(expected):
        os.close(descriptor)
        raise FormalProcessSandboxError(
            f"projected file changed between lstat and open: {path}"
        )
    return descriptor, opened.st_size


def _hash_verified_regular(path: Path, expected: os.stat_result) -> tuple[str, int]:
    descriptor, size = _open_verified_regular(path, expected)
    digest = hashlib.sha256()
    try:
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        final = os.fstat(descriptor)
        if _stat_identity(final) != _stat_identity(expected):
            raise FormalProcessSandboxError(
                f"projected file changed while it was hashed: {path}"
            )
    finally:
        os.close(descriptor)
    return digest.hexdigest(), size


def _internal_regular_symlink(
    path: Path,
    *,
    base: Path,
    link_stat: os.stat_result,
) -> tuple[str, Path]:
    """Resolve a stable file symlink and prove its final inode stays in ``base``."""

    try:
        raw_target = os.readlink(path)
        resolved = path.resolve(strict=True)
        resolved.relative_to(base)
    except (OSError, RuntimeError, ValueError) as exc:
        raise FormalProcessSandboxError(
            f"projected symlink is broken or escapes its model repository: {path}"
        ) from exc
    target_stat = resolved.lstat()
    if not stat.S_ISREG(target_stat.st_mode) or target_stat.st_nlink != 1:
        raise FormalProcessSandboxError(
            f"projected symlink must resolve to an internal single-link regular file: {path}"
        )
    if (
        _stat_identity(path.lstat()) != _stat_identity(link_stat)
        or os.readlink(path) != raw_target
    ):
        raise FormalProcessSandboxError(
            f"projected symlink changed while it was resolved: {path}"
        )
    return raw_target, resolved


def tree_manifest_sha256(root: str | Path) -> tuple[str, int]:
    """Hash names, safe internal file symlinks, and exact regular bytes."""

    base_path = Path(root)
    try:
        root_stat = base_path.lstat()
    except OSError as exc:
        raise FormalProcessSandboxError(
            f"manifest root is unavailable: {base_path}"
        ) from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise FormalProcessSandboxError(f"manifest root is not a directory: {base_path}")
    base = base_path.resolve(strict=True)
    digest = hashlib.sha256(b"libero-formal-tree-manifest.v1\0")
    count = 0
    for path in sorted(base.rglob("*"), key=lambda item: item.relative_to(base).as_posix()):
        entry_stat = path.lstat()
        relative = path.relative_to(base).as_posix().encode("utf-8")
        if stat.S_ISLNK(entry_stat.st_mode):
            raw_target, resolved = _internal_regular_symlink(
                path,
                base=base,
                link_stat=entry_stat,
            )
            target = raw_target.encode("utf-8")
            target_sha, _ = _hash_verified_regular(resolved, resolved.lstat())
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            digest.update(b"L")
            digest.update(len(target).to_bytes(4, "big"))
            digest.update(target)
            digest.update(bytes.fromhex(target_sha))
        elif stat.S_ISDIR(entry_stat.st_mode):
            continue
        elif stat.S_ISREG(entry_stat.st_mode):
            file_sha, size = _hash_verified_regular(path, entry_stat)
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            digest.update(b"F")
            digest.update(size.to_bytes(8, "big"))
            digest.update(bytes.fromhex(file_sha))
        else:
            raise FormalProcessSandboxError(
                f"unsupported projected filesystem entry: {path}"
            )
        count += 1
    if count < 1:
        raise FormalProcessSandboxError(f"manifest tree is empty: {base}")
    return digest.hexdigest(), count


def _copy_verified_regular(
    source: Path,
    destination: Path,
    source_stat: os.stat_result,
) -> None:
    descriptor, expected_size = _open_verified_regular(source, source_stat)
    destination.parent.mkdir(parents=True, exist_ok=True)
    copied = 0
    try:
        with destination.open("xb") as output:
            while True:
                block = os.read(descriptor, 4 * 1024 * 1024)
                if not block:
                    break
                output.write(block)
                copied += len(block)
        final = os.fstat(descriptor)
        if (
            copied != expected_size
            or _stat_identity(final) != _stat_identity(source_stat)
        ):
            raise FormalProcessSandboxError(
                f"model source changed while it was materialized: {source}"
            )
    finally:
        os.close(descriptor)
    destination.chmod(0o444)


def _materialize_model_repository(source: Path, destination: Path) -> None:
    """Freeze a validated model tree beneath the private projection root."""

    source = source.resolve(strict=True)
    source_stat = source.lstat()
    if not stat.S_ISDIR(source_stat.st_mode):
        raise FormalProcessSandboxError(
            f"model repository must be an exact directory: {source}"
        )
    destination.mkdir(parents=True, exist_ok=False)
    for entry in sorted(
        source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()
    ):
        relative = entry.relative_to(source)
        frozen = destination / relative
        entry_stat = entry.lstat()
        if stat.S_ISLNK(entry_stat.st_mode):
            _, resolved = _internal_regular_symlink(
                entry,
                base=source,
                link_stat=entry_stat,
            )
            target_relative = resolved.relative_to(source)
            frozen.parent.mkdir(parents=True, exist_ok=True)
            frozen.symlink_to(
                os.path.relpath(destination / target_relative, frozen.parent)
            )
            if (
                _stat_identity(entry.lstat()) != _stat_identity(entry_stat)
                or entry.resolve(strict=True) != resolved
            ):
                raise FormalProcessSandboxError(
                    f"model symlink changed while it was materialized: {entry}"
                )
        elif stat.S_ISDIR(entry_stat.st_mode):
            frozen.mkdir(parents=True, exist_ok=True)
        elif stat.S_ISREG(entry_stat.st_mode):
            _copy_verified_regular(entry, frozen, entry_stat)
        else:
            raise FormalProcessSandboxError(
                f"unsupported model repository entry: {entry}"
            )
    _make_world_readable(destination)
    # Re-read the frozen tree before it is ever exposed to the worker.  This
    # also proves every reconstructed link resolves inside the frozen copy.
    tree_manifest_sha256(destination)


def fixed_child_environment(*, accelerator: bool = False) -> dict[str, str]:
    """Return the complete environment; no parent value is inherited."""

    return {
        "HF_HUB_OFFLINE": "1",
        "HUGGINGFACE_HUB_CACHE": "/tmp/huggingface/hub",
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LD_LIBRARY_PATH": "/runtime/lib:/usr/lib/wsl/lib" if accelerator else "/runtime/lib",
        "MPLCONFIGDIR": "/tmp/matplotlib",
        "NUMBA_CACHE_DIR": "/tmp/numba",
        "NUMBA_DISABLE_JIT": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PATH": "/runtime/bin",
        "PWD": "/tmp",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_CACHE": "/tmp/huggingface/transformers",
        "TRANSFORMERS_OFFLINE": "1",
        "TZ": "UTC",
    }


def environment_sha256(environment: dict[str, str]) -> str:
    encoded = json.dumps(
        environment,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(b"libero-fixed-child-environment.v1\0" + encoded).hexdigest()


def validate_policy_runtime_clock_source(path: str | Path) -> str:
    """Reject evaluator-clock APIs in the isolated policy entry module."""

    source_path = Path(path)
    payload = source_path.read_bytes()
    tree = ast.parse(payload, filename=str(source_path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in {"time", "datetime"}:
                    violations.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".", 1)[0] in {"time", "datetime"}:
                violations.append(node.module)
        elif isinstance(node, ast.Name) and node.id in {
            "time",
            "datetime",
            "monotonic",
            "perf_counter",
        }:
            violations.append(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in {
            "monotonic",
            "monotonic_ns",
            "perf_counter",
            "perf_counter_ns",
            "time",
            "time_ns",
        }:
            violations.append(node.attr)
    if violations:
        raise FormalProcessSandboxError(
            "isolated policy runtime uses a forbidden clock symbol: "
            + ", ".join(sorted(set(violations)))
        )
    return hashlib.sha256(b"libero-policy-clock-source-gate.v1\0" + payload).hexdigest()


def _copy_readonly(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise FormalProcessSandboxError(f"projected source must be a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o444)


def _make_world_readable(root: Path) -> None:
    for directory in (root, *[path for path in root.rglob("*") if path.is_dir()]):
        directory.chmod(0o555)


def _gallery_files(asset_root: Path) -> tuple[Path, ...]:
    from libero_system.perception.gallery import PUBLIC_ASSET_GALLERY_SPECS

    result: set[Path] = set()
    for spec in PUBLIC_ASSET_GALLERY_SPECS.values():
        directory = asset_root / spec.collection / spec.folder
        result.add(directory / spec.xml_name)
        result.update(directory / name for name in spec.texture_names)
    missing = sorted(str(path) for path in result if not path.is_file())
    if missing:
        raise FormalProcessSandboxError(
            "gallery projection is incomplete: " + ", ".join(missing[:4])
        )
    return tuple(sorted(result))


def _model_repository(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    for candidate in (resolved, *resolved.parents):
        if (candidate / "blobs").is_dir() and (candidate / "snapshots").is_dir():
            return candidate
    return resolved


def _destination_parent_args(destinations: Iterable[Path]) -> list[str]:
    directories: set[Path] = set()
    for destination in destinations:
        current = destination.parent
        while current != Path("/"):
            directories.add(current)
            current = current.parent
    arguments: list[str] = []
    for directory in sorted(directories, key=lambda item: (len(item.parts), str(item))):
        arguments.extend(("--dir", str(directory)))
    return arguments


def _remove_projection_root(root: Path) -> None:
    """Remove one exact private projection, including read-only model bytes."""

    if not root.exists():
        return
    root.chmod(0o700)
    for path in root.rglob("*"):
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            continue
        path.chmod(0o700 if stat.S_ISDIR(mode) else 0o600)
    shutil.rmtree(root)


@dataclass(slots=True)
class SandboxProjection:
    temporary_root: Path
    policy_root: Path
    gallery_root: Path
    empty_mask_root: Path
    identity_root: Path
    runtime_source: Path
    python_source: Path
    python_destination: Path
    model_source: Path | None
    model_destination: Path | None
    source_sha256: str
    source_file_count: int
    gallery_sha256: str
    gallery_file_count: int
    model_sha256: str | None
    model_file_count: int
    python_sha256: str
    clock_source_gate_sha256: str
    bwrap_sha256: str
    bwrap_version: str
    system_runtime_files: tuple[tuple[str, str], ...]
    accelerator: bool = False

    @classmethod
    def create(
        cls,
        *,
        python_executable: Path,
        perception_backend: str,
        perception_model: Path | None,
        accelerator: bool = False,
    ) -> "SandboxProjection":
        repository = Path(__file__).resolve().parents[2]
        if accelerator and not (Path("/dev/dxg").exists() and Path("/usr/lib/wsl").is_dir()):
            raise FormalProcessSandboxError("CUDA policy requires the WSL GPU device and driver runtime")
        runtime_source = Path(sys.prefix).resolve(strict=True)
        try:
            python_relative = python_executable.relative_to(runtime_source)
        except ValueError as exc:
            raise FormalProcessSandboxError(
                "formal Python executable must belong to the active conda runtime"
            ) from exc
        root = Path(tempfile.mkdtemp(prefix="libero-route-c-sandbox-", dir="/tmp"))
        policy_root = root / "policy"
        gallery_root = root / "gallery"
        empty_mask = root / "empty-mask"
        identity_root = root / "identity"
        try:
            for relative_name in POLICY_SOURCE_FILES:
                _copy_readonly(repository / relative_name, policy_root / relative_name)
            clock_gate = validate_policy_runtime_clock_source(
                policy_root / "libero_system/integration/route_c_policy_runtime.py"
            )
            asset_source = Path("/home/kzoacn/.cache/libero/assets")
            for source in _gallery_files(asset_source):
                _copy_readonly(
                    source,
                    gallery_root / source.relative_to(asset_source),
                )
            empty_mask.mkdir(parents=True, exist_ok=True)
            empty_mask.chmod(0o555)
            identity_root.mkdir(parents=True, exist_ok=True)
            (identity_root / "passwd").write_text(
                "nobody:x:65534:65534:Route C policy:/tmp:/usr/sbin/nologin\n",
                encoding="utf-8",
            )
            (identity_root / "group").write_text(
                "nogroup:x:65534:\n",
                encoding="utf-8",
            )
            (identity_root / "passwd").chmod(0o444)
            (identity_root / "group").chmod(0o444)
            identity_root.chmod(0o555)
            _make_world_readable(policy_root)
            _make_world_readable(gallery_root)
            source_sha, source_count = tree_manifest_sha256(policy_root)
            gallery_sha, gallery_count = tree_manifest_sha256(gallery_root)

            model_path = perception_model
            if perception_backend in {"auto", "grounding-dino"} and model_path is None:
                from libero_system.perception.grounding_dino import (
                    DEFAULT_GROUNDING_DINO_TINY,
                )

                model_path = DEFAULT_GROUNDING_DINO_TINY
            model_source = None
            model_destination = None
            model_sha = None
            model_count = 0
            if model_path is not None:
                model_origin = _model_repository(Path(model_path))
                if not model_origin.is_dir():
                    raise FormalProcessSandboxError(
                        "formal perception model must resolve to a model repository"
                    )
                model_source = root / "model-repository"
                model_destination = model_origin
                _materialize_model_repository(model_origin, model_source)
                model_sha, model_count = tree_manifest_sha256(model_source)

            root.chmod(0o555)

            if not BWRAP_PATH.is_file():
                raise FormalProcessSandboxError("/usr/bin/bwrap is required")
            version = subprocess.run(
                (str(BWRAP_PATH), "--version"),
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
            ).stdout.strip()
            if not version.startswith("bubblewrap "):
                raise FormalProcessSandboxError("bubblewrap version output is invalid")
            system_runtime_files = tuple(
                (destination, sha256_file(Path(destination).resolve(strict=True)))
                for destination in SYSTEM_RUNTIME_FILES
            )
            return cls(
                temporary_root=root,
                policy_root=policy_root,
                gallery_root=gallery_root,
                empty_mask_root=empty_mask,
                identity_root=identity_root,
                runtime_source=runtime_source,
                python_source=python_executable,
                python_destination=SANDBOX_RUNTIME_ROOT / python_relative,
                model_source=model_source,
                model_destination=model_destination,
                source_sha256=source_sha,
                source_file_count=source_count,
                gallery_sha256=gallery_sha,
                gallery_file_count=gallery_count,
                model_sha256=model_sha,
                model_file_count=model_count,
                python_sha256=sha256_file(python_executable),
                clock_source_gate_sha256=clock_gate,
                bwrap_sha256=sha256_file(BWRAP_PATH),
                bwrap_version=version,
                system_runtime_files=system_runtime_files,
                accelerator=accelerator,
            )
        except BaseException:
            _remove_projection_root(root)
            raise

    def mount_commitments(self) -> dict[str, object]:
        result = {
            "policy_source": {
                "destination": str(POLICY_SOURCE_ROOT),
                "sha256": self.source_sha256,
                "file_count": self.source_file_count,
                "read_only": True,
            },
            "gallery_assets": {
                "destination": str(SANDBOX_ASSET_ROOT),
                "sha256": self.gallery_sha256,
                "file_count": self.gallery_file_count,
                "read_only": True,
            },
            "model_repository": (
                None
                if self.model_source is None
                else {
                    "destination": str(self.model_destination),
                    "sha256": self.model_sha256,
                    "file_count": self.model_file_count,
                    "read_only": True,
                }
            ),
            "conda_runtime": {
                "destination": str(SANDBOX_RUNTIME_ROOT),
                "python_sha256": self.python_sha256,
                "read_only": True,
            },
            "private_tmp": {"destination": "/tmp", "kind": "tmpfs"},
            "private_proc": {"destination": "/proc", "kind": "proc"},
            "device_nodes": [
                {
                    "destination": "/dev/null",
                    "kind": "dev-bind",
                    "access": "read_write",
                },
                {
                    "destination": "/dev/urandom",
                    "kind": "dev-bind",
                    "access": "read_write",
                },
            ],
            "system_runtime_files": [
                {
                    "destination": destination,
                    "sha256": sha256,
                    "read_only": True,
                }
                for destination, sha256 in self.system_runtime_files
            ],
            "fixed_identity_files": [
                {
                    "destination": f"/etc/{name}",
                    "sha256": sha256_file(self.identity_root / name),
                    "read_only": True,
                }
                for name in ("passwd", "group")
            ],
        }
        if self.accelerator:
            result["accelerator_runtime"] = {
                "destination": "/usr/lib/wsl", "read_only": True,
                "device": "/dev/dxg",
            }
        return result

    def command(
        self,
        *,
        child_fd: int,
        bootstrap: str,
        test_policy: bool,
    ) -> tuple[str, ...]:
        environment = fixed_child_environment(accelerator=self.accelerator)
        destinations = [
            POLICY_SOURCE_ROOT,
            SANDBOX_ASSET_ROOT,
            SANDBOX_RUNTIME_ROOT,
            Path("/dev/null"),
            Path("/dev/urandom"),
            Path("/etc/passwd"),
            Path("/etc/group"),
            *[Path(path) for path in SYSTEM_RUNTIME_FILES],
        ]
        if self.model_destination is not None:
            destinations.append(self.model_destination)
        command: list[str] = [
            str(BWRAP_PATH),
            "--unshare-user",
            "--unshare-pid",
            "--unshare-net",
            "--unshare-ipc",
            "--unshare-uts",
            "--disable-userns",
            "--assert-userns-disabled",
            "--as-pid-1",
            "--new-session",
            "--die-with-parent",
            "--cap-drop",
            "ALL",
            "--uid",
            "65534",
            "--gid",
            "65534",
            "--hostname",
            SANDBOX_HOSTNAME,
            "--clearenv",
        ]
        for name, value in sorted(environment.items()):
            command.extend(("--setenv", name, value))
        command.extend(_destination_parent_args(destinations))
        command.extend(("--ro-bind", str(self.runtime_source), str(SANDBOX_RUNTIME_ROOT)))
        site_packages = SANDBOX_RUNTIME_ROOT / "lib/python3.12/site-packages"
        for package in MASKED_SITE_PACKAGES:
            source = self.runtime_source / "lib/python3.12/site-packages" / package
            if source.exists():
                command.extend(
                    ("--ro-bind", str(self.empty_mask_root), str(site_packages / package))
                )
        command.extend(("--ro-bind", str(self.policy_root), str(POLICY_SOURCE_ROOT)))
        command.extend(
            ("--ro-bind", str(self.gallery_root), str(SANDBOX_ASSET_ROOT))
        )
        if self.model_source is not None and self.model_destination is not None:
            command.extend(
                ("--ro-bind", str(self.model_source), str(self.model_destination))
            )
        for name in ("passwd", "group"):
            command.extend(
                (
                    "--ro-bind",
                    str(self.identity_root / name),
                    f"/etc/{name}",
                )
            )
        for source_name in SYSTEM_RUNTIME_FILES:
            source = Path(source_name).resolve(strict=True)
            command.extend(("--ro-bind", str(source), source_name))
        if self.accelerator:
            command.extend((
                "--dir", "/usr", "--dir", "/usr/lib",
                "--ro-bind", "/usr/lib/wsl", "/usr/lib/wsl",
                "--dev-bind", "/dev/dxg", "/dev/dxg",
            ))
        command.extend(
            (
                "--proc",
                "/proc",
                "--tmpfs",
                "/tmp",
                "--dev-bind",
                "/dev/null",
                "/dev/null",
                "--dev-bind",
                "/dev/urandom",
                "/dev/urandom",
                "--chdir",
                "/tmp",
                str(self.python_destination),
                "-I",
                "-S",
                "-B",
                "-X",
                "pycache_prefix=/dev/null",
                "-c",
                bootstrap,
                str(child_fd),
                "1" if test_policy else "0",
            )
        )
        return tuple(command)

    def cleanup(self) -> None:
        root = self.temporary_root
        if not root.name.startswith("libero-route-c-sandbox-") or root.parent != Path(
            "/tmp"
        ):
            raise FormalProcessSandboxError("refusing to remove an unknown sandbox path")
        _remove_projection_root(root)


__all__ = [
    "BWRAP_PATH",
    "EXCLUDED_POLICY_FILES",
    "FormalProcessSandboxError",
    "MASKED_SITE_PACKAGES",
    "POLICY_SOURCE_FILES",
    "POLICY_SOURCE_ROOT",
    "SANDBOX_ASSET_ROOT",
    "SANDBOX_HOSTNAME",
    "SANDBOX_RUNTIME_ROOT",
    "SYSTEM_RUNTIME_FILES",
    "SandboxProjection",
    "environment_sha256",
    "fixed_child_environment",
    "sha256_file",
    "tree_manifest_sha256",
    "validate_policy_runtime_clock_source",
]
