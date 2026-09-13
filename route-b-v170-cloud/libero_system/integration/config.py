"""Validated command-line configuration for real LIBERO evaluations."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import re
import stat
from typing import Any, Sequence

from libero_system.common.env_adapter import UNIFIED_EVALUATION_HORIZON


SUITE_ALIASES = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "90": "libero_90",
    "10": "libero_10",
    "libero_spatial": "libero_spatial",
    "libero_object": "libero_object",
    "libero_goal": "libero_goal",
    "libero_90": "libero_90",
    "libero_10": "libero_10",
}

SUITE_TASK_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_90": 90,
    "libero_10": 10,
}

DEFAULT_MAX_STEPS = {
    suite: UNIFIED_EVALUATION_HORIZON
    for suite in SUITE_TASK_COUNTS
}


def source_tree_sha256() -> str:
    """Fingerprint the Python implementation loaded by a formal route run.

    Paths and bytes are hashed in sorted order.  Because this value is stored
    in ``run_config``, resume provenance rejects a source-code change instead
    of silently mixing episodes from different controller versions.
    """

    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256(b"libero-source-tree-exact.v2\0")
    allowed_suffixes = {".py", ".md"}
    entries: list[tuple[str, Path | None]] = []
    for current, directories, filenames in os.walk(package_root, followlinks=False):
        current_path = Path(current)
        if current_path.name == "__pycache__":
            directories[:] = []
            continue
        for name in tuple(directories):
            candidate = current_path / name
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(f"formal source tree contains a symlink: {candidate}")
            if not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError(
                    f"formal source tree contains a non-directory entry: {candidate}"
                )
        directories[:] = sorted(name for name in directories if name != "__pycache__")
        relative_directory = current_path.relative_to(package_root).as_posix()
        entries.append((f"D:{relative_directory}", None))
        for name in sorted(filenames):
            candidate = current_path / name
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(
                    f"formal source tree contains an unsupported entry: {candidate}"
                )
            if metadata.st_nlink != 1:
                raise RuntimeError(f"formal source file must have one link: {candidate}")
            if candidate.suffix in {".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib"}:
                raise RuntimeError(
                    f"formal source tree contains executable shadow bytes: {candidate}"
                )
            if candidate.suffix not in allowed_suffixes:
                raise RuntimeError(
                    f"formal source tree contains an unexpected file: {candidate}"
                )
            relative = candidate.relative_to(package_root).as_posix()
            entries.append((f"F:{relative}", candidate))
    for tagged_name, path in sorted(entries, key=lambda item: item[0]):
        relative = tagged_name.encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        if path is None:
            digest.update((0).to_bytes(8, "big"))
            continue
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def normalize_suite(value: str) -> str:
    try:
        return SUITE_ALIASES[value.strip().lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "suite must be spatial, object, goal, 90, 10, or the corresponding libero_* name"
        ) from exc


def parse_task_ids(value: str) -> tuple[int, ...]:
    """Parse comma-separated IDs and inclusive ranges such as ``0-3,7,9``."""

    ids: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        if match is None:
            raise argparse.ArgumentTypeError(f"invalid task id/range: {token!r}")
        start = int(match.group(1))
        stop = int(match.group(2) or start)
        if stop < start:
            raise argparse.ArgumentTypeError(f"descending task range is not allowed: {token!r}")
        ids.extend(range(start, stop + 1))
    unique = tuple(dict.fromkeys(ids))
    if not unique:
        raise argparse.ArgumentTypeError("at least one task ID is required")
    return unique


def resolve_device(value: str) -> str:
    value = value.strip().lower()
    if value == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if value == "cpu" or re.fullmatch(r"cuda(?::\d+)?", value):
        return value
    raise argparse.ArgumentTypeError("device must be auto, cpu, cuda, or cuda:N")


@dataclass(frozen=True)
class EvaluationConfig:
    route: str
    suite: str
    task_ids: tuple[int, ...]
    episodes_per_task: int
    output_dir: Path
    run_name: str
    device: str
    perception_backend: str
    perception_model: Path | None
    perception_factory: str | None
    libero_config_path: Path
    max_steps: int
    image_size: int
    seed: int
    resume: bool
    record_video: bool
    video_fps: int
    video_stride: int
    render_backend: str
    # First fixed LIBERO init-state index used for every selected task.  This
    # is part of run provenance so held-out schedules cannot be resumed into
    # the historical 0-based runs.
    init_state_start: int = 0
    formal_provenance_path: Path | None = None
    formal_provenance_sha256: str | None = None

    @property
    def formal(self) -> bool:
        return self.formal_provenance_path is not None

    @property
    def total_episodes(self) -> int:
        return len(self.task_ids) * self.episodes_per_task

    @property
    def trace_path(self) -> Path:
        return self.output_dir / self.run_name / "episodes.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / self.run_name / "summary.json"

    @property
    def video_dir(self) -> Path:
        return self.output_dir / self.run_name / "videos"

    def serializable(self) -> dict[str, Any]:
        result = asdict(self)
        for key, value in tuple(result.items()):
            if isinstance(value, Path):
                result[key] = str(value)
        result["total_episodes"] = self.total_episodes
        result["source_tree_sha256"] = source_tree_sha256()
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m libero_system.integration",
        description="Evaluate sensor-only Route B or Route C in real LIBERO episodes.",
    )
    parser.add_argument("--route", choices=("b", "c"), required=True)
    parser.add_argument(
        "--suite",
        type=normalize_suite,
        choices=tuple(dict.fromkeys(SUITE_ALIASES.values())),
        required=True,
    )
    parser.add_argument(
        "--task-ids",
        type=parse_task_ids,
        default=None,
        help="task IDs/ranges; defaults to every task in the selected suite",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=5,
        help="episodes for every selected task",
    )
    parser.add_argument(
        "--init-state-start",
        type=int,
        default=0,
        help="first fixed LIBERO init-state index for every selected task",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/routes"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--perception-backend",
        choices=("auto", "gallery", "grounding-dino"),
        default="auto",
    )
    parser.add_argument(
        "--perception-model",
        type=Path,
        default=None,
        help="local frozen detector snapshot; no weights are trained by this runner",
    )
    parser.add_argument(
        "--perception-factory",
        default=None,
        help="advanced module:callable override receiving a PerceptionFactoryContext",
    )
    parser.add_argument(
        "--libero-config-path",
        type=Path,
        default=Path(".local/libero"),
        help="directory containing LIBERO config.yaml",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--formal-provenance",
        type=Path,
        default=None,
        help="immutable parent-created shard provenance (formal campaigns only)",
    )
    parser.add_argument(
        "--formal-provenance-sha256",
        default=None,
        help="expected SHA-256 of --formal-provenance",
    )
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--render-backend", choices=("osmesa", "egl"), default="osmesa")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the schedule without importing MuJoCo or models",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> EvaluationConfig:
    episodes = int(args.episodes_per_task)
    if episodes < 1:
        raise ValueError("--episodes-per-task must be positive")
    init_state_start = int(args.init_state_start)
    if init_state_start < 0:
        raise ValueError("--init-state-start must be non-negative")
    if args.image_size < 64 or args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--image-size must be >=64 and --max-steps must be positive")
    if args.video_fps < 1 or args.video_stride < 1:
        raise ValueError("video fps and stride must be positive")
    if (args.formal_provenance is None) != (
        args.formal_provenance_sha256 is None
    ):
        raise ValueError(
            "--formal-provenance and --formal-provenance-sha256 must be provided together"
        )
    if args.formal_provenance is not None and args.resume:
        raise ValueError("formal campaign runs are fresh-only; --resume is forbidden")
    if args.formal_provenance_sha256 is not None and re.fullmatch(
        r"[0-9a-f]{64}", str(args.formal_provenance_sha256)
    ) is None:
        raise ValueError(
            "--formal-provenance-sha256 must be 64 lowercase hex characters"
        )
    device = resolve_device(args.device)
    model = args.perception_model.expanduser().resolve() if args.perception_model else None
    if model is not None and not model.exists():
        raise ValueError(f"local perception model path does not exist: {model}")
    suite = str(args.suite)
    task_count = SUITE_TASK_COUNTS[suite]
    task_ids = tuple(range(task_count)) if args.task_ids is None else tuple(args.task_ids)
    if any(task_id < 0 or task_id >= task_count for task_id in task_ids):
        raise ValueError(f"--task-ids for {suite} must be within 0..{task_count - 1}")
    run_name = args.run_name or f"route_{args.route}_{suite}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError("--run-name may contain only letters, digits, dot, dash, and underscore")
    return EvaluationConfig(
        route=args.route,
        suite=suite,
        task_ids=task_ids,
        episodes_per_task=episodes,
        output_dir=args.output_dir.expanduser().resolve(),
        run_name=run_name,
        device=device,
        perception_backend=args.perception_backend,
        perception_model=model,
        perception_factory=args.perception_factory,
        libero_config_path=args.libero_config_path.expanduser().resolve(),
        max_steps=args.max_steps or DEFAULT_MAX_STEPS[suite],
        image_size=int(args.image_size),
        seed=int(args.seed),
        resume=bool(args.resume),
        record_video=bool(args.video),
        video_fps=int(args.video_fps),
        video_stride=int(args.video_stride),
        render_backend=str(args.render_backend),
        init_state_start=init_state_start,
        formal_provenance_path=(
            args.formal_provenance.expanduser().resolve()
            if args.formal_provenance is not None
            else None
        ),
        formal_provenance_sha256=args.formal_provenance_sha256,
    )


def parse_config(argv: Sequence[str] | None = None) -> tuple[EvaluationConfig, bool]:
    args = build_parser().parse_args(argv)
    return config_from_args(args), bool(args.dry_run)
