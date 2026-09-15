#!/usr/bin/env python3
"""Prepare resources and evaluate ANCHOR in LIBERO."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
CAPS = {"spatial": 220, "object": 280, "goal": 300, "long": 520}
SUITES = {"spatial": "libero_spatial", "object": "libero_object",
          "goal": "libero_goal", "long": "libero_10"}


def resource_paths() -> dict[str, Path]:
    return {
        "assets": Path(os.environ.get("LIBERO_ASSET_ROOT", ROOT / "resources/assets")).expanduser().resolve(),
        "model": Path(os.environ.get("ANCHOR_MODEL_PATH", ROOT / "resources/grounding-dino-tiny")).expanduser().resolve(),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resource_errors(group: dict, directory: Path, *, full: bool) -> list[str]:
    errors = []
    for entry in group["files"]:
        path = directory / entry["path"]
        if not path.is_file():
            errors.append(f"missing: {entry['path']}")
        elif path.stat().st_size != entry["size"]:
            errors.append(f"size mismatch: {entry['path']}")
        elif full and file_sha256(path) != entry["sha256"]:
            errors.append(f"SHA256 mismatch: {entry['path']}")
    return errors


def check_resources(*, full: bool = False) -> dict[str, Path]:
    lock = json.loads((ROOT / "configs/resources.lock.json").read_text())
    paths = resource_paths()
    for name, path in paths.items():
        errors = resource_errors(lock[name], path, full=full)
        if errors:
            raise RuntimeError(f"{name}: {len(errors)} resource errors ({'; '.join(errors[:3])}). Run bash scripts/run.sh prepare")
    return paths


def configure_libero() -> None:
    """Resolve resource locations before LIBERO's import-time configuration."""
    import yaml

    paths = check_resources()
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("hf-libero is missing; run bash scripts/setup.sh first")
    benchmark_root = Path(next(iter(spec.submodule_search_locations))) / "libero"
    for name in ("bddl_files", "init_files"):
        if not (benchmark_root / name).is_dir():
            raise RuntimeError(f"hf-libero installation lacks {name}: {benchmark_root}")
    config_root = Path(os.environ.get("LIBERO_CONFIG_PATH", ROOT / "runtime/libero")).resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "assets": str(paths["assets"]),
        "datasets": str(ROOT / "runtime/unused-datasets"),
    }
    config_file = config_root / "config.yaml"
    content = yaml.safe_dump(config, sort_keys=True)
    if not config_file.exists() or config_file.read_text() != content:
        config_file.write_text(content)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
    os.environ["LIBERO_ASSET_ROOT"] = str(paths["assets"])
    os.environ["ANCHOR_MODEL_PATH"] = str(paths["model"])
    # hf-libero 0.1.4's mesh loader uses this cache instead of config.yaml.
    # Set its asset location without editing the installed third-party package.
    from libero import libero as benchmark_config
    benchmark_config._assets_path_cache = str(paths["assets"])


def prepare() -> int:
    # This explicit command is the only deployment operation allowed to fetch.
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    from huggingface_hub import snapshot_download

    lock = json.loads((ROOT / "configs/resources.lock.json").read_text())
    for name, directory in resource_paths().items():
        group = lock[name]
        errors = resource_errors(group, directory, full=True)
        if errors:
            print(f"Downloading {name}: {group['repo_id']}@{group['revision']}", flush=True)
            directory.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=group["repo_id"], repo_type=group["repo_type"],
                revision=group["revision"], local_dir=directory,
                allow_patterns=[item["path"] for item in group["files"]],
                force_download=any(not item.startswith("missing:") for item in errors),
            )
            errors = resource_errors(group, directory, full=True)
            if errors:
                raise RuntimeError(f"Downloaded {name} did not verify: {errors[:5]}")
        print(f"{name}: {len(group['files'])} files verified, {group['bytes'] / 1024**2:.1f} MiB")
    configure_libero()
    print("Resources are ready. Evaluation runs use local files only.")
    return 0


def offline_runtime() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    configure_libero()


def doctor(args: argparse.Namespace) -> int:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Python 3.12 is required")
    paths = check_resources(full=True)
    offline_runtime()
    import cv2
    import mujoco
    import numpy as np
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from libero.libero import benchmark, get_assets_path
    from libero.libero.envs import OffScreenRenderEnv
    from anchor.integration.adapters import AnchorPolicy
    from anchor.integration.config import source_tree_sha256
    from anchor.manipulation import TaskCompiler

    counts = {}
    compiler = TaskCompiler()
    for name in SUITES.values():
        suite = benchmark.get_benchmark_dict()[name]()
        counts[name] = len(suite.tasks)
        for task in suite.tasks:
            compiler.compile(task.language)
    if get_assets_path() != str(paths["assets"]):
        raise RuntimeError("LIBERO resolved the wrong asset directory")
    result = {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "numpy": np.__version__,
        "mujoco": mujoco.__version__, "opencv": cv2.__version__,
        "compiled_tasks": counts, "source_tree_sha256": source_tree_sha256(),
        "resource_hashes": "all verified", "render_checked": False,
    }
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the driver/container GPU option, or use --device cpu")
    if args.render:
        from anchor.common.env_adapter import LiberoEnvAdapter, LiberoEnvConfig
        from anchor.common.policy import OSCAction
        env = LiberoEnvAdapter(LiberoEnvConfig(suite_name="libero_object", task_id=0, seed=7))
        try:
            observation = env.reset()
            env.step(OSCAction.hold(-1.0))
            result["camera_shapes"] = {name: list(frame.rgb.shape) for name, frame in observation.cameras.items()}
            result["render_checked"] = True
        finally:
            env.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cli_args(args: argparse.Namespace, *, suite: str, name: str) -> list[str]:
    values = [
        "--route", "anchor", "--suite", SUITES[suite],
        "--task-ids", args.task_ids,
        "--episodes-per-task", str(args.episodes_per_task),
        "--init-state-start", str(args.init_start),
        "--device", args.device, "--seed", str(args.seed),
        "--image-size", "256", "--render-backend", "osmesa",
        "--perception-backend", "grounding-dino",
        "--perception-model", str(resource_paths()["model"]),
        "--libero-config-path", os.environ["LIBERO_CONFIG_PATH"],
        "--max-steps", str(args.max_steps if args.max_steps is not None else CAPS[suite]),
        "--run-name", name, "--output-dir", str(args.output_dir.resolve()),
        "--video-fps", "20", "--video-stride", "2",
    ]
    if args.video:
        values.append("--video")
    if args.resume:
        values.append("--resume")
    if args.dry_run:
        values.append("--dry-run")
    return values


def evaluate(args: argparse.Namespace) -> int:
    if args.resume and not args.run_name:
        raise ValueError("--resume requires the original --run-name")
    if not 0 <= args.init_start < 50 or not 1 <= args.episodes_per_task <= 50 - args.init_start:
        raise ValueError("Initial states must be a contiguous subset of 0..49")
    offline_runtime()
    from anchor.integration.cli import main as run_cli
    from anchor.integration.config import source_tree_sha256

    name = args.run_name or (datetime.now(timezone.utc).strftime("anchor_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:6])
    suites = list(CAPS) if args.command == "campaign" else [args.suite]
    reports = []
    for suite in suites:
        run_name = f"{name}_{suite}" if args.command == "campaign" else name
        status = run_cli(cli_args(args, suite=suite, name=run_name))
        if status:
            return status
        if not args.dry_run:
            summary_path = args.output_dir.resolve() / run_name / "summary.json"
            summary = json.loads(summary_path.read_text())
            reports.append({"suite": SUITES[suite], "summary": str(summary_path), **summary["overall"]})
    if args.command == "campaign" and not args.dry_run:
        episodes = sum(row["episodes"] for row in reports)
        successes = sum(row["successes"] for row in reports)
        result = {
            "kind": "cloud_evaluation", "controller_base": "v170",
            "source_tree_sha256": source_tree_sha256(),
            "episodes": episodes, "successes": successes,
            "success_rate": successes / episodes if episodes else None,
            "per_suite": reports,
        }
        path = args.output_dir.resolve() / f"{name}_campaign.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"Campaign summary: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="anchor", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Download pinned resources and verify their SHA256 hashes")
    check = sub.add_parser("doctor", help="Verify resources, runtime imports and all 40 task instructions")
    check.add_argument("--render", action="store_true", help="Also reset and step one MuJoCo environment")
    check.add_argument("--device", default="auto")
    for command in ("run", "smoke", "campaign"):
        item = sub.add_parser(command, help={"run": "Evaluate one suite", "smoke": "Evaluate Object task 0/init 0", "campaign": "Evaluate all four suites sequentially"}[command])
        item.add_argument("--suite", choices=tuple(CAPS), default="object" if command == "smoke" else "spatial")
        item.add_argument("--task-ids", default="0" if command == "smoke" else "0-9")
        item.add_argument("--episodes-per-task", type=int, default=1 if command == "smoke" else 5)
        item.add_argument("--init-start", type=int, default=0)
        item.add_argument("--max-steps", type=int)
        item.add_argument("--device", default="auto")
        item.add_argument("--seed", type=int, default=7)
        item.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
        item.add_argument("--output-dir", type=Path, default=ROOT / "runtime/outputs")
        item.add_argument("--run-name")
        item.add_argument("--resume", action="store_true")
        item.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        return prepare()
    if args.command == "doctor":
        return doctor(args)
    return evaluate(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"ANCHOR deployment error: {exc}", file=sys.stderr)
        raise SystemExit(1)
