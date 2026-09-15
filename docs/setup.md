# Setup and execution

Run the commands below from the repository root. The supported environment is
Linux x86-64 with Python 3.12. GPU execution requires an NVIDIA driver compatible
with the selected PyTorch build; MuJoCo renders through OSMesa.

## Install

On Ubuntu 24.04:

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv ca-certificates libosmesa6 libgl1 libglib2.0-0
TORCH_BACKEND=cu128 bash scripts/setup.sh
```

Setup creates `.venv/` and `resources/`, installs the pinned dependencies in
[configs/](../configs/), and downloads the detector and simulator assets listed
in [resources.lock.json](../configs/resources.lock.json). It does not download
robot demonstrations. `TORCH_BACKEND` accepts `cpu`, `cu126`, `cu128` or `cu130`.

Check resources, task compilation and rendering, then execute one episode:

```bash
bash scripts/run.sh doctor --render --device cuda
bash scripts/run.sh smoke --device cuda
```

The CLI is also available as `PYTHONPATH=src python3 -m anchor --help`.
Use `scripts/run.sh` for execution so the environment and rendering paths are set.

## Evaluate

Run a suite or a selected task:

```bash
bash scripts/run.sh run --suite object --episodes-per-task 10 --device cuda
bash scripts/run.sh run --suite goal --task-ids 3 --episodes-per-task 10 --device cuda
```

Run all 40 tasks with ten initial states each:

```bash
python3 scripts/run_batch.py --all --label libero400 --workers 4
```

Each batch freezes the source and writes its manifest, status, records and videos
to `runtime/evaluations/<label>/`. Use a fresh label for each run. Single CLI runs
write to `runtime/outputs/` by default. See [evaluation details](evaluation.md).

## Reuse an installed environment

A deployment directory contains `.venv/` and `resources/`. To run the current
repository source with an existing deployment:

```bash
ANCHOR_DEPLOYED_ROOT=/path/to/deployment bash scripts/run.sh doctor --render --device cuda
python3 scripts/run_batch.py --all --label reproduction --workers 4 --deployed-root /path/to/deployment
```

| Variable | Purpose |
| --- | --- |
| `ANCHOR_DEPLOYED_ROOT` | Environment and resource directory; defaults to the repository |
| `ANCHOR_PYTHON` | Python executable for setup dependencies or execution |
| `ANCHOR_MODEL_PATH` | Override the detector directory |
| `LIBERO_ASSET_ROOT` | Override the simulator asset directory |
| `ANCHOR_BOOTSTRAP_PYTHON` | Python 3.12 executable used to create the environment |
| `ANCHOR_VENV` | Environment path used by setup; set `ANCHOR_PYTHON` for later runs if customized |
| `LIBERO_CONFIG_PATH` | Writable LIBERO configuration directory |
| `TORCH_BACKEND` | PyTorch wheel backend selected during installation |

## Docker

Build from the repository root:

```bash
docker build -f docker/Dockerfile --build-arg TORCH_BACKEND=cu128 -t anchor .
mkdir -p resources runtime/outputs
docker run --rm -v "$PWD/resources:/opt/anchor/resources" anchor prepare
docker run --rm --gpus all -v "$PWD/resources:/opt/anchor/resources:ro" anchor doctor --render --device cuda
docker run --rm --gpus all -v "$PWD/resources:/opt/anchor/resources:ro" -v "$PWD/runtime/outputs:/opt/anchor/runtime/outputs" anchor smoke --device cuda
```

For CPU execution, build with `TORCH_BACKEND=cpu`, omit `--gpus all` and select
`--device cpu`. The image contains code and dependencies; resources and outputs
are mounted separately.
