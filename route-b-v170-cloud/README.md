# Cloud deployment guide

Run ANCHOR's Route B implementation in the LIBERO simulator on a Linux cloud server.
The original deployment package was created from the v170 source snapshot, which had
completed a four-suite, 2,000-episode evaluation. It supports fixed and wrist RGB-D
cameras, geometric perception, grasp/place control, and contact skills. This repository
also includes the subsequent controller improvements documented in the
[improvement records](../experiments/route_b_90/README.md).

The deployment package contains runtime code, installation and launch scripts, and
resource versions and checksums. The Grounding DINO model (approximately 658 MiB) and
LIBERO simulator assets (approximately 403 MiB) are downloaded during initial server
setup. Demonstration datasets, SmolVLA, and SmolVLM are not downloaded.

## Recommended setup: Docker

Use a Linux x86_64 server with Docker. GPU execution also requires an NVIDIA driver
and NVIDIA Container Toolkit on the host. The container installs Python 3.12 and uses
OSMesa for headless rendering.

For the current repository version:

```bash
git clone git@github.com:kzoacn/ICRA2027.git
cd ICRA2027/route-b-v170-cloud
sha256sum -c SHA256SUMS.current

# NVIDIA GPU: select cu126, cu128, or cu130 to match the server driver.
docker build --build-arg TORCH_BACKEND=cu128 -t route-b:v170 .

mkdir -p resources outputs
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/resources:/opt/route-b/resources" \
  route-b:v170 prepare

docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/resources:/opt/route-b/resources:ro" \
  route-b:v170 doctor --render --device cuda

# Complete Object task 0 / init 0 and save dual-camera video and evaluation results.
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/resources:/opt/route-b/resources:ro" \
  -v "$PWD/outputs:/opt/route-b/outputs" \
  route-b:v170 smoke --device cuda
```

The original uploaded archive can be unpacked with `tar -xzf route-b-v170-cloud.tar.gz`.
Its historical `SHA256SUMS` applies to that archive's original files and filenames;
use `SHA256SUMS.current` for the current checkout.

For CPU-only execution, build with `TORCH_BACKEND=cpu`, omit `--gpus all`, and use
`--device cpu`. Resource preparation does not require a GPU. GPU wheel combinations
use torch 2.11.0 / torchvision 0.26.0 from the [official PyTorch installation list](https://pytorch.org/get-started/previous-versions/).
The original local environment used cu130; this package defaults to cu128 installation.

The examples map the container user to the current host user, so downloaded resources
and evaluation results belong to that user. Temporary container configuration and
Hugging Face caches are written to `/tmp`.

## Direct installation on an existing Ubuntu 24.04 server

Install system dependencies once. Subsequent Python packages are installed in this
directory's `.venv`.

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv ca-certificates \
  libosmesa6 libgl1 libglib2.0-0

TORCH_BACKEND=cu128 ./setup.sh
./run.sh doctor --render --device cuda
./run.sh smoke --device cuda
```

`setup.sh` installs dependencies, downloads pinned resources, and runs checks.
For CPU installation, use `TORCH_BACKEND=cpu ./setup.sh`. If an isolated Python 3.12
environment is already available, use
`ROUTE_B_PYTHON=/path/to/python TORCH_BACKEND=cu128 bash scripts/install.sh`, then use
the same `ROUTE_B_PYTHON` for `run.sh prepare` and subsequent commands.

## Running tasks

These examples use a direct installation. With Docker, append the same arguments
after `route-b:v170`.

```bash
# Spatial: 10 tasks, 5 initial states per task, 50 episodes in total.
./run.sh run --suite spatial --episodes-per-task 5 --device cuda

# Select tasks and initial states. Suites: spatial, object, goal, or long.
./run.sh run --suite goal --task-ids 0,3 --init-start 5 \
  --episodes-per-task 2 --device cuda --run-name goal_probe_01

# Four suites, 10 tasks each, 50 official initial states per task: 2,000 sequential episodes.
./run.sh campaign --episodes-per-task 50 --device cuda --run-name cloud_2000_01

# Inspect the effective scheduling arguments without running the simulator.
./run.sh run --suite spatial --episodes-per-task 5 --dry-run

# Save output space when videos are unnecessary.
./run.sh run --suite object --episodes-per-task 5 --no-video

# Resume with the same arguments and run name; use a new name for each new experiment.
./run.sh campaign --episodes-per-task 50 --device cuda \
  --run-name cloud_2000_01 --resume
```

Defaults are seed=7 and two 256×256 cameras. Spatial/Object/Goal/Long action budgets
are 220/280/300/520 steps, respectively. Per-suite results are saved in
`outputs/<run_name>/summary.json`, `episodes.jsonl`, and `videos/`. A four-suite
campaign also produces `<run_name>_campaign.json`. Omitting the run name generates
a unique name automatically.

The package provides cloud execution and reporting entry points. The 85.9% score is
the historical formal v170 result from the original environment. A packaged-environment
check or single-task smoke run does not replace a fresh complete 2,000-episode evaluation.
When using different PyTorch CUDA wheels, OSMesa versions, or hardware, preserve the
new run's own measured score. The current ANCHOR result and its 400-episode protocol
are documented in the [project overview](../README.md).

## Resources and paths

| Resource | Pinned source |
|---|---|
| Perception model | `IDEA-Research/grounding-dino-tiny@a2bb814dd30d776dcf7e30523b00659f4f141c71` |
| Simulator assets | `lerobot/libero-assets@0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`, dataset repository |
| Task definitions and initial states | Bundled with `hf-libero==0.1.4` |
| Simulator | `robosuite==1.4.0`, `mujoco==3.3.2` |

`resources.lock.json` pins the size and SHA256 of every model and asset file.
`prepare` and `doctor` verify all files. Evaluation uses local resources and disables
Hugging Face online lookups.

Point to existing resources with environment variables to avoid downloading them again:

```bash
export LIBERO_ASSET_ROOT=/data/libero-assets
export ROUTE_B_MODEL_PATH=/data/grounding-dino-tiny
export LIBERO_CONFIG_PATH=/data/route-b-runtime/libero
./run.sh doctor --render
```

`run.sh` configures dependency and resource paths; launch through this entry point.
LIBERO YAML configuration is generated for the current installation location, so the
original machine's user directories and Conda paths do not need to exist. If downloads
require a proxy, use the cloud server's own network/proxy settings. The package contains
no account credentials, tokens, or proxy configuration.

## Package contents and original packaging choices

- `libero_system/`: v170 Python runtime code. Shared Route C types and geometry modules
  are retained because existing perception/integration modules import them; the launch
  entry point runs Route B.
- `cloud.py`, `run.sh`, `setup.sh`, and `Dockerfile`: deployment entry points and installation methods.
- `provenance/`: original score summaries, source differences, and packaging validation summaries.
- The original deployment package excludes historical candidates/development files,
  videos, sensor dumps, research logs, old README files, tests, `.git`, `.local`,
  virtual environments, and model caches.

The three upstream simulator packages are installed with `--no-deps`; required runtime
dependencies are then installed explicitly from `requirements.txt`. This avoids
hf-libero's training, experiment-tracking, and notebook dependencies. `doctor` checks
runtime imports and compilation of all 40 task instructions; `smoke` validates the
perception and control pipeline using the actual MuJoCo simulator.

Video encoding uses the encoder bundled with the `imageio-ffmpeg` wheel. A separate
system FFmpeg installation and its audio/video dependencies are unnecessary.

The original portability changes preserved the v170 controller, perception, and
contact-skill functions. Two default model/asset paths were replaced with environment
variables and project-relative paths. See `provenance/PORTABILITY.patch` and
`provenance/source.json` for those changes. The deployment entry points handle resource
location, arguments, and result summaries. Later controller improvements are recorded
separately in the improvement records linked above.
