# Original server deployment

Route B v170 was deployed to the server's data disk and can be accessed through:

```bash
cd /root/route-b-upload/route-b-v170-cloud
./run.sh doctor --render --device cuda
./run.sh smoke --device cuda
```

The actual directory is `/root/autodl-tmp/route-b-v170-cloud`; `route-b-v170-cloud`
in the upload directory is a symbolic link to it. The Python environment, resources,
and evaluation outputs are all on the data disk, occupying approximately 9.3 GiB
after deployment. Launch through `run.sh`; manual virtual-environment activation is unnecessary.

Deployment validation results (2026-09-13 UTC):

- Ubuntu 22.04, Python 3.12.3, PyTorch 2.11.0+cu128, and an RTX 5090; actual CUDA computation passed.
- All 27 pinned project dependencies matched; all 585 simulator assets and 8 model files passed SHA256 verification.
- All 40 task instructions across the four suites compiled successfully; both fixed and wrist cameras provided 256×256 RGB images.
- `deployment_smoke_20260913`: LIBERO Object task 0 / init 0, 1/1 success, 127 steps, 39.571 seconds.
- Dual-camera video: 512×256 at 20 fps, with all 64 frames decodable.
- Deployment validation covered one task only; see [FULL_TEST.md](FULL_TEST.md) for the subsequent full campaign.

Results and logs (paths are relative to the actual deployment directory):

- `outputs/deployment_smoke_20260913/summary.json`: evaluation summary.
- `outputs/deployment_smoke_20260913/episodes.jsonl`: episode record.
- `outputs/deployment_smoke_20260913/videos/route-b_libero_object_task-00_ep-0000.mp4`: dual-camera video.
- `logs/doctor-render.json`, `logs/cuda-check.json`, and `logs/video-check.json`: environment and video checks.
- `logs/environment-freeze.txt`: complete version record of installed Python packages.
- `provenance/server-deployment.json`: deployment record for this host.

Host compatibility adjustments:

- Installed the system OSMesa library and related runtime dependencies.
- Linked `.venv/lib/libstdc++.so.6` to `/usr/lib/x86_64-linux-gnu/libstdc++.so.6` and configured `run.sh` to preload that environment's C++ library. This resolves the `GLIBCXX_3.4.30` conflict between the base Conda Python's older library and OSMesa/LLVM.
- Added `socksio==1.0.0`, required by `httpx[socks]==0.28.1`, to the project virtual environment to support the server's existing proxy configuration.
- Configured `.venv/pip.conf` to use `https://pypi.org/simple`; PyTorch uses the official cu128 index specified by the installer. Global Python and pip settings were not modified.

At deployment, the controller source fingerprint remained
`67c93be1df7da7572ecbb8faff53d01a47aea42ec36affb07ac54ba36e59edba`, matching the uploaded
package. Of its 104 files, only `run.sh` received the compatibility adjustments above.
The original is saved as `provenance/run.sh.uploaded`, with the diff in
`provenance/SERVER_RUNTIME.patch`. The original `SHA256SUMS` is retained;
`sha256sum -c SHA256SUMS.deployed` verifies the files of that historical server deployment.

Examples for subsequent runs:

```bash
# Spatial: 10 tasks, 5 initial states per task, 50 episodes in total.
./run.sh run --suite spatial --episodes-per-task 5 --device cuda

# Full four-suite evaluation: 2,000 episodes; use a new run name.
./run.sh campaign --episodes-per-task 50 --device cuda --run-name cloud_2000_01

# Resume an interrupted run with the same arguments and original run name.
./run.sh campaign --episodes-per-task 50 --device cuda --run-name cloud_2000_01 --resume
```

Once resources are prepared, evaluation uses local models and assets. New experiments
write to `outputs/<run_name>/`; omitting the run name generates a unique name automatically.
