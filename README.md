# ANCHOR · ICRA 2027

**ANCHOR: Asset-Informed Geometric Skills for Language-Conditioned Manipulation**.
The project builds on the original Route B v170 implementation. Source code is in
[`route-b-v170-cloud/`](route-b-v170-cloud/), including the controller, perception
and LIBERO evaluation code, pinned dependencies, resource checksums, and deployment scripts.

The current frozen version achieves **360/400 (90.0%)** in a fresh 400-episode run,
compared with the 347/400 baseline. Spatial / Object / Goal / Long have
**93 / 98 / 91 / 78** successes, respectively, out of 100 episodes each.
The run preserves all 40 tasks, official initial-state indices 0–9, seed 7,
the original action budgets, and external ever-success scoring.
Thirteen previous failures become successes, with no previously successful episode regressing.
See [accepted_result.json](experiments/route_b_90/accepted_result.json) for the
complete validation and task-by-task comparison.

## Getting started

```bash
git clone git@github.com:kzoacn/ICRA2027.git
cd ICRA2027/route-b-v170-cloud
sha256sum -c SHA256SUMS.current
```

See the [deployment guide](route-b-v170-cloud/README.md) for system dependencies,
Python 3.12 setup, and Docker instructions. After installing the system dependencies:

```bash
TORCH_BACKEND=cu128 ./setup.sh
./run.sh doctor --render --device cuda
./run.sh smoke --device cuda

# Run all four suites sequentially: 40 tasks, 10 official initial states each, 400 episodes.
./run.sh campaign --episodes-per-task 10 --device cuda --run-name cloud_400_01
```

`setup.sh` installs dependencies and downloads the model and simulator assets pinned
in `resources.lock.json`. Virtual environments, downloaded resources, logs, evaluation
outputs, and videos remain on the execution host and are excluded from Git.

## Deployment and evaluation records

- [Server deployment and validation](DEPLOYMENT.md)
- [Original 400-episode parallel evaluation](RUN_400_PARALLEL.md)
- [Original 2,000-episode campaign record](FULL_TEST.md)
- [LIBERO evaluation protocol review](LIBERO_EVALUATION_PROTOCOLS.md)
- [90% improvement objective, development records, and full reevaluation](experiments/route_b_90/README.md)

Absolute paths and process information in these records refer to the original
deployment server. `route-b-v170-cloud/runtime/jobs/` retains that server's dispatcher
scripts and fixed configurations. Their manifests contain host-specific paths and
references to reused historical results, so they cannot be resumed directly on a new
machine. Use the standard entry points above for new evaluations.

The repository retains the original deployment provenance alongside the improved
current source. `SHA256SUMS` and `SHA256SUMS.deployed` preserve the historical checksums
and filenames of the uploaded package and original server deployment, respectively.
Use `SHA256SUMS.current` to verify the current files. Provenance and source differences
are in [`provenance/`](route-b-v170-cloud/provenance/); improvement records are in
[`experiments/route_b_90/`](experiments/route_b_90/).

## ICRA 2027 manuscript

The anonymous English draft, LaTeX source, PDF, figures, tables, and verifiable
experimental statistics are in [paper/](paper/).

- [Read the paper PDF](paper/main.pdf)
- [LaTeX source](paper/main.tex)
- [Build instructions and scope of the evidence](paper/README.md)
