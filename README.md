# ANCHOR

**Asset-Informed Geometric Skills for Language-Conditioned Manipulation**

ANCHOR combines a frozen 172M-parameter object detector, calibrated RGB-D,
asset geometry and geometric skills for language-conditioned robot manipulation.
No robot demonstrations are used to train an action policy.

The complete local LIBERO evaluation achieves **376/400 (94.0%)**:
Spatial **93%**, Object **98%**, Goal **98%**, Long **87%**.
See the [paper](paper/main.pdf) and [evaluation record](experiments/libero400/README.md).

## Run

Install the prerequisites in the [setup guide](docs/setup.md), then run from the repository root:

```bash
bash scripts/setup.sh
bash scripts/run.sh doctor --render --device cuda
bash scripts/run.sh smoke --device cuda
```

Run the full 400-episode evaluation:

```bash
python3 scripts/run_batch.py --all --label libero400 --workers 4
```

## Paper

```bash
make -C paper
make -C paper check
```

See [paper/README.md](paper/README.md) for data, figure generation and build requirements.

## Repository

| Directory | Contents |
| --- | --- |
| [src/anchor/](src/anchor/) | Perception, geometric skills and evaluation code |
| [scripts/](scripts/) | Setup, execution and repository checks |
| [configs/](configs/) | Dependency pins and resource checksums |
| [docker/](docker/) | Container build files |
| [tests/](tests/) | Geometry regression checks and fixtures |
| [paper/](paper/) | Manuscript, figures and measured data |
| [experiments/](experiments/) | Current evaluation record |
| [docs/](docs/) | Setup, evaluation and technical documentation |
| [archive/](archive/) | Earlier experiments and original source snapshots |

Local environments, resources and generated runs are excluded from Git.
See [repository conventions](docs/repository.md) for module names and archived records.
