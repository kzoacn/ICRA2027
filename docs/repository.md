# Repository conventions

The root contains this project's README, Git configuration and directories.
Current code and commands use the name **ANCHOR**. Put installation and technical
notes in `docs/`, current measured results in `experiments/`, and superseded
material in `archive/`. Keep generated runs, resources and virtual environments
out of Git.

## Source modules

| Module | Responsibility |
| --- | --- |
| `anchor.common` | Observations, camera geometry and simulator adapter |
| `anchor.perception` | Detection, depth processing and asset geometry |
| `anchor.manipulation` | Task compilation and geometric manipulation skills |
| `anchor.contact` | Contact skills and their local completion checks |
| `anchor.integration` | Evaluation, policy isolation, scoring and recording |
| `anchor.planning` | Retained planning modules and dependencies |

The public entry point is `bash scripts/run.sh`. Direct Python imports require
`PYTHONPATH=src`; the command-line module is `python3 -m anchor`.

## Naming migration

| Former location or name | Current location or name |
| --- | --- |
| `route-b-v170-cloud/libero_system/` | `src/anchor/` |
| `libero_system.route_b` | `anchor.manipulation` |
| `libero_system.route_c` | `anchor.planning` |
| `libero_system.goal_skills` | `anchor.contact` |
| `RouteB`, `route_b`, `ROUTE_B` | `Anchor`, `anchor`, `ANCHOR` |
| `route-b-v170-cloud/cloud.py` | `src/anchor/cli.py` |
| Deployment and server reports | `docs/history/`, `archive/deployment/` |
| Earlier development experiments | `archive/experiments/` |

[configs/naming_migration.json](../configs/naming_migration.json) records the path
mapping and exact transformations applied to the 93 original core source files.
It includes resource-path adjustments and descriptive CLI names. New records
retain the existing `b`/`c` method identifiers for compatibility with recorded
schemas; the CLI accepts `anchor` and `planning`.

## Evaluated source and historical records

The measured 94.0% result belongs to the original frozen source snapshot
`7bdcb97a7b847debeb0bd47c951455934701f81d7b561ad097fe5b41269de256`.
Its exact source files are preserved in
[full400-source.zip](../archive/evaluated-source/full400-source.zip), with checksums
in the adjacent [manifest](../archive/evaluated-source/manifest.json).

Renaming files and symbols changes the current source fingerprint. The historical
fingerprint and measurements remain unchanged. To verify the archive and that the
current core files match the recorded reorganization:

```bash
python3 scripts/source_archive.py
```

This migration check describes the reorganization, and should be revised when
future controller changes intentionally diverge from the evaluated source.
The paper's model audit resolves its recorded original source path through this
archive. Historical JSON/JSONL records, manifests and checksum files retain their
original identifiers and paths. Absolute host paths describe their original run.
Existing frozen directories under `runtime/` remain in place so recorded video
paths still resolve; new batches use `runtime/evaluations/`.

## Checks

From the repository root:

```bash
python3 scripts/check_repo.py
LIBERO_ASSET_ROOT=/path/to/resources/assets .venv/bin/python tests/check_scene_geometry.py
LIBERO_ASSET_ROOT=/path/to/resources/assets .venv/bin/python tests/check_drawer_geometry.py
make -C paper check
```

The repository check validates layout, Python syntax, local documentation links
and the evaluated source archive. Geometry checks require the installed simulator
and resources. The paper check validates the PDF and its numerical evidence.
