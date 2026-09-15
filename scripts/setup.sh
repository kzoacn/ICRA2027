#!/usr/bin/env bash
set -euo pipefail
anchor_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
anchor_deployment="${ANCHOR_DEPLOYED_ROOT:-${anchor_root}}"
anchor_bootstrap_python="${ANCHOR_BOOTSTRAP_PYTHON:-python3.12}"
anchor_venv="${ANCHOR_VENV:-${anchor_deployment}/.venv}"
"${anchor_bootstrap_python}" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
if [[ ! -x "${anchor_venv}/bin/python" ]]; then
    "${anchor_bootstrap_python}" -m venv "${anchor_venv}"
fi
ANCHOR_PYTHON="${anchor_venv}/bin/python" bash "${anchor_root}/scripts/install.sh"
ANCHOR_PYTHON="${anchor_venv}/bin/python" bash "${anchor_root}/scripts/run.sh" prepare
ANCHOR_PYTHON="${anchor_venv}/bin/python" bash "${anchor_root}/scripts/run.sh" doctor
