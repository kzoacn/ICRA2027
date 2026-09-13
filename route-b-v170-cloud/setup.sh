#!/usr/bin/env bash
set -euo pipefail
route_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
route_bootstrap_python="${ROUTE_B_BOOTSTRAP_PYTHON:-python3.12}"
route_venv="${ROUTE_B_VENV:-${route_root}/.venv}"
"${route_bootstrap_python}" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
if [[ ! -x "${route_venv}/bin/python" ]]; then
    "${route_bootstrap_python}" -m venv "${route_venv}"
fi
ROUTE_B_PYTHON="${route_venv}/bin/python" bash "${route_root}/scripts/install.sh"
ROUTE_B_PYTHON="${route_venv}/bin/python" bash "${route_root}/run.sh" prepare
ROUTE_B_PYTHON="${route_venv}/bin/python" bash "${route_root}/run.sh" doctor
