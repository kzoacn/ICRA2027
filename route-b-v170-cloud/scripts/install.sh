#!/usr/bin/env bash
set -euo pipefail
route_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
route_python="${ROUTE_B_PYTHON:-python3}"
route_backend="${TORCH_BACKEND:-cu128}"
case "${route_backend}" in
    cpu|cu126|cu128|cu130) ;;
    *) echo 'TORCH_BACKEND must be cpu, cu126, cu128 or cu130.' >&2; exit 2 ;;
esac
"${route_python}" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
"${route_python}" -m pip install --no-cache-dir 'pip>=25,<27' 'setuptools==80.10.2'
"${route_python}" -m pip install --no-cache-dir \
    'torch==2.11.0' 'torchvision==0.26.0' \
    --index-url "https://download.pytorch.org/whl/${route_backend}"
"${route_python}" -m pip install --no-cache-dir -r "${route_root}/requirements.txt"
"${route_python}" -m pip install --no-cache-dir --no-deps -r "${route_root}/requirements-simulator.txt"
echo 'Python runtime installed. Next: ./run.sh prepare'
