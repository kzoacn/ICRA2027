#!/usr/bin/env bash
set -euo pipefail
anchor_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
anchor_python="${ANCHOR_PYTHON:-python3}"
anchor_backend="${TORCH_BACKEND:-cu128}"
case "${anchor_backend}" in
    cpu|cu126|cu128|cu130) ;;
    *) echo 'TORCH_BACKEND must be cpu, cu126, cu128 or cu130.' >&2; exit 2 ;;
esac
"${anchor_python}" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
"${anchor_python}" -m pip install --no-cache-dir 'pip>=25,<27' 'setuptools==80.10.2'
"${anchor_python}" -m pip install --no-cache-dir \
    'torch==2.11.0' 'torchvision==0.26.0' \
    --index-url "https://download.pytorch.org/whl/${anchor_backend}"
"${anchor_python}" -m pip install --no-cache-dir -r "${anchor_root}/configs/requirements.txt"
"${anchor_python}" -m pip install --no-cache-dir --no-deps -r "${anchor_root}/configs/requirements-simulator.txt"
echo 'Python runtime installed. Next: bash scripts/run.sh prepare'
