#!/usr/bin/env bash
set -euo pipefail
anchor_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
anchor_deployment="${ANCHOR_DEPLOYED_ROOT:-${anchor_root}}"
anchor_default_python="${anchor_deployment}/.venv/bin/python"
if [[ ! -x "${anchor_default_python}" ]]; then anchor_default_python=python3; fi
anchor_python="${ANCHOR_PYTHON:-${anchor_default_python}}"
export PYTHONPATH="${anchor_root}/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export NUMBA_DISABLE_JIT=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/anchor-matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/anchor-numba}"
export LIBERO_ASSET_ROOT="${LIBERO_ASSET_ROOT:-${anchor_deployment}/resources/assets}"
export ANCHOR_MODEL_PATH="${ANCHOR_MODEL_PATH:-${anchor_deployment}/resources/grounding-dino-tiny}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${anchor_deployment}/runtime/libero}"
anchor_prefix_lib="$("${anchor_python}" -c 'import sys; print(sys.prefix + "/lib")')"
anchor_base_lib="$("${anchor_python}" -c 'import sys; print(sys.base_prefix + "/lib")')"
export LD_LIBRARY_PATH="${anchor_prefix_lib}:${anchor_base_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# Prefer the environment's C++ runtime over a base Conda Python's DT_RPATH.
if [[ -r "${anchor_prefix_lib}/libstdc++.so.6" ]]; then
    export LD_PRELOAD="${anchor_prefix_lib}/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
elif [[ -r "${anchor_base_lib}/libstdc++.so.6" ]]; then
    export LD_PRELOAD="${anchor_base_lib}/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
exec "${anchor_python}" -m anchor "$@"
