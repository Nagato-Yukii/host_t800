#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export AMP_MJLAB_ROOT="${AMP_MJLAB_ROOT:-/home/ye/BeyondMimicExp/AMP_mjlab}"
export PYTHONPATH="$PWD:$AMP_MJLAB_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=4
exec "${T800_PYTHON:-/home/ye/miniconda3/envs/amp_mjlab/bin/python}" -m t800_amp_getup.cli "$@"
