#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export AMP_MJLAB_ROOT="${AMP_MJLAB_ROOT:-$PWD/vendor/AMP_mjlab}"
export PYTHONPATH="$PWD:$AMP_MJLAB_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "${T800_PYTHON:-python}" -m t800_amp_getup.cli "$@"
