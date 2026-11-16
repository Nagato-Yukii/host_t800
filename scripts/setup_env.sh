#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use a dedicated Python 3.11 environment"'
python -c 'import torch; assert torch.cuda.is_available(), "Install a compatible CUDA PyTorch build first"; print(torch.__version__, torch.cuda.get_device_name(0))'
python -m pip install -r requirements-runtime.txt
python -m pip install --no-deps -e .
# Use vendored src and rsl_rl through run.sh PYTHONPATH; do not replace installed packages.
python scripts/patch_mjlab.py
./run.sh check
