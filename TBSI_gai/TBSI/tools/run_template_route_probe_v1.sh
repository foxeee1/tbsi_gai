#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
cd "$ROOT_DIR"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TBSI_INFER_FP16=0
exec "$PYTHON_BIN" -u tools/template_route_probe_v1.py "$@"
