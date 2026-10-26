#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
OUT="$ROOT/output/experiments/synthetic_joint_interference_v1"
mkdir -p "$OUT"
exec "$PYTHON_BIN" -u "$ROOT/tools/synthetic_joint_interference_v1.py" "$@" 2>&1 | tee -a "$OUT/launcher.log"
