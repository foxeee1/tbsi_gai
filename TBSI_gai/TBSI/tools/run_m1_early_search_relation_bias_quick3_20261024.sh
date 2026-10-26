#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON="/root/autodl-tmp/conda_envs/tbsi/bin/python"
NAME="M1_EARLY_SEARCH_RELATION_BIAS_QUICK3_20261024"
OUT="$ROOT/output/experiments/$NAME"
mkdir -p "$OUT/logs"
if [[ -e "$OUT/checkpoints/TBSITrack_ep0003.pth.tar" ]]; then
  echo "Refusing to overwrite existing checkpoint" >&2
  exit 2
fi
cd "$ROOT"
{
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "git_head=$(git rev-parse HEAD)"
  sha256sum experiments/tbsi_track/${NAME}.yaml
  sha256sum output/experiments/M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023/checkpoints/TBSITrack_ep0015.pth.tar
  "$PYTHON" -c 'import sys,torch; print(sys.version); print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NA")'
  nvidia-smi -L || true
} > "$OUT/provenance.txt" 2>&1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  "$PYTHON" -u tracking/train.py --script tbsi_track --config "$NAME" --save_dir output --mode single --use_lmdb 0 \
  > "$OUT/logs/train.log" 2>&1
test -s "$OUT/checkpoints/TBSITrack_ep0003.pth.tar"
echo "completed_utc=$(date -u +%FT%TZ)" >> "$OUT/provenance.txt"
