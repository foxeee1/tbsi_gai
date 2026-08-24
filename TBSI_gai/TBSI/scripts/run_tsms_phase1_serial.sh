#!/usr/bin/env bash
set -euo pipefail

# Serial Phase-1 runner: 3-epoch full-data preflight -> 15-epoch full train ->
# full 244-sequence FP32 test. Default is dry-run; --run is required to launch.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/root/autodl-tmp/conda_envs/tbsi/bin/python"
PREFLIGHT="v1.3.1-tsms-full-3ep"
FULL="v1.3.1-tsms-full-15ep"
PREFLIGHT_DIR="$ROOT/output/experiments/$PREFLIGHT"
FULL_DIR="$ROOT/output/experiments/$FULL"
LOG_DIR="$ROOT/output/experiments/v1.3.1-tsms-phase1-serial/logs"
CANONICAL="$ROOT/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
MIN_FREE_GB=3
DO_RUN=0
[[ "${1:-}" == "--run" ]] && DO_RUN=1

die() { echo "[TSMS][拒绝启动] $*" >&2; exit 2; }
[[ -x "$PYTHON" ]] || die "Python 入口不存在"
[[ -f "$ROOT/experiments/tbsi_track/$PREFLIGHT.yaml" ]] || die "3 epoch 配置不存在"
[[ -f "$ROOT/experiments/tbsi_track/$FULL.yaml" ]] || die "full 配置不存在"
[[ -f "$CANONICAL" ]] || die "canonical baseline 缺失"
[[ -z "$(git -C "$ROOT" diff -- "$CANONICAL")" ]] || die "canonical baseline YAML 已被修改"
FREE_GB="$(df -Pk "$ROOT" | awk 'NR==2 {printf "%.0f", $4/1024/1024}')"

if (( DO_RUN == 1 )); then
  (( FREE_GB >= MIN_FREE_GB )) || die "磁盘剩余 ${FREE_GB} GiB，低于启动阈值 ${MIN_FREE_GB} GiB"
  if pgrep -af 'tracking/(train|test)\.py' | grep -v grep >/dev/null; then
    die "检测到已有 TBSI train/test 进程"
  fi
else
  echo "[TSMS][dry-run] disk_free=${FREE_GB}GiB, run_threshold=${MIN_FREE_GB}GiB"
fi

mkdir -p "$LOG_DIR"
cat > "$LOG_DIR/manifest.txt" <<EOF
phase=1
protocol=serial
parent=A1 SMSA-only
new_variable=TSMS coverage-only
preflight_config=$PREFLIGHT
full_config=$FULL
preflight_data=LasHeR_train full sequence sampler
full_data=LasHeR_train full sequence sampler
full_test=LasHeR_test 244 sequences
threads=0
infer_fp16=0
preflight_status=planned
full_status=blocked_until_preflight
EOF

echo "[TSMS] serial package ready: 3ep preflight -> 15ep full train -> 244-sequence test"
if (( DO_RUN == 0 )); then
  echo "[TSMS] no training started; run '$0 --run' after freeing at least ${MIN_FREE_GB} GiB"
  exit 0
fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TBSI_ATTR_STATS=1
cd "$ROOT"

PREFLIGHT_LOG="$PREFLIGHT_DIR/logs/training.log"
PREFLIGHT_COMPLETE=0
if [[ -s "$PREFLIGHT_LOG" ]] && grep -q "\[train: 3, 1872 / 1872\]" "$PREFLIGHT_LOG"; then
  PREFLIGHT_COMPLETE=1
  echo "[TSMS] reusing completed 3 epoch preflight" | tee -a "$LOG_DIR/preflight-3ep.training.log"
else
  sed -i 's/^preflight_status=.*/preflight_status=running/' "$LOG_DIR/manifest.txt"
  "$PYTHON" -u tracking/train.py --script tbsi_track --config "$PREFLIGHT" --save_dir ./output --mode single \
    2>&1 | tee "$LOG_DIR/preflight-3ep.training.log"
fi

[[ -s "$PREFLIGHT_LOG" ]] || die "3 epoch training.log 缺失"
if grep -Eiq '(^|[[:space:],:=])[-+]?(nan|inf)([[:space:],,]|$)|Traceback \(most recent call last\)|Training crashed' "$PREFLIGHT_LOG"; then
  sed -i 's/^preflight_status=.*/preflight_status=failed/' "$LOG_DIR/manifest.txt"
  die "3 epoch preflight 出现 NaN/Inf/异常，已停止，不进入 full"
fi
grep -q 'TSMS/available' "$PREFLIGHT_LOG" || die "3 epoch 日志未发现 TSMS 统计，已停止"
sed -i 's/^preflight_status=.*/preflight_status=passed/' "$LOG_DIR/manifest.txt"

# The preflight checkpoint is disposable; preserve its logs and metrics, then
# remove only this run's generated checkpoint before the full run.
find "$PREFLIGHT_DIR/checkpoints" -maxdepth 1 -type f -name 'TBSITrack_ep*.pth.tar' -delete 2>/dev/null || true

sed -i 's/^full_status=.*/full_status=running/' "$LOG_DIR/manifest.txt"
"$PYTHON" -u tracking/train.py --script tbsi_track --config "$FULL" --save_dir ./output --mode single \
  2>&1 | tee "$LOG_DIR/full-15ep.training.log"

export TBSI_INFER_FP16=0
"$PYTHON" -u tracking/test.py tbsi_track "$FULL" --dataset_name lasher_test --threads 0 --num_gpus 1 \
  2>&1 | tee "$LOG_DIR/full-244seq.test.log"
"$PYTHON" -u tracking/analysis_results.py --tracker_name tbsi_track --tracker_param "$FULL" --dataset_name lasher_test \
  2>&1 | tee "$LOG_DIR/full.analysis.log"
sed -i 's/^full_status=.*/full_status=completed/' "$LOG_DIR/manifest.txt"
echo "[TSMS] serial Phase 1 completed"
