#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/root/autodl-tmp/conda_envs/tbsi/bin/python"
CONFIG="v1.3.3-tsms-v3-full-15ep"
RUN_DIR="$ROOT/output/experiments/$CONFIG"
LOG_DIR="$RUN_DIR/logs"
CANONICAL="$ROOT/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
MIN_FREE_GB="${TSMS_V3_MIN_FREE_GB:-3}"

die() { echo "[TSMS-v3-full][拒绝启动] $*" >&2; exit 2; }
[[ -x "$PYTHON" ]] || die "Python 入口不存在"
[[ -s "$ROOT/experiments/tbsi_track/$CONFIG.yaml" ]] || die "配置不存在"
[[ -z "$(git -C "$ROOT" diff -- "$CANONICAL")" ]] || die "canonical baseline YAML 已被修改"
FREE_GB="$(df -Pk "$ROOT" | awk 'NR==2 {printf "%d", $4/1024/1024}')"
(( FREE_GB >= MIN_FREE_GB )) || die "磁盘余量 ${FREE_GB} GiB 低于 ${MIN_FREE_GB} GiB"
if pgrep -af 'tracking/(train|test|analysis_results)\.py' | grep -v grep >/dev/null; then
  die "检测到已有 TBSI 进程"
fi
[[ ! -e "$RUN_DIR" ]] || die "输出目录已存在，不覆盖"

mkdir -p "$LOG_DIR"
cat >"$LOG_DIR/manifest.txt" <<EOF
method=TSMS-v3 task-utility-ranking
config=$CONFIG
parent=v1.0.7-smsa-only-full-15ep
utility=-dLtrack/dA stop-gradient teacher
hard_frame_iou_threshold=0.50
utility_topk=8
utility_margin=0.01
utility_lambda=0.005
coverage=disabled
preservation=disabled
train_dataset=LasHeR_train
test_dataset=lasher_test
test_threads=0
infer_fp16=0
status=running
EOF

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TBSI_INFER_FP16=0 TBSI_ATTR_STATS=1
cd "$ROOT"
run_stage() {
  local name="$1"; shift
  local log="$LOG_DIR/${name}.stdout.log"
  echo "[$(date -Is)] START $name" | tee -a "$LOG_DIR/pipeline.log"
  "$@" >"$log" 2>&1
  echo "[$(date -Is)] DONE $name" | tee -a "$LOG_DIR/pipeline.log"
}
run_stage train "$PYTHON" -u tracking/train.py --script tbsi_track --config "$CONFIG" --save_dir ./output --mode single
run_stage test env TBSI_INFER_FP16=0 "$PYTHON" -u tracking/test.py tbsi_track "$CONFIG" --dataset_name lasher_test --threads 0 --num_gpus 1
run_stage analysis "$PYTHON" -u tracking/analysis_results.py --tracker_name tbsi_track --tracker_param "$CONFIG" --dataset_name lasher_test
sed -i 's/^status=.*/status=completed/' "$LOG_DIR/manifest.txt"
echo "[$(date -Is)] PIPELINE COMPLETE" | tee -a "$LOG_DIR/pipeline.log"
