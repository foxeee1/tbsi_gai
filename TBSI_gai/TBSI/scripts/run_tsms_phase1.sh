#!/usr/bin/env bash
set -euo pipefail

# Phase 1 unified runner. Default is a safe dry-run; pass --run to launch.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/root/autodl-tmp/conda_envs/tbsi/bin/python"
CONFIG="v1.3.0-tsms-rpp-15ep"
RUN_NAME="v1.3.0-tsms-rpp-15ep"
RUN_ROOT="$ROOT/output/experiments/$RUN_NAME"
PROXY_ROOT="$ROOT/output/proxy_runs/rpp-v1-run-20260819-remaining3"
TRAIN_LIST="$PROXY_ROOT/proxy_train_sequences.txt"
TEST_LIST="$PROXY_ROOT/proxy_test_sequences.txt"
LOG_DIR="$RUN_ROOT/logs"
MIN_FREE_GB="6"
DO_RUN=0

if [[ "${1:-}" == "--run" ]]; then DO_RUN=1; fi
CANONICAL="$ROOT/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"

die() { echo "[TSMS][拒绝启动] $*" >&2; exit 2; }
[[ -x "$PYTHON" ]] || die "Python 入口不存在: $PYTHON"
[[ -f "$ROOT/experiments/tbsi_track/$CONFIG.yaml" ]] || die "配置不存在"
[[ -f "$TRAIN_LIST" && -f "$TEST_LIST" ]] || die "冻结 proxy sequence list 缺失"
[[ -f "$CANONICAL" ]] || die "canonical baseline 缺失"
[[ -z "$(git -C "$ROOT" diff -- "$CANONICAL")" ]] || die "canonical baseline YAML 已被修改"
FREE_GB="$(df -Pk "$ROOT" | awk 'NR==2 {printf "%.0f", $4/1024/1024}')"
if (( DO_RUN == 1 )); then
  (( FREE_GB >= MIN_FREE_GB )) || die "磁盘剩余 ${FREE_GB} GiB，低于安全阈值 ${MIN_FREE_GB} GiB"
  if pgrep -af 'tracking/(train|test)\.py' | grep -v grep >/dev/null; then
    die "检测到已有 TBSI train/test 进程"
  fi
else
  echo "[TSMS][dry-run] disk_free=${FREE_GB}GiB (run threshold=${MIN_FREE_GB}GiB)"
fi

mkdir -p "$LOG_DIR"
cat > "$LOG_DIR/run_manifest.txt" <<EOF
phase=1
group=A2
method=SMSA+TSMS coverage-only
config=$CONFIG
parent_smsa=v1.0.7-rpp-anchor-smsa-15ep
proxy_root=$PROXY_ROOT
train_list=$TRAIN_LIST
test_list=$TEST_LIST
test_threads=0
infer_fp16=0
status=planned
EOF

echo "[TSMS] Phase 1 package ready"
echo "[TSMS] A0=canonical baseline; A1=existing SMSA-only; A2=$CONFIG"
echo "[TSMS] train proxy=$(wc -l < "$TRAIN_LIST") sequences; test proxy=$(wc -l < "$TEST_LIST") sequences"
echo "[TSMS] disk_free=${FREE_GB}GiB; threads=0; TBSI_INFER_FP16=0"
if (( DO_RUN == 0 )); then
  echo "[TSMS] dry-run only. Review the package, then run: $0 --run"
  exit 0
fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TBSI_RPP_TRAIN_LIST="$TRAIN_LIST"
export TBSI_RPP_TEST_LIST="$TEST_LIST"
export TBSI_ATTR_STATS=1

sed -i 's/^status=.*/status=running/' "$LOG_DIR/run_manifest.txt"
(
  cd "$ROOT"
  "$PYTHON" -u tracking/train.py --script tbsi_track --config "$CONFIG" --save_dir ./output --mode single
) 2>&1 | tee "$LOG_DIR/training.stdout.log"

export TBSI_INFER_FP16=0
(
  cd "$ROOT"
  "$PYTHON" -u tracking/test.py tbsi_track "$CONFIG" --dataset_name rpp_lasher_test --threads 0 --num_gpus 1
) 2>&1 | tee "$LOG_DIR/test.stdout.log"
(
  cd "$ROOT"
  "$PYTHON" -u tracking/analysis_results.py --tracker_name tbsi_track --tracker_param "$CONFIG" --dataset_name rpp_lasher_test
) 2>&1 | tee "$LOG_DIR/analysis.stdout.log"
sed -i 's/^status=.*/status=completed/' "$LOG_DIR/run_manifest.txt"
echo "[TSMS] completed; proxy metrics are mechanism/screening evidence only because RPP gate is not passed"
