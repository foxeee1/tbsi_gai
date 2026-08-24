#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/root/autodl-tmp/conda_envs/tbsi/bin/python"
CONFIG="v1.3.2-tsms-v2-rpp-15ep"
RUN_ROOT="$ROOT/output/proxy_runs/tsms-v2-rpp-20260822"
PROXY_ROOT="$ROOT/output/proxy_runs/rpp-v1-run-20260819-remaining3"
LOG_DIR="$RUN_ROOT/logs"
CANONICAL="$ROOT/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
MIN_FREE_GB="${TSMS_V2_MIN_FREE_GB:-3}"

die() { echo "[TSMS-v2][拒绝启动] $*" >&2; exit 2; }
[[ -x "$PYTHON" ]] || die "Python 入口不存在"
[[ -s "$ROOT/experiments/tbsi_track/$CONFIG.yaml" ]] || die "配置不存在"
[[ -s "$PROXY_ROOT/proxy_train_sequences.txt" ]] || die "proxy train list 缺失"
[[ -s "$PROXY_ROOT/proxy_test_sequences.txt" ]] || die "proxy test list 缺失"
[[ -s "$PROXY_ROOT/proxy_train_manifest.json" ]] || die "proxy train manifest 缺失"
[[ -s "$PROXY_ROOT/proxy_test_manifest.json" ]] || die "proxy test manifest 缺失"
[[ -z "$(git -C "$ROOT" diff -- "$CANONICAL")" ]] || die "canonical baseline YAML 已被修改"
FREE_GB="$(df -Pk "$ROOT" | awk 'NR==2 {printf "%d", $4/1024/1024}')"
(( FREE_GB >= MIN_FREE_GB )) || die "磁盘余量 ${FREE_GB} GiB 低于 ${MIN_FREE_GB} GiB"
if pgrep -af 'tracking/(train|test|analysis_results)\.py' | grep -v grep >/dev/null; then
  die "检测到已有 TBSI 进程"
fi
[[ ! -e "$RUN_ROOT" ]] || die "独立 run 目录已存在，不覆盖"
[[ ! -e "$ROOT/output/experiments/$CONFIG" ]] || die "配置输出目录已存在，不覆盖"

mkdir -p "$LOG_DIR"
cat >"$LOG_DIR/manifest.txt" <<EOF
method=TSMS-v2 relative-coverage-plus-JS-preservation
config=$CONFIG
parent=v1.0.7-rpp-anchor-smsa-15ep
proxy_root=$PROXY_ROOT
train_list=$PROXY_ROOT/proxy_train_sequences.txt
test_list=$PROXY_ROOT/proxy_test_sequences.txt
relative_delta=0.02
lambda_cov=0.02
lambda_preserve=0.01
test_threads=6
infer_fp16=0
status=running
EOF

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TBSI_INFER_FP16=0 TBSI_ATTR_STATS=1
export TBSI_RPP_TRAIN_LIST="$PROXY_ROOT/proxy_train_sequences.txt"
export TBSI_RPP_TEST_LIST="$PROXY_ROOT/proxy_test_sequences.txt"
cd "$ROOT"

"$PYTHON" -u tracking/train.py --script tbsi_track --config "$CONFIG" --save_dir ./output --mode single \
  >"$LOG_DIR/train.stdout.log" 2>&1
"$PYTHON" -u tracking/test.py tbsi_track "$CONFIG" --dataset_name rpp_lasher_test --threads 6 --num_gpus 1 \
  >"$LOG_DIR/test.stdout.log" 2>&1
"$PYTHON" -u tracking/analysis_results.py --tracker_name tbsi_track --tracker_param "$CONFIG" --dataset_name rpp_lasher_test \
  >"$LOG_DIR/analysis.stdout.log" 2>&1
sed -i 's/^status=.*/status=completed/' "$LOG_DIR/manifest.txt"
echo "[TSMS-v2] proxy run completed; full confirmation still requires gate review"
