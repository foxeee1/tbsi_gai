#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/root/autodl-tmp/conda_envs/tbsi/bin/python"
PROXY_ROOT="$ROOT/output/proxy_runs/rpp-v1-run-20260819-remaining3"
CAMPAIGN="$ROOT/output/experiments/tcmr-phase2-campaign-20260823"
LOCK_DIR="$ROOT/output/locks"
LOCK_FILE="$LOCK_DIR/tcmr-phase2.lock"
CANONICAL="$ROOT/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
MIN_FREE_GB="${TCMR_MIN_FREE_GB:-3}"
RESUME_A3="${TCMR_RESUME_A3:-0}"
RESUME_A4="${TCMR_RESUME_A4:-0}"
SKIP_A3="${TCMR_SKIP_A3:-0}"

die() { echo "[TCMR-phase2][拒绝启动] $*" >&2; exit 2; }
mkdir -p "$LOCK_DIR" "$CAMPAIGN"
exec 9>"$LOCK_FILE"
flock -n 9 || die "已有 TCMR campaign 持有锁"
[[ -x "$PYTHON" ]] || die "Python 入口不存在"
[[ -s "$PROXY_ROOT/proxy_train_sequences.txt" ]] || die "proxy train list 缺失"
[[ -s "$PROXY_ROOT/proxy_test_sequences.txt" ]] || die "proxy test list 缺失"
[[ -s "$PROXY_ROOT/proxy_train_manifest.json" ]] || die "proxy train manifest 缺失"
[[ -s "$PROXY_ROOT/proxy_test_manifest.json" ]] || die "proxy test manifest 缺失"
[[ -z "$(git -C "$ROOT" diff -- "$CANONICAL")" ]] || die "canonical baseline YAML 已被修改"
if pgrep -af 'tracking/(train|test|analysis_results)\.py' | grep -v grep >/dev/null; then
  die "检测到已有 TBSI 进程"
fi
FREE_GB="$(df -Pk "$ROOT" | awk 'NR==2 {printf "%d", $4/1024/1024}')"
(( FREE_GB >= MIN_FREE_GB )) || die "磁盘余量 ${FREE_GB} GiB 低于 ${MIN_FREE_GB} GiB"

printf '%s\n' \
  'campaign=tcmr-phase2-20260823' \
  'sequence=TCMR_P0 -> A3_proxy -> A4_proxy -> A4_full' \
  'proxy_protocol=RPP_LasHeR_train/rpp_lasher_test, 15 epochs, threads=0 (serial)' \
  'full_protocol=LasHeR_train/lasher_test, 15 epochs, FP32, threads=0' \
  'parent=TSMS-v3 v1.3.3' \
  'status=running' > "$CAMPAIGN/manifest.txt"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TBSI_INFER_FP16=0 TBSI_ATTR_STATS=1
export TBSI_RPP_TRAIN_LIST="$PROXY_ROOT/proxy_train_sequences.txt"
export TBSI_RPP_TEST_LIST="${TCMR_TEST_LIST:-$PROXY_ROOT/proxy_test_sequences.txt}"
cd "$ROOT"

run_logged() {
  local name="$1"; shift
  local log="$CAMPAIGN/${name}.console.log"
  echo "[$(date -Is)] START $name" | tee -a "$CAMPAIGN/pipeline.log"
  "$@" >"$log" 2>&1
  echo "[$(date -Is)] DONE $name" | tee -a "$CAMPAIGN/pipeline.log"
}

if [[ "$SKIP_A3" != "1" ]]; then
  run_logged tcmr_p0 "$PYTHON" scripts/tcmr_p0_check.py
fi

run_proxy() {
  local stage="$1"
  local config="$2"
  local run_root="$ROOT/output/proxy_runs/tcmr-phase2-20260823/$stage"
  local log_dir="$run_root/logs"
  local checkpoint="$ROOT/output/experiments/$config/checkpoints/TBSITrack_ep0015.pth.tar"
  if [[ "$stage" == "a3" && "$RESUME_A3" == "1" ]]; then
    [[ -f "$checkpoint" ]] || die "A3 resume checkpoint missing: $checkpoint"
  elif [[ "$stage" == "a4" && "$RESUME_A4" == "1" ]]; then
    [[ -f "$checkpoint" ]] || die "A4 resume checkpoint missing: $checkpoint"
  else
    [[ ! -e "$run_root" ]] || die "$stage proxy output exists: $run_root"
    [[ ! -e "$ROOT/output/experiments/$config" ]] || die "$stage experiment output exists"
  fi
  mkdir -p "$log_dir"
  printf '%s\n' \
    "stage=$stage" "config=$config" "proxy_root=$PROXY_ROOT" \
    'train=RPP_LasHeR_train' 'test=rpp_lasher_test' \
    'test_threads=0' 'infer_fp16=0' 'status=running' > "$log_dir/manifest.txt"
  if [[ ! ( "$stage" == "a3" && "$RESUME_A3" == "1" ) && ! ( "$stage" == "a4" && "$RESUME_A4" == "1" ) ]]; then
    "$PYTHON" -u tracking/train.py --script tbsi_track --config "$config" --save_dir ./output --mode single \
      >"$log_dir/train.stdout.log" 2>&1
  fi
  env TBSI_INFER_FP16=0 "$PYTHON" -u tracking/test.py tbsi_track "$config" \
    --dataset_name rpp_lasher_test --threads 0 --num_gpus 1 >"$log_dir/test.stdout.log" 2>&1
  export TBSI_RPP_TEST_LIST="$PROXY_ROOT/proxy_test_sequences.txt"
  "$PYTHON" -u tracking/analysis_results.py --tracker_name tbsi_track --tracker_param "$config" \
    --dataset_name rpp_lasher_test >"$log_dir/analysis.stdout.log" 2>&1
  printf '%s\n' 'status=completed' >> "$log_dir/manifest.txt"
  if [[ -f "$checkpoint" ]]; then
    sha256sum "$checkpoint" >> "$run_root/released_checkpoint.sha256"
    stat -c 'size=%s path=%n' "$checkpoint" >> "$run_root/released_checkpoint.txt"
    unlink "$checkpoint"
    echo "[$(date -Is)] RELEASED $stage proxy checkpoint" | tee -a "$CAMPAIGN/pipeline.log"
  fi
}

if [[ "$SKIP_A3" != "1" ]]; then
  run_proxy a3 v1.4.0-tcmr-a3-rpp-15ep
fi
run_proxy a4 v1.4.1-tcmr-a4-rpp-15ep

FULL_CONFIG=v1.4.1-tcmr-a4-full-15ep
FULL_RUN="$ROOT/output/experiments/$FULL_CONFIG"
[[ ! -e "$FULL_RUN" ]] || die "A4 full output exists: $FULL_RUN"
mkdir -p "$FULL_RUN/logs"
printf '%s\n' \
  'stage=A4_full' "config=$FULL_CONFIG" 'parent=v1.3.3-tsms-v3-full-15ep' \
  'train=LasHeR_train' 'test=lasher_test' 'test_threads=0' 'infer_fp16=0' 'status=running' \
  > "$FULL_RUN/logs/manifest.txt"
echo "[$(date -Is)] START a4_full_train" | tee -a "$CAMPAIGN/pipeline.log"
"$PYTHON" -u tracking/train.py --script tbsi_track --config "$FULL_CONFIG" --save_dir ./output --mode single \
  >"$FULL_RUN/logs/train.stdout.log" 2>&1
echo "[$(date -Is)] DONE a4_full_train" | tee -a "$CAMPAIGN/pipeline.log"
env TBSI_INFER_FP16=0 "$PYTHON" -u tracking/test.py tbsi_track "$FULL_CONFIG" \
  --dataset_name lasher_test --threads 0 --num_gpus 1 >"$FULL_RUN/logs/test.stdout.log" 2>&1
echo "[$(date -Is)] DONE a4_full_test" | tee -a "$CAMPAIGN/pipeline.log"
"$PYTHON" -u tracking/analysis_results.py --tracker_name tbsi_track --tracker_param "$FULL_CONFIG" \
  --dataset_name lasher_test >"$FULL_RUN/logs/analysis.stdout.log" 2>&1
sed -i 's/^status=running/status=completed/' "$FULL_RUN/logs/manifest.txt"
printf '%s\n' 'status=completed' >> "$CAMPAIGN/manifest.txt"
echo "[$(date -Is)] PIPELINE COMPLETE" | tee -a "$CAMPAIGN/pipeline.log"
