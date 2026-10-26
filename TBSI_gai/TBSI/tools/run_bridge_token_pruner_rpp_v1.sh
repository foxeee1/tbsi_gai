#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
BASELINE_YAML="${ROOT_DIR}/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
MANIFEST_DIR="${ROOT_DIR}/experiments/proxy/lasher_rpp_v1"
SEED="${BTP_SEED:-42}"
CONFIG="BTP_RPP_v1_b16_seed${SEED}"
MIN_FREE_GB="${BTP_MIN_FREE_GB:-8}"
TEST_THREADS=6

case "$SEED" in 42|3407) ;; *) echo "BTP_SEED must be 42 or 3407" >&2; exit 2 ;; esac
[[ -x "$PYTHON_BIN" ]] || { echo "missing Python: $PYTHON_BIN" >&2; exit 3; }
[[ -f "$BASELINE_YAML" ]] || { echo "missing canonical baseline YAML" >&2; exit 3; }
[[ -z "$(git -C "$ROOT_DIR" diff -- "$BASELINE_YAML")" ]] || { echo "canonical baseline YAML has a diff" >&2; exit 4; }
for f in proxy_train_sequences.txt proxy_train_manifest.json proxy_test_sequences.txt proxy_test_manifest.json; do
  [[ -s "${MANIFEST_DIR}/${f}" ]] || { echo "missing manifest: ${MANIFEST_DIR}/${f}" >&2; exit 5; }
done
[[ -f "${ROOT_DIR}/experiments/tbsi_track/${CONFIG}.yaml" ]] || { echo "missing config" >&2; exit 5; }
if pgrep -af '[t]racking/(train|test)\.py|[l]ib/train/run_training\.py' >/tmp/tbsi_btp_processes.$$ 2>/dev/null; then
  cat /tmp/tbsi_btp_processes.$$ >&2
  rm -f /tmp/tbsi_btp_processes.$$; exit 7
fi
rm -f /tmp/tbsi_btp_processes.$$
FREE_KB="$(df -Pk "$ROOT_DIR" | awk 'NR==2 {print $4}')"
(( FREE_KB >= MIN_FREE_GB * 1024 * 1024 )) || { echo "insufficient free disk: ${FREE_KB} KiB" >&2; exit 6; }

LOCK_DIR="${ROOT_DIR}/output/.locks"
mkdir -p "$LOCK_DIR"
exec 9>"${LOCK_DIR}/bridge_token_pruner_rpp_v1.lock"
flock -n 9 || { echo "another BTP campaign holds the lock" >&2; exit 7; }

TRAIN_LIST="${MANIFEST_DIR}/proxy_train_sequences.txt"
TEST_LIST="${MANIFEST_DIR}/proxy_test_sequences.txt"
RUN_DIR="${ROOT_DIR}/output/experiments/${CONFIG}"
RESULT_DIR="${ROOT_DIR}/output/test/tracking_results/tbsi_track/${CONFIG}"
CHECKPOINT="${RUN_DIR}/checkpoints/TBSITrack_ep0015.pth.tar"
REPORT="${RUN_DIR}/logs/${CONFIG}_run_report.md"
[[ ! -e "$RUN_DIR" ]] || { echo "refusing to reuse run directory: $RUN_DIR" >&2; exit 8; }
[[ ! -e "$RESULT_DIR" ]] || { echo "refusing to reuse result directory: $RESULT_DIR" >&2; exit 8; }
mkdir -p "${RUN_DIR}/logs"
cp "${ROOT_DIR}/experiments/tbsi_track/${CONFIG}.yaml" "${RUN_DIR}/config.snapshot.yaml"
{
  echo "# ${CONFIG} run report"
  echo "- start_utc: $(date -u +%FT%TZ)"
  echo "- git: $(git -C "$ROOT_DIR" rev-parse HEAD)"
  echo "- baseline_yaml_sha256: $(sha256sum "$BASELINE_YAML" | awk '{print $1}')"
  echo "- train_manifest_sha256: $(sha256sum "${MANIFEST_DIR}/proxy_train_manifest.json" | awk '{print $1}')"
  echo "- test_manifest_sha256: $(sha256sum "${MANIFEST_DIR}/proxy_test_manifest.json" | awk '{print $1}')"
  echo "- seed: ${SEED}"
  echo "- rolling_checkpoint_policy: keep newest completed checkpoint only"
  echo "- final_artifact_policy: final checkpoint, metrics, config snapshot, provenance"
  echo "- gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
  echo "- python: $($PYTHON_BIN --version 2>&1)"
  echo "- torch: $($PYTHON_BIN -c 'import torch; print(torch.__version__, torch.version.cuda)')"
  echo ""
  echo '## Live command output'
} > "$REPORT"

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TBSI_RPP_TRAIN_LIST="$TRAIN_LIST" TBSI_RPP_TEST_LIST="$TEST_LIST" TBSI_INFER_FP16=0
cd "$ROOT_DIR"
{
  echo "[TRAIN] $(date -u +%FT%TZ)"
  "$PYTHON_BIN" -u lib/train/run_training.py --script tbsi_track --config "$CONFIG" \
    --save_dir ./output --seed "$SEED" --local_rank -1 --use_lmdb 0 \
    --config_prv baseline --use_wandb 0
  echo "[TRAIN_DONE] $(date -u +%FT%TZ)"
  [[ -s "$CHECKPOINT" ]] || { echo "final checkpoint missing"; exit 9; }
  echo "[TEST] $(date -u +%FT%TZ)"
  "$PYTHON_BIN" -u tracking/test.py tbsi_track "$CONFIG" --dataset_name rpp_lasher_test \
    --threads "$TEST_THREADS" --num_gpus 1
  echo "[TEST_DONE] $(date -u +%FT%TZ)"
  "$PYTHON_BIN" -u tracking/analysis_results.py --tracker_name tbsi_track \
    --tracker_param "$CONFIG" --dataset_name rpp_lasher_test
  echo "[ANALYSIS_DONE] $(date -u +%FT%TZ)"
} 2>&1 | tee -a "$REPORT"
