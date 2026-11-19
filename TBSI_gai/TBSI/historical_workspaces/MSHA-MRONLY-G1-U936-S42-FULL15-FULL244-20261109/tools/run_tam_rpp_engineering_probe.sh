#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="/root/autodl-tmp/TBSI_gai/TBSI"
PYTHON_BIN="/root/autodl-tmp/conda_envs/tbsi/bin/python"
MANIFEST_DIR="$ROOT_DIR/experiments/proxy/lasher_rpp_v1"
BASELINE="$ROOT_DIR/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
M1="$ROOT_DIR/output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar"
TAM_ONLY="$ROOT_DIR/output/experiments/tam_online_replay_train_audit_v1_20261007/tam_image_replay_smoke_checkpoint.pth"
OUT="$ROOT_DIR/output/experiments/tam_rpp_engineering_probe_v1_20261007"
MERGED="$OUT/TBSITrack_M1_ep0015_TAM100.pth.tar"
THREADS=6
mkdir -p "$OUT"
exec 9>"$OUT/.lock"; flock -n 9 || { echo 'duplicate TAM engineering probe' >&2; exit 75; }
cd "$ROOT_DIR"
test -x "$PYTHON_BIN"; test -s "$M1"; test -s "$TAM_ONLY"
test -s "$MANIFEST_DIR/proxy_test_sequences.txt"; test -s "$MANIFEST_DIR/proxy_test_manifest.json"
git diff --quiet -- "$BASELINE" || { echo 'protected baseline yaml changed' >&2; exit 4; }
FREE_KB="$(df -Pk "$ROOT_DIR" | awk 'NR==2 {print $4}')"
(( FREE_KB > 5 * 1024 * 1024 )) || { echo 'less than 5 GiB free; refusing checkpoint copy/test' >&2; exit 6; }
RUNNING_TBSI="$(ps -eo pid=,comm=,args= | awk '$2 ~ /^python([0-9.]*)?$/ && $0 ~ /tracking\/(train|test)\.py|lib\/train\/run_training\.py/')"
if [[ -n "$RUNNING_TBSI" ]]; then
  echo "$RUNNING_TBSI" >&2
  echo 'another TBSI train/test process is running' >&2; exit 7
fi
for cfg in TAM_RPP_ENGINEERING_C0_seed42 TAM_RPP_ENGINEERING_TAM100_seed42; do
  test -f "$ROOT_DIR/experiments/tbsi_track/${cfg}.yaml"
  test ! -e "$ROOT_DIR/output/test/tracking_results/tbsi_track/${cfg}"
done
mkdir -p "$ROOT_DIR/output/.locks"
exec 8>"$ROOT_DIR/output/.locks/tam_rpp_engineering_probe.lock"
flock -n 8 || { echo 'another TAM RPP engineering probe holds lock' >&2; exit 7; }
exec > >(tee "$OUT/probe.log") 2>&1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TBSI_RPP_TEST_LIST="$MANIFEST_DIR/proxy_test_sequences.txt"
export TBSI_INFER_FP16=0
echo "TAM_RPP_ENGINEERING_PROBE_START $(date -u +%FT%TZ)"
echo "RPP_V1_STATUS exploratory-only; historical admission not established"
echo "TEST_SEQUENCE_COUNT $(wc -l < "$TBSI_RPP_TEST_LIST") THREADS $THREADS"
echo "M1_SHA256 $(sha256sum "$M1" | awk '{print $1}')"
echo "TAM_ONLY_SHA256 $(sha256sum "$TAM_ONLY" | awk '{print $1}')"
"$PYTHON_BIN" tools/merge_tam_adapter_checkpoint.py --parent "$M1" --tam-checkpoint "$TAM_ONLY" --output "$MERGED"

export TBSI_CHECKPOINT_OVERRIDE="$M1"
"$PYTHON_BIN" -u tracking/test.py tbsi_track TAM_RPP_ENGINEERING_C0_seed42 \
  --dataset_name rpp_lasher_test --threads "$THREADS" --num_gpus 1
"$PYTHON_BIN" -u tracking/analysis_results.py --tracker_name tbsi_track \
  --tracker_param TAM_RPP_ENGINEERING_C0_seed42 --dataset_name rpp_lasher_test

export TBSI_CHECKPOINT_OVERRIDE="$MERGED"
"$PYTHON_BIN" -u tracking/test.py tbsi_track TAM_RPP_ENGINEERING_TAM100_seed42 \
  --dataset_name rpp_lasher_test --threads "$THREADS" --num_gpus 1
"$PYTHON_BIN" -u tracking/analysis_results.py --tracker_name tbsi_track \
  --tracker_param TAM_RPP_ENGINEERING_TAM100_seed42 --dataset_name rpp_lasher_test
echo "TAM_RPP_ENGINEERING_PROBE_FINISH $(date -u +%FT%TZ)"
