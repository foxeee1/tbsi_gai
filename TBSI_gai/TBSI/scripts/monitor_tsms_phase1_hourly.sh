#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/output/experiments/v1.3.1-tsms-phase1-serial/logs"
LOG_FILE="$LOG_DIR/hourly_monitor.log"
LOCK_FILE="/tmp/tbsi_tsms_phase1_hourly.lock"
MIN_FREE_GB="${TSMS_MIN_FREE_GB:-8}"
INTERVAL_SEC="${TSMS_CHECK_INTERVAL_SEC:-3600}"
CANONICAL="$ROOT/experiments/tbsi_track/v1.0.0-单卡3090-48G-等效全局128基线.yaml"
PREFLIGHT_DIR="$ROOT/output/experiments/v1.3.1-tsms-full-3ep"
FULL_DIR="$ROOT/output/experiments/v1.3.1-tsms-full-15ep"

mkdir -p "$LOG_DIR"
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "[$(date -Is)] another TSMS hourly monitor is active" >>"$LOG_FILE"; exit 2; }

log() { echo "[$(date -Is)] $*" | tee -a "$LOG_FILE"; }

check_and_start() {
  local free_gb
  free_gb="$(df -Pk "$ROOT" | awk 'NR==2 {printf "%d", $4/1024/1024}')"
  log "check disk_free=${free_gb}GiB threshold=${MIN_FREE_GB}GiB"

  [[ -z "$(git -C "$ROOT" diff -- "$CANONICAL")" ]] || { log "STOP canonical baseline YAML is modified"; return 10; }
  if pgrep -af 'tracking/(train|test|analysis_results)\.py' | grep -v grep >/dev/null; then
    log "wait another TBSI process is active"
    return 1
  fi
  if [[ -e "$PREFLIGHT_DIR/checkpoints" || -e "$FULL_DIR/checkpoints" ]]; then
    log "STOP TSMS output checkpoint directory already exists; refusing overwrite"
    return 11
  fi
  (( free_gb >= MIN_FREE_GB )) || { log "wait insufficient disk"; return 1; }

  log "START scripts/run_tsms_phase1_serial.sh --run"
  cd "$ROOT"
  bash scripts/run_tsms_phase1_serial.sh --run >>"$LOG_FILE" 2>&1
  log "TSMS serial phase 1 finished"
  return 0
}

while true; do
  set +e
  check_and_start
  rc=$?
  set -e
  case "$rc" in
    0) exit 0 ;;
    10|11) exit "$rc" ;;
  esac
  log "next check in ${INTERVAL_SEC}s"
  sleep "$INTERVAL_SEC"
done
