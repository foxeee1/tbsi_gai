#!/usr/bin/env bash
set -u
PID="$1"
OUT="$2"
mkdir -p "$(dirname "$OUT")"
while kill -0 "$PID" 2>/dev/null; do
  {
    date -u +%FT%TZ
    echo "pid=$PID"
    ps -o pid,ppid,stat,etime,%cpu,%mem,cmd -p "$PID" || true
    pgrep -af "M1_CENTER_RANK_LOSS_FULL15_W1_seed42_20261016" || true
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader || true
    df -h /root/autodl-tmp | tail -1
    tail -1 /root/autodl-tmp/TBSI_gai/TBSI/output/experiments/M1_CENTER_RANK_LOSS_FULL15_W1_seed42_20261016/logs/training.log 2>/dev/null || true
    echo "---"
  } >> "$OUT"
  sleep 3600
done
{
  date -u +%FT%TZ
  echo "training_pid=$PID exited"
} >> "$OUT"
