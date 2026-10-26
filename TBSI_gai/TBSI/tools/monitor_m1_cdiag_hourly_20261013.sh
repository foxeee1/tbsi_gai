#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARENT_PID="${1:?parent pid required}"
OUT="${2:?output directory required}"
LOG="$OUT/hourly_monitor.log"

while kill -0 "$PARENT_PID" 2>/dev/null; do
    {
        date -u '+timestamp=%Y-%m-%dT%H:%M:%SZ'
        ps -o pid,ppid,stat,etime,%cpu,%mem,cmd -p "$PARENT_PID" || true
        tail -1 "$OUT/train.stdout.log" 2>/dev/null || true
        find "$OUT" -name 'TBSITrack_ep*.pth.tar' -printf 'checkpoint=%p\n' 2>/dev/null | sort | tail -3
        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
        printf '\n'
    } >> "$LOG"
    sleep 3600
done

{
    date -u '+timestamp=%Y-%m-%dT%H:%M:%SZ'
    echo 'parent_process=EXITED'
    tail -20 "$OUT/train.stdout.log" 2>/dev/null || true
    find "$OUT" -name 'TBSITrack_ep*.pth.tar' -o -name 'official_metric_and_persistence.json' 2>/dev/null | sort
} >> "$LOG"
