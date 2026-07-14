#!/usr/bin/env bash
#===============================================================================
# TBSI 训练进度监控器 — 每10分钟显示进度条 + GPU 利用率 + ETA
#
# 基于 指南/复现.txt ⑤-H 效果验证：
#   watch -n 1 nvidia-smi     → 看 GPU-Util 列
#   pynvml GPU 利用率记录      → 训练日志中自动记录
#
# 用法:
#   ./monitor_train.sh                                           # 默认 config
#   ./monitor_train.sh vitb_256_tbsi_32x1_1e4_lasher_15ep_sot   # 指定 config
#   ./monitor_train.sh --interval 300                            # 自定义间隔(秒)
#   ./monitor_train.sh --tail                                    # 显示日志末尾
#   ./monitor_train.sh --no-gpu                                  # 不显示GPU信息
#===============================================================================

set -euo pipefail

CONFIG="${1:-vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_reproduce}"
SCRIPT="tbsi_track"
SAVE_DIR="./output"
LOG_DIR="${SAVE_DIR}/logs"
TRAIN_LOG="${LOG_DIR}/${SCRIPT}-${CONFIG}.log"
CKPT_DIR="${SAVE_DIR}/checkpoints/train/${SCRIPT}/${CONFIG}"
INTERVAL=600
TAIL_MODE=0
SHOW_GPU=1

# Parse args (skip first if it's a config name)
FIRST_ARG="${1:-}"
case "${FIRST_ARG}" in
    --*) ;;  # first arg is an option, no config name given
    *)   CONFIG="${FIRST_ARG}"; shift ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --interval) INTERVAL="$2"; shift 2 ;;
        --tail)     TAIL_MODE=1; shift ;;
        --no-gpu)   SHOW_GPU=0; shift ;;
        --)         shift; break ;;
        *)          shift ;;
    esac
done

# ── 检测总 epoch ──
detect_total_epochs() {
    local cfg_file="experiments/${SCRIPT}/${CONFIG}.yaml"
    if [ -f "${cfg_file}" ]; then
        grep -oP '^\s*EPOCH:\s*\K\d+' "${cfg_file}" | head -1 || echo "15"
    else
        echo "15"
    fi
}
TOTAL_EPOCHS=$(detect_total_epochs)

# ── GPU 利用率采集（后台进程） ──
GPU_LOG="/tmp/tbsi_gpu_mon_$$.csv"

start_gpu_collect() {
    if [ "${SHOW_GPU}" -ne 1 ]; then return; fi
    (
        while kill -0 "${PPID}" 2>/dev/null; do
            echo "$(date +%s),$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)"
            sleep 10
        done
    ) &
    GPU_COLLECT_PID=$!
}

stop_gpu_collect() {
    [ -n "${GPU_COLLECT_PID:-}" ] && kill "${GPU_COLLECT_PID}" 2>/dev/null || true
}

get_gpu_now() {
    [ "${SHOW_GPU}" -ne 1 ] && return
    local stats
    stats=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "${stats}" | awk -F',' '{printf "GPU:%s%% MEM:%s%% %s°C %sW", $1, $2, $3, $4}'
}

get_gpu_avg() {
    [ "${SHOW_GPU}" -ne 1 ] && return
    [ ! -f "${GPU_LOG}" ] && { get_gpu_now; return; }
    local lines
    lines=$(wc -l < "${GPU_LOG}")
    [ "${lines}" -le 1 ] && { get_gpu_now; return; }
    local avg_gpu avg_mem max_temp
    avg_gpu=$(awk -F',' 'NR>1 {s+=$2; c++} END {printf "%.0f", s/c}' "${GPU_LOG}")
    avg_mem=$(awk -F',' 'NR>1 {s+=$3; c++} END {printf "%.0f", s/c}' "${GPU_LOG}")
    max_temp=$(awk -F',' 'NR>1 {if(max<$4) max=$4} END {printf "%.0f", max}' "${GPU_LOG}")
    echo "GPU:${avg_gpu}% MEM:${avg_mem}% MAX:${max_temp}°C (avg)"
}

# ── 界面 ──
draw_header() {
    clear 2>/dev/null || true
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║           TBSI 训练进度监控                                 ║"
    printf "║  Config: %-45s║\n" "${CONFIG}"
    printf "║  Refresh: 每 ${INTERVAL}s"
    printf "%$((43 - ${#INTERVAL}))s║\n" ""
    echo "║  Total: ${TOTAL_EPOCHS} epochs"
    echo "║  Log:  ${TRAIN_LOG}"
    echo "╚══════════════════════════════════════════════════════════════╝"
}

show_progress() {
    local start_time="$1"
    local now
    now=$(date +%s)

    # Parse training log
    local epoch="" iter="" total_iter="" fps="" line="" loss_str="" gpu_str=""
    local data_time="" forward_time="" total_time="" lr=""

    line=$(grep -oP '\[train: \d+, \d+ / \d+\].*' "${TRAIN_LOG}" 2>/dev/null | tail -1)

    if [ -n "${line}" ]; then
        epoch=$(echo "${line}" | grep -oP '\[train: \K\d+')
        local it
        it=$(echo "${line}" | grep -oP '\[train: \d+, \K[\d /]+\]' | tr -d '[]')
        iter=$(echo "${it}" | cut -d'/' -f1 | tr -d ' ')
        total_iter=$(echo "${it}" | cut -d'/' -f2 | tr -d ' ')
        fps=$(echo "${line}" | grep -oP 'FPS: \K[0-9.]+')

        local g l f
        g=$(echo "${line}" | grep -oP 'giou: \K[0-9.]+' | head -1)
        l=$(echo "${line}" | grep -oP 'l1: \K[0-9.]+' | head -1)
        f=$(echo "${line}" | grep -oP 'focal: \K[0-9.]+' | head -1)
        [ -n "${g}" ] && loss_str+="giou=${g}  "
        [ -n "${l}" ] && loss_str+="l1=${l}  "
        [ -n "${f}" ] && loss_str+="focal=${f}"

        data_time=$(echo "${line}" | grep -oP 'DataTime: \K[0-9.]+')
        forward_time=$(echo "${line}" | grep -oP 'ForwardTime: \K[0-9.]+')
        total_time=$(echo "${line}" | grep -oP 'TotalTime: \K[0-9.]+')
        lr=$(echo "${line}" | grep -oP 'LearningRate/group0: \K[0-9.e-]+')

        # Also parse GPU_UTIL from log if present (from pynvml instrumentation)
        local gpu_log_line
        gpu_log_line=$(grep -oP 'GPU_UTIL: [0-9]+%.*' "${TRAIN_LOG}" 2>/dev/null | tail -1)
        if [ -n "${gpu_log_line}" ]; then
            gpu_str=$(echo "${gpu_log_line}" | grep -oP 'GPU_UTIL: [0-9]+% \| MEM_UTIL: [0-9]+%')
        fi
    fi

    # fallback
    if [ -z "${epoch}" ]; then
        epoch=$(grep -oP 'EPOCH: \K\d+' "${TRAIN_LOG}" 2>/dev/null | tail -1)
    fi

    local pct=0
    if [ -n "${epoch}" ] && [ "${TOTAL_EPOCHS}" -gt 0 ] 2>/dev/null; then
        pct=$(( epoch * 100 / TOTAL_EPOCHS ))
    fi

    # Progress bar
    local bar_len=40
    local filled=$(( pct * bar_len / 100 ))
    local empty=$(( bar_len - filled ))
    local bar="["
    for ((i=0; i<filled; i++)); do bar+="█"; done
    for ((i=0; i<empty; i++)); do bar+="░"; done
    bar+="]"

    # ETA
    local elapsed=$(( now - start_time ))
    local eta_str="--"
    if [ "${pct}" -gt 0 ] && [ "${pct}" -lt 100 ] 2>/dev/null; then
        local eta_sec=$(( elapsed * 100 / pct - elapsed ))
        eta_str=$(printf "%dh%dm" $(( eta_sec / 3600 )) $(( (eta_sec % 3600) / 60 )))
    fi
    local elapsed_fmt
    elapsed_fmt=$(printf "%dh%dm" $(( elapsed / 3600 )) $(( (elapsed % 3600) / 60 )))

    # GPU
    if [ "${SHOW_GPU}" -eq 1 ]; then
        [ -z "${gpu_str}" ] && gpu_str=$(get_gpu_avg)
    fi

    # Output
    echo ""
    [ -n "${gpu_str}" ] && echo "  ${gpu_str}"
    echo "  ⏱  ${elapsed_fmt} elapsed  |  ETA ${eta_str}"
    echo ""
    echo "  Epoch ${epoch:-?}/${TOTAL_EPOCHS}  ${bar}  ${pct}%"

    if [ -n "${iter}" ] && [ -n "${total_iter}" ] && [ "${total_iter}" -gt 0 ] 2>/dev/null; then
        local iter_pct=$(( iter * 100 / total_iter ))
        local ib=20 ifilled=$(( iter * ib / total_iter ))
        printf "  Iter  "
        for ((i=0; i<ifilled; i++)); do printf "▓"; done
        for ((i=0; i<ib-ifilled; i++)); do printf "▒"; done
        printf "  %s/%s  FPS:%s\n" "${iter}" "${total_iter}" "${fps:-N/A}"
    fi

    if [ -n "${lr}" ]; then
        echo "  LR: ${lr}"
    fi

    if [ -n "${loss_str}" ]; then
        echo "  📉 ${loss_str}"
    fi

    if [ -n "${data_time}" ] && [ -n "${total_time}" ]; then
        local data_pct
        data_pct=$(echo "scale=1; ${data_time} / ${total_time} * 100" | bc -l 2>/dev/null || echo "?")
        echo "  ⏳ Data:${data_time}s  Fwd:${forward_time}s  Total:${total_time}s  (Data=${data_pct}%)"
    fi

    # Checkpoints
    if [ -d "${CKPT_DIR}" ]; then
        local ckpt_count
        ckpt_count=$(ls "${CKPT_DIR}"/*.pth.tar 2>/dev/null | wc -l)
        if [ "${ckpt_count}" -gt 0 ]; then
            local latest_ckpt latest_epoch ckpt_size
            latest_ckpt=$(ls -t "${CKPT_DIR}"/*.pth.tar 2>/dev/null | head -1)
            latest_epoch=$(basename "${latest_ckpt}" | grep -oP 'ep\K\d+')
            ckpt_size=$(du -h "${latest_ckpt}" 2>/dev/null | cut -f1)
            echo ""
            echo "  💾 Checkpoints: ${ckpt_count} files"
            echo "     Latest: ep${latest_epoch} (${ckpt_size})"
        fi
    fi
    echo ""
}

# ── 检查进程 ──
check_alive() {
    local pid
    pid=$(pgrep -f "train.*${CONFIG}" 2>/dev/null | head -1 || true)
    [ -n "${pid}" ] && return 0
    [ -f "/tmp/tbsi_train_${CONFIG}.pid" ] && kill -0 "$(cat "/tmp/tbsi_train_${CONFIG}.pid")" 2>/dev/null && return 0
    return 1
}

# ═══════════════════════════════
# Main Loop
# ═══════════════════════════════
START_TIME=$(date +%s)
FIRST=1

start_gpu_collect

draw_header
echo "   等待训练启动..."
echo "   日志: ${TRAIN_LOG}"
echo ""

while true; do
    if ! check_alive; then
        if [ -f "${TRAIN_LOG}" ] && grep -q "Finished training" "${TRAIN_LOG}" 2>/dev/null; then
            draw_header
            echo ""
            echo "   🎉 训练已完成！"
            echo ""
            show_progress "${START_TIME}"
            echo "   日志: ${TRAIN_LOG}"
            stop_gpu_collect
            [ -f "${GPU_LOG}" ] && rm -f "${GPU_LOG}"
            exit 0
        fi

        if [ ${FIRST} -eq 1 ]; then
            sleep 5; FIRST=0; continue
        fi

        draw_header
        echo ""
        echo "   ⚠️  未检测到训练进程 (${CONFIG})"
        echo "   请先运行: ./run_fast_pipeline.sh"
        echo "   或检查 config 名称是否正确"
        if [ -f "${TRAIN_LOG}" ]; then
            echo ""
            echo "   日志最后 3 行:"
            tail -3 "${TRAIN_LOG}" 2>/dev/null | sed 's/^/   /'
        fi
        echo ""
        echo "   等待中 (每 ${INTERVAL}s 重试) ..."
        sleep "${INTERVAL}"
        continue
    fi

    FIRST=0
    draw_header
    show_progress "${START_TIME}"
    if [ "${TAIL_MODE}" -eq 1 ] && [ -f "${TRAIN_LOG}" ]; then
        echo "  ── 日志末尾 ──"
        tail -3 "${TRAIN_LOG}" 2>/dev/null | sed 's/^/  /'
    fi
    echo ""
    echo "   🔄 下次刷新: $(date -d "+${INTERVAL} seconds" '+%H:%M:%S')   Ctrl+C 退出（训练继续）"
    sleep "${INTERVAL}"
done
