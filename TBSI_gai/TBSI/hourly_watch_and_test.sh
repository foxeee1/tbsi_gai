#!/usr/bin/env bash
#===============================================================================
# TBSI 每小时监控 + 训练结束后自动全量测试
#
# 设计为被 cron 每小时调用：
#   1. 训练中 → 打印进度
#   2. 训练完成 → 自动启动测试 + 分析
#
# 用法:
#   ./hourly_watch_and_test.sh                    # 单次检查
#   ./hourly_watch_and_test.sh --force-test       # 跳过检查，强制测试
#===============================================================================

set -euo pipefail

CONFIG="${CONFIG:-vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_fast}"
SCRIPT="${SCRIPT:-tbsi_track}"
SAVE_DIR="${SAVE_DIR:-./output}"
DATASET="${DATASET:-lasher_test}"
CONDA_ENV="${CONDA_ENV:-tbsi}"

LOG_DIR="${SAVE_DIR}/logs"
TRAIN_LOG="${LOG_DIR}/${SCRIPT}-${CONFIG}.log"
CKPT_DIR="${SAVE_DIR}/checkpoints/train/${SCRIPT}/${CONFIG}"
TEST_LOG="${LOG_DIR}/test_${CONFIG}_$(date +%Y%m%d_%H%M%S).log"
ANALYSIS_LOG="${LOG_DIR}/analysis_${CONFIG}_$(date +%Y%m%d_%H%M%S).log"
TRAINING_FINISHED_MARKER="/tmp/tbsi_${CONFIG}_done.flag"

FORCE_TEST=0
for arg in "$@"; do
    [ "$arg" = "--force-test" ] && FORCE_TEST=1
done

# ── 环境 ──
setup_env() {
    unset OMP_NUM_THREADS
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export PYTHONUNBUFFERED=1
    if [ -f "/root/autodl-tmp/conda_envs/${CONDA_ENV}/etc/profile.d/conda.sh" ]; then
        source "/root/autodl-tmp/conda_envs/${CONDA_ENV}/etc/profile.d/conda.sh"
    fi
    conda activate "${CONDA_ENV}" 2>/dev/null || true
}

# ── 训练是否还在运行？ ──
training_alive() {
    pgrep -f "run_training.*${CONFIG}" >/dev/null 2>&1 && return 0
    pgrep -f "tracking/train.*${CONFIG}" >/dev/null 2>&1 && return 0
    return 1
}

# ── 训练是否已完成？ ──
training_finished() {
    [ -f "${TRAINING_FINISHED_MARKER}" ] && return 0
    if [ -f "${TRAIN_LOG}" ] && grep -q "Finished training" "${TRAIN_LOG}" 2>/dev/null; then
        touch "${TRAINING_FINISHED_MARKER}"
        return 0
    fi
    # 也检查 epoch 15 的完成（日志末尾有 Epoch Time 但没有 Finished training 的情况）
    if [ -f "${TRAIN_LOG}" ]; then
        last_epoch=$(grep -oP '\[train: \K\d+' "${TRAIN_LOG}" 2>/dev/null | tail -1)
        if [ -n "${last_epoch}" ] && [ "${last_epoch}" -ge 15 ] 2>/dev/null; then
            # 确认最后一行的 iter 也到了 1875
            local last_line
            last_line=$(grep -oP '\[train: 15, \K\d+' "${TRAIN_LOG}" 2>/dev/null | tail -1)
            if [ -n "${last_line}" ] && [ "${last_line}" -ge 1875 ] 2>/dev/null; then
                touch "${TRAINING_FINISHED_MARKER}"
                return 0
            fi
        fi
    fi
    return 1
}

# ── 报告训练进度 ──
report_progress() {
    if ! training_alive && ! training_finished; then
        echo "[$(date '+%H:%M:%S')] ⏸️  训练进程未运行，日志中未发现完成标记"
        echo "   启动训练的方式: cd /root/autodl-tmp/TBSI_gai/TBSI && python tracking/train.py --script tbsi_track --config ${CONFIG} --save_dir ${SAVE_DIR} --mode single"
        return
    fi

    local line
    line=$(grep -oP '\[train: \d+, \d+ / \d+\].*' "${TRAIN_LOG}" 2>/dev/null | tail -1)

    if [ -z "${line}" ]; then
        echo "[$(date '+%H:%M:%S')] ⌛ 训练启动中，等待第一个输出..."
        return
    fi

    local epoch iter total_iter fps ft dt tt loss iou
    epoch=$(echo "${line}" | grep -oP '\[train: \K\d+')
    local it
    it=$(echo "${line}" | grep -oP '\[train: \d+, \K[\d /]+\]' | tr -d '[]')
    iter=$(echo "${it}" | cut -d'/' -f1 | tr -d ' ')
    total_iter=$(echo "${it}" | cut -d'/' -f2 | tr -d ' ')
    fps=$(echo "${line}" | grep -oP 'FPS: \K[0-9.]+')
    ft=$(echo "${line}" | grep -oP 'ForwardTime: \K[0-9.]+')
    dt=$(echo "${line}" | grep -oP 'DataTime: \K[0-9.]+')
    tt=$(echo "${line}" | grep -oP 'TotalTime: \K[0-9.]+')
    loss=$(echo "${line}" | grep -oP 'Loss/total: \K[0-9.]+')
    iou=$(echo "${line}" | grep -oP 'IoU: \K[0-9.]+')

    # GPU info
    local gpu_util gpu_mem gpu_temp
    IFS=',' read -r gpu_util gpu_mem gpu_temp <<< "$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)"

    # Process running time
    local proc_time="?"
    local pid
    pid=$(pgrep -f "run_training.*${CONFIG}" 2>/dev/null | head -1)
    [ -n "${pid}" ] && proc_time=$(ps -p "${pid}" -o etime --no-headers 2>/dev/null | tr -d ' ')

    # Speed status
    local speed_tag=""
    if [ -n "${dt}" ] && [ -n "${tt}" ] && [ "${dt}" != "0" ] 2>/dev/null; then
        local dt_val
        dt_val=$(echo "${dt}" | bc 2>/dev/null || echo "0")
        if [ "$(echo "${dt_val} < 0.05" | bc 2>/dev/null)" = "1" ]; then
            speed_tag="⚡ CACHED"
        elif [ "$(echo "${dt_val} < 0.5" | bc 2>/dev/null)" = "1" ]; then
            speed_tag="🔥 WARMING"
        else
            speed_tag="💾 DISK_IO"
        fi
    fi

    # Epoch progress bar
    local pct=0
    [ -n "${epoch}" ] && [ "${epoch}" -gt 0 ] 2>/dev/null && pct=$(( epoch * 100 / 15 ))
    local bar_len=30
    local filled=$(( pct * bar_len / 100 ))
    local ep_bar="["
    for ((i=0; i<filled; i++)); do ep_bar+="█"; done
    for ((i=0; i<bar_len-filled; i++)); do ep_bar+="░"; done
    ep_bar+="]"

    # Iter progress bar
    local iter_pct=0
    [ -n "${iter}" ] && [ -n "${total_iter}" ] && [ "${total_iter}" -gt 0 ] 2>/dev/null && iter_pct=$(( iter * 100 / total_iter ))
    local ib=20 ifilled=$(( iter_pct * 20 / 100 ))
    local iter_bar="["
    for ((i=0; i<ifilled; i++)); do iter_bar+="▓"; done
    for ((i=0; i<20-ifilled; i++)); do iter_bar+="▒"; done
    iter_bar+="]"

    echo ""
    echo "  ╔══════════════════════════════════════════════════════════╗"
    echo "  ║  TBSI 训练报告  ${speed_tag}"
    echo "  ║  $(date '+%Y-%m-%d %H:%M:%S')  |  运行 ${proc_time}"
    echo "  ╠══════════════════════════════════════════════════════════╣"
    echo "  ║  Epoch ${epoch}/15  ${ep_bar}  ${pct}%"
    echo "  ║  Iter  ${iter_bar}  ${iter}/${total_iter}"
    echo "  ║"
    echo "  ║  FPS: ${fps:-?}  |  IoU: ${iou:-?}  |  Loss: ${loss:-?}"
    echo "  ║  Forward: ${ft:-?}s  |  Data: ${dt:-?}s  |  Total: ${tt:-?}s"
    echo "  ║"
    echo "  ║  GPU: ${gpu_util:-?}%  |  MEM: ${gpu_mem:-?} MiB  |  TEMP: ${gpu_temp:-?}°C"
    echo "  ╚══════════════════════════════════════════════════════════╝"
    echo ""

    # 如果 DataTime < 0.05s，提醒用户
    if [ -n "${dt}" ] && [ "$(echo "${dt} < 0.05" | bc 2>/dev/null)" = "1" ]; then
        echo "  ⚡ 数据已完全缓存在内存中！速度达到最高速！"
    fi
}

# ── 测试 ──
run_test() {
    echo ""
    echo "  ╔══════════════════════════════════════════════════════════╗"
    echo "  ║  🧪 开始全量测试  ${CONFIG}  @ ${DATASET}"
    echo "  ╚══════════════════════════════════════════════════════════╝"
    echo ""

    # 清理残留
    pkill -f "test.py" 2>/dev/null || true
    sleep 3

    python tracking/test.py \
        "${SCRIPT}" \
        "${CONFIG}" \
        --dataset_name "${DATASET}" \
        --threads 0 \
        --num_gpus 1 2>&1 | tee "${TEST_LOG}"

    local exit_code=$?
    echo ""
    if [ "${exit_code}" -eq 0 ]; then
        echo "  ✅ 测试成功完成！"
        echo "  日志: ${TEST_LOG}"
    else
        echo "  ⚠️  测试返回 exit=${exit_code}，见日志"
    fi
    echo ""
    return ${exit_code}
}

# ── 分析 ──
run_analysis() {
    echo ""
    echo "  ╔══════════════════════════════════════════════════════════╗"
    echo "  ║  📊 结果分析  ${CONFIG}  @ ${DATASET}"
    echo "  ╚══════════════════════════════════════════════════════════╝"
    echo ""

    python tracking/analysis_results.py \
        --tracker_name "${SCRIPT}" \
        --tracker_param "${CONFIG}" \
        --dataset_name "${DATASET}" 2>&1 | tee "${ANALYSIS_LOG}"

    local exit_code=$?
    if [ "${exit_code}" -eq 0 ]; then
        echo "  ✅ 分析完成！"
    else
        echo "  ⚠️  分析 exit=${exit_code}"
    fi
    echo ""
    return ${exit_code}
}

# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

setup_env
cd /root/autodl-tmp/TBSI_gai/TBSI 2>/dev/null || true

if [ "${FORCE_TEST}" -eq 1 ]; then
    echo "[$(date '+%H:%M:%S')] 🚀 强制测试模式"
    run_test
    run_analysis
    exit $?
fi

if training_finished; then
    echo ""
    echo "  ┌──────────────────────────────────────────────────────────┐"
    echo "  │  🏁 训练已完成！启动测试流程...                              │"
    echo "  └──────────────────────────────────────────────────────────┘"
    echo ""

    # 确认 checkpoint 存在
    local ckpt="${CKPT_DIR}/TBSITrack_ep0015.pth.tar"
    if [ ! -f "${ckpt}" ]; then
        local latest
        latest=$(ls -t "${CKPT_DIR}"/TBSITrack_ep*.pth.tar 2>/dev/null | head -1)
        if [ -z "${latest}" ]; then
            echo "  ❌ 无 checkpoint 可用！"
            exit 1
        fi
        echo "  使用最新 checkpoint: ${latest}"
    else
        local ckpt_size
        ckpt_size=$(du -h "${ckpt}" | cut -f1)
        echo "  ✅ ep0015 checkpoint: ${ckpt_size}"
    fi

    run_test
    local test_ret=$?

    if [ "${test_ret}" -eq 0 ]; then
        run_analysis
    fi

    echo ""
    echo "  ┌──────────────────────────────────────────────────────────┐"
    echo "  │  🏁 全部完成！                                             │"
    echo "  │  测试日志: ${TEST_LOG}"
    echo "  │  分析日志: ${ANALYSIS_LOG}"
    echo "  └──────────────────────────────────────────────────────────┘"

elif training_alive; then
    report_progress
else
    # 进程不在，日志也没完成 -> 可能被杀了
    if [ -f "${TRAIN_LOG}" ] && [ -s "${TRAIN_LOG}" ]; then
        last_epoch=$(grep -oP '\[train: \K\d+' "${TRAIN_LOG}" 2>/dev/null | tail -1)
        echo "[$(date '+%H:%M:%S')] ⚠️  训练进程已终止！上次位于 epoch ${last_epoch:-?}"
        echo "   如果训练意外中断，请运行:"
        echo "   cd /root/autodl-tmp/TBSI_gai/TBSI && python tracking/train.py --script tbsi_track --config ${CONFIG} --save_dir ${SAVE_DIR} --mode single"
    else
        echo "[$(date '+%H:%M:%S')] ⏳ 等待训练启动..."
    fi
fi
