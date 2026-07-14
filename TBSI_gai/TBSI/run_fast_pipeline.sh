#!/usr/bin/env bash
#===============================================================================
# TBSI Fast Pipeline v2 — 全量训练 + 测试 + 10分钟进度监控
#
# 用于完美复现 CVPR 2023 TBSI 基线：
#   Config: vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_reproduce
#   AUC 55.46 / Precision 68.98 / Norm Precision 65.23
#   从零开始训练 → 测试 → 结果分析
#   内置每10分钟进度条 + GPU 利用率监控
#
# 用法:
#   ./run_fast_pipeline.sh                          # 全流程：训练→测试→分析
#   ./run_fast_pipeline.sh --test-only              # 仅测试已有 checkpoint
#   ./run_fast_pipeline.sh --train-only             # 仅训练
#   ./run_fast_pipeline.sh --monitor                # 仅监控已有训练进程
#   ./run_fast_pipeline.sh --interval 300           # 自定义监控间隔（秒）
#   ./run_fast_pipeline.sh --fresh                  # 从头训练（删已有ckpt）
#   ./run_fast_pipeline.sh --config <name>          # 指定其他 config
#===============================================================================

set -euo pipefail

# ======================== 配置 ========================
CONFIG="${CONFIG:-vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_reproduce}"
SCRIPT="${SCRIPT:-tbsi_track}"
SAVE_DIR="${SAVE_DIR:-./output}"
DATASET="${DATASET:-lasher_test}"
CONDA_ENV="${CONDA_ENV:-tbsi}"

LOG_DIR="${SAVE_DIR}/logs"
TRAIN_LOG="${LOG_DIR}/${SCRIPT}-${CONFIG}.log"
CKPT_DIR="${SAVE_DIR}/checkpoints/train/${SCRIPT}/${CONFIG}"
PID_FILE="/tmp/tbsi_train_${CONFIG}.pid"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-600}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PIPELINE_LOG="${LOG_DIR}/fast_pipeline_${TIMESTAMP}.log"
TOTAL_EPOCHS=15

# ======================== Flags ========================
TEST_ONLY=0
TRAIN_ONLY=0
FRESH=0
MONITOR_MODE=0

parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --test-only)    TEST_ONLY=1; shift ;;
            --train-only)   TRAIN_ONLY=1; shift ;;
            --fresh)        FRESH=1; shift ;;
            --monitor)      MONITOR_MODE=1; shift ;;
            --config)       CONFIG="$2"; shift 2 ;;
            --interval)     MONITOR_INTERVAL="$2"; shift 2 ;;
            -h|--help)
                echo "用法: $0 [选项]"
                echo "  --config <name>  指定 config（默认: ${CONFIG}）"
                echo "  --test-only      仅测试已有 checkpoint"
                echo "  --train-only     仅训练"
                echo "  --fresh          从头训练（删除已有 checkpoint）"
                echo "  --monitor        仅监控已有训练进程"
                echo "  --interval <秒>  监控刷新间隔（默认600）"
                exit 0
                ;;
            *)
                echo "未知参数: $1"; exit 1 ;;
        esac
    done
}

parse_args "$@"

# ======================== 工具函数 ========================

info()  { local m="$*"; echo "[$(date '+%H:%M:%S')] ${m}" | tee -a "${PIPELINE_LOG}"; }
ok()    { info "✅ $*"; }
err()   { info "❌ $*"; }
warn()  { info "⚠️  $*"; }
separator() {
    echo "" | tee -a "${PIPELINE_LOG}"
    echo "══════════════════════════════════════════════════════════════" | tee -a "${PIPELINE_LOG}"
    echo " $*" | tee -a "${PIPELINE_LOG}"
    echo "══════════════════════════════════════════════════════════════" | tee -a "${PIPELINE_LOG}"
}

setup_env() {
    unset OMP_NUM_THREADS
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export NUMEXPR_NUM_THREADS=4
    export PYTHONUNBUFFERED=1
    export TORCH_SHOW_CPP_STACKTRACES=0

    local conda_sh="/root/autodl-tmp/conda_envs/${CONDA_ENV}/etc/profile.d/conda.sh"
    if [ -f "${conda_sh}" ]; then
        source "${conda_sh}"
    fi
    conda activate "${CONDA_ENV}" 2>/dev/null || warn "conda env '${CONDA_ENV}' 无法激活，尝试直接运行"

    mkdir -p "${LOG_DIR}"
    touch "${PIPELINE_LOG}"
}

# ======================== GPU 利用率采集 ========================
GPU_STATS_FILE="/tmp/tbsi_gpu_stats_$$.csv"

start_gpu_monitor() {
    (
        echo "timestamp,util_gpu,util_mem,temp,power" > "${GPU_STATS_FILE}"
        while kill -0 "${PPID}" 2>/dev/null; do
            local stats
            stats=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)
            echo "$(date +%s),${stats}" >> "${GPU_STATS_FILE}"
            sleep 5
        done
    ) &
    GPU_MONITOR_PID=$!
}

stop_gpu_monitor() {
    [ -n "${GPU_MONITOR_PID:-}" ] && kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
}

get_gpu_stats() {
    if [ -f "${GPU_STATS_FILE}" ] && [ "$(wc -l < "${GPU_STATS_FILE}")" -gt 1 ]; then
        local avg_gpu avg_mem max_temp
        avg_gpu=$(awk -F',' 'NR>1 {s+=$2; c++} END {if(c>0) printf "%.0f", s/c}' "${GPU_STATS_FILE}")
        avg_mem=$(awk -F',' 'NR>1 {s+=$3; c++} END {if(c>0) printf "%.0f", s/c}' "${GPU_STATS_FILE}")
        max_temp=$(awk -F',' 'NR>1 {if(max<$4) max=$4} END {printf "%.0f", max}' "${GPU_STATS_FILE}")
        echo "GPU:${avg_gpu:-?}% MEM:${avg_mem:-?}% MAX:${max_temp:-?}°C"
    else
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F',' '{printf "GPU:%s%% MEM:%s%% TEMP:%s°C", $1, $2, $3}'
    fi
}

# ======================== 进度监控（每10分钟） ========================

start_monitor() {
    local log_file="$1"
    local start_time="$2"
    local total_epochs="${3:-15}"
    local interval="${4:-600}"

    # 等待日志文件出现
    for i in $(seq 1 30); do
        [ -f "${log_file}" ] && { sleep 2; break; }
        sleep 1
    done

    info "📊 进度监控已启动 (每 ${interval}s 刷新)"

    while true; do
        # 检查进程存活
        local alive=0
        if [ -f "${PID_FILE}" ]; then
            local pid
            pid=$(cat "${PID_FILE}")
            kill -0 "${pid}" 2>/dev/null && alive=1
        fi

        if [ "${alive}" -eq 0 ]; then
            if [ -f "${log_file}" ] && grep -q "Finished training" "${log_file}" 2>/dev/null; then
                info "📊 训练已完成！"
                return 0
            fi
            local now
            now=$(date +%s)
            if [ $(( now - start_time )) -gt 120 ]; then
                info "📊 训练进程已结束，监控退出"
                return 0
            fi
        fi

        # ── 解析训练日志 ──
        local epoch="" iter="" total_iter="" fps="" loss_giou="" loss_l1="" loss_cls="" lr=""
        local data_time="" forward_time="" total_time=""

        local train_line
        train_line=$(grep -oP '\[train: \d+, \d+ / \d+\].*' "${log_file}" 2>/dev/null | tail -1)

        if [ -n "${train_line}" ]; then
            epoch=$(echo "${train_line}" | grep -oP '\[train: \K\d+')
            local it
            it=$(echo "${train_line}" | grep -oP '\[train: \d+, \K[\d /]+\]' | tr -d '[]')
            iter=$(echo "${it}" | cut -d'/' -f1 | tr -d ' ')
            total_iter=$(echo "${it}" | cut -d'/' -f2 | tr -d ' ')
            fps=$(echo "${train_line}" | grep -oP 'FPS: \K[0-9.]+')
            loss_giou=$(echo "${train_line}" | grep -oP 'giou: \K[0-9.]+')
            loss_l1=$(echo "${train_line}" | grep -oP 'l1: \K[0-9.]+')
            loss_cls=$(echo "${train_line}" | grep -oP 'focal: \K[0-9.]+')
            lr=$(echo "${train_line}" | grep -oP 'LearningRate/group0: \K[0-9.e-]+')
            data_time=$(echo "${train_line}" | grep -oP 'DataTime: \K[0-9.]+')
            forward_time=$(echo "${train_line}" | grep -oP 'ForwardTime: \K[0-9.]+')
            total_time=$(echo "${train_line}" | grep -oP 'TotalTime: \K[0-9.]+')
        fi

        [ -z "${epoch}" ] && epoch=$(grep -oP 'EPOCH: \K\d+' "${log_file}" 2>/dev/null | tail -1)

        # ── 计算进度 ──
        local pct=0
        if [ -n "${epoch}" ] && [ "${total_epochs}" -gt 0 ] 2>/dev/null; then
            pct=$(( epoch * 100 / total_epochs ))
        fi

        # ── 进度条 ──
        local bar_len=40
        local filled=$(( pct * bar_len / 100 ))
        local empty=$(( bar_len - filled ))
        local bar="["
        for ((i=0; i<filled; i++)); do bar+="█"; done
        for ((i=0; i<empty; i++)); do bar+="░"; done
        bar+="]"

        # ── ETA ──
        local now
        now=$(date +%s)
        local elapsed=$(( now - start_time ))
        local eta_str="--"
        if [ "${pct}" -gt 0 ] && [ "${pct}" -lt 100 ] 2>/dev/null; then
            local eta_sec=$(( elapsed * 100 / pct - elapsed ))
            eta_str=$(printf "%dh%dm" $(( eta_sec / 3600 )) $(( (eta_sec % 3600) / 60 )))
        fi
        local elapsed_fmt
        elapsed_fmt=$(printf "%dh%dm" $(( elapsed / 3600 )) $(( (elapsed % 3600) / 60 )))

        # ── GPU 统计 ──
        local gpu_stats
        gpu_stats=$(get_gpu_stats)

        # ── 输出 ──
        echo "" | tee -a "${PIPELINE_LOG}"
        echo "  ┌─────────────────────────────────────────────────────────────" | tee -a "${PIPELINE_LOG}"
        echo "  │ 🔍 训练进度 @ $(date '+%H:%M:%S')" | tee -a "${PIPELINE_LOG}"
        echo "  │ ${gpu_stats}" | tee -a "${PIPELINE_LOG}"
        echo "  │ ⏱  ${elapsed_fmt} elapsed  |  ETA ${eta_str}" | tee -a "${PIPELINE_LOG}"
        echo "  ├─────────────────────────────────────────────────────────────" | tee -a "${PIPELINE_LOG}"
        echo "  │ Epoch ${epoch:-?}/${total_epochs}  ${bar}  ${pct}%" | tee -a "${PIPELINE_LOG}"
        if [ -n "${iter}" ] && [ -n "${total_iter}" ] && [ "${total_iter}" -gt 0 ] 2>/dev/null; then
            echo "  │ Iter  ${iter}/${total_iter}  FPS: ${fps:-N/A}" | tee -a "${PIPELINE_LOG}"
        fi
        if [ -n "${lr}" ]; then
            echo "  │ LR: ${lr}" | tee -a "${PIPELINE_LOG}"
        fi
        if [ -n "${loss_giou}" ] || [ -n "${loss_l1}" ] || [ -n "${loss_cls}" ]; then
            echo "  │ 📉 Loss: giou=${loss_giou:-N/A}  l1=${loss_l1:-N/A}  focal=${loss_cls:-N/A}" | tee -a "${PIPELINE_LOG}"
        fi
        if [ -n "${data_time}" ] && [ -n "${forward_time}" ] && [ -n "${total_time}" ]; then
            local data_pct
            data_pct=$(echo "scale=1; ${data_time} / ${total_time} * 100" | bc -l 2>/dev/null || echo "?")
            local bottleneck=""
            [ "$(echo "${data_pct} > 50" | bc -l 2>/dev/null)" = "1" ] && bottleneck=" 🚨 DataLoader瓶颈"
            [ "$(echo "${data_pct} > 30" | bc -l 2>/dev/null)" = "1" ] && [ -z "${bottleneck}" ] && bottleneck=" ⚠️ DataLoader偏高"
            echo "  │ ⏳ Fwd:${forward_time}s  Data:${data_time}s (${data_pct}%${bottleneck})  Total:${total_time}s" | tee -a "${PIPELINE_LOG}"
        fi
        echo "  └─────────────────────────────────────────────────────────────" | tee -a "${PIPELINE_LOG}"

        sleep "${interval}"
    done
}

# ======================== 训练 ========================
run_train() {
    separator "🚀 训练阶段: ${CONFIG}  |  从头开始 → ep15"

    # ── Fresh / 检查 ──
    if [ -d "${CKPT_DIR}" ] && [ "$(ls "${CKPT_DIR}"/*.pth.tar 2>/dev/null | wc -l)" -gt 0 ]; then
        local latest_ckpt
        latest_ckpt=$(ls -t "${CKPT_DIR}"/*.pth.tar 2>/dev/null | head -1)
        if [ "${FRESH}" -eq 1 ] || [ "${TEST_ONLY}" -eq 0 ]; then
            warn "⚠️  发现已有检查点（将删除以从零开始）"
            info "  现有: $(basename "${latest_ckpt}")"
            info "  删除中..."
            rm -f "${CKPT_DIR}"/*.pth.tar 2>/dev/null || true
            # 也删除 optimizer checkpoint
            rm -f "${CKPT_DIR}"/*.optim 2>/dev/null || true
            ok "已清除旧检查点"
        fi
    fi

    echo "" | tee -a "${PIPELINE_LOG}"

    # ── 训练命令 ──
    local TRAIN_CMD="python tracking/train.py \
        --script ${SCRIPT} \
        --config ${CONFIG} \
        --save_dir ${SAVE_DIR} \
        --mode single"

    info "运行: ${TRAIN_CMD}"

    # ── 启动 GPU 采集（后台） ──
    start_gpu_monitor

    # ── 启动训练 ──
    ${TRAIN_CMD} 2>&1 | tee -a "${PIPELINE_LOG}" &
    local train_pid=$!
    echo "${train_pid}" > "${PID_FILE}"
    info "训练 PID: ${train_pid}"

    # ── 启动监控（后台） ──
    local START_TIME
    START_TIME=$(date +%s)
    start_monitor "${TRAIN_LOG}" "${START_TIME}" "${TOTAL_EPOCHS}" "${MONITOR_INTERVAL}" &
    local monitor_pid=$!

    # ── 等待训练完成 ──
    set +e
    wait "${train_pid}" 2>/dev/null
    local exit_code=$?
    set -e

    # ── 清理 ──
    rm -f "${PID_FILE}"
    stop_gpu_monitor
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true

    if [ "${exit_code}" -eq 0 ]; then
        ok "训练成功完成"
    else
        err "训练异常退出 (exit=${exit_code})"
        return ${exit_code}
    fi

    # ── 验证最终 checkpoint ──
    local final_ckpt
    final_ckpt=$(ls -t "${CKPT_DIR}"/TBSITrack_ep*.pth.tar 2>/dev/null | head -1)
    if [ -n "${final_ckpt}" ]; then
        local ckpt_size
        ckpt_size=$(du -h "${final_ckpt}" | cut -f1)
        ok "最终 checkpoint: $(basename "${final_ckpt}") (${ckpt_size})"
    else
        err "无 checkpoint 生成"
        return 1
    fi

    # ── GPU 利用率摘要 ──
    if [ -f "${GPU_STATS_FILE}" ]; then
        echo "" | tee -a "${PIPELINE_LOG}"
        info "📊 GPU 利用率统计摘要:"
        local avg_gpu avg_mem max_temp
        avg_gpu=$(awk -F',' 'NR>1 {s+=$2; c++} END {printf "%.0f", s/c}' "${GPU_STATS_FILE}" 2>/dev/null)
        avg_mem=$(awk -F',' 'NR>1 {s+=$3; c++} END {printf "%.0f", s/c}' "${GPU_STATS_FILE}" 2>/dev/null)
        max_temp=$(awk -F',' 'NR>1 {if(max<$4) max=$4} END {printf "%.0f", max}' "${GPU_STATS_FILE}" 2>/dev/null)
        info "  平均 GPU 利用率: ${avg_gpu:-N/A}%"
        info "  平均 显存利用率: ${avg_mem:-N/A}%"
        info "  最高 温度: ${max_temp:-N/A}°C"
        rm -f "${GPU_STATS_FILE}"
    fi

    return 0
}

# ======================== 测试 ========================
run_test() {
    separator "🧪 测试阶段: ${CONFIG}  @ ${DATASET}"

    # ── 定位 checkpoint ──
    local ckpt
    ckpt=$(ls -t "${CKPT_DIR}"/TBSITrack_ep*.pth.tar 2>/dev/null | head -1)
    if [ -z "${ckpt}" ]; then
        err "无 checkpoint 可用，无法测试"
        return 1
    fi
    info "使用 checkpoint: $(basename "${ckpt}")"

    # 清理残留
    pkill -f "tracking/test.py" 2>/dev/null || true
    sleep 2

    echo "" | tee -a "${PIPELINE_LOG}"

    local TEST_CMD="python tracking/test.py \
        ${SCRIPT} \
        ${CONFIG} \
        --dataset_name ${DATASET} \
        --threads 0 \
        --num_gpus 1"

    info "运行: ${TEST_CMD}"

    local test_start
    test_start=$(date +%s)
    ${TEST_CMD} 2>&1 | tee -a "${PIPELINE_LOG}"
    local exit_code=$?
    local test_duration=$(( $(date +%s) - test_start ))

    if [ "${exit_code}" -eq 0 ]; then
        ok "测试完成 (耗时: $(printf '%dh%dm' $(( test_duration / 3600 )) $(( (test_duration % 3600) / 60 ))))"
    else
        err "测试异常退出 (exit=${exit_code})"
    fi
    return ${exit_code}
}

# ======================== 分析 ========================
run_analysis() {
    separator "📊 分析阶段: ${CONFIG}  @ ${DATASET}"

    local ANALYSIS_CMD="python tracking/analysis_results.py \
        --tracker_name ${SCRIPT} \
        --tracker_param ${CONFIG} \
        --dataset_name ${DATASET}"

    info "运行: ${ANALYSIS_CMD}"
    ${ANALYSIS_CMD} 2>&1 | tee -a "${PIPELINE_LOG}"
    local exit_code=$?

    if [ "${exit_code}" -eq 0 ]; then
        ok "分析完成"
    else
        warn "分析返回非零 (exit=${exit_code})"
    fi
    return ${exit_code}
}

# ======================== 仅监控模式 ========================
run_monitor_only() {
    info "📊 仅监控模式：查找已有训练进程..."

    local train_pid=""
    [ -f "${PID_FILE}" ] && train_pid=$(cat "${PID_FILE}") && kill -0 "${train_pid}" 2>/dev/null || train_pid=""
    [ -z "${train_pid}" ] && train_pid=$(pgrep -f "train.*${CONFIG}" 2>/dev/null | head -1 || true)

    if [ -n "${train_pid}" ]; then
        info "找到训练进程 PID=${train_pid}"
        echo "${train_pid}" > "${PID_FILE}"
        start_monitor "${TRAIN_LOG}" "$(date +%s)" "${TOTAL_EPOCHS}" "${MONITOR_INTERVAL}"
    elif [ -f "${TRAIN_LOG}" ] && grep -q "Finished training" "${TRAIN_LOG}" 2>/dev/null; then
        info "训练日志显示训练已结束"
        info "日志: ${TRAIN_LOG}"
    else
        err "未找到训练进程 (${CONFIG})"
        err "请先运行: $0"
        return 1
    fi
}

# ======================== 主入口 ========================
main() {
    local start_ts
    start_ts=$(date +%s)

    setup_env

    echo "" | tee -a "${PIPELINE_LOG}"
    echo "╔══════════════════════════════════════════════════════════════╗" | tee -a "${PIPELINE_LOG}"
    echo "║  TBSI Pipeline — 基线复现" | tee -a "${PIPELINE_LOG}"
    echo "║  ${TIMESTAMP}" | tee -a "${PIPELINE_LOG}"
    echo "╠══════════════════════════════════════════════════════════════╣" | tee -a "${PIPELINE_LOG}"
    echo "║  Config:     ${CONFIG}" | tee -a "${PIPELINE_LOG}"
    echo "║  Save Dir:   ${SAVE_DIR}" | tee -a "${PIPELINE_LOG}"
    echo "║  Dataset:    ${DATASET}" | tee -a "${PIPELINE_LOG}"
    echo "║  Monitor:    每 ${MONITOR_INTERVAL}s 刷新" | tee -a "${PIPELINE_LOG}"
    echo "║  训练参数与原始 baseline 完全一致" | tee -a "${PIPELINE_LOG}"
    echo "╚══════════════════════════════════════════════════════════════╝" | tee -a "${PIPELINE_LOG}"
    echo "" | tee -a "${PIPELINE_LOG}"
    info "日志: ${PIPELINE_LOG}"

    if [ "${MONITOR_MODE}" -eq 1 ]; then
        run_monitor_only
    elif [ "${TEST_ONLY}" -eq 1 ]; then
        run_test && run_analysis
    elif [ "${TRAIN_ONLY}" -eq 1 ]; then
        run_train
    else
        if run_train; then
            run_test && run_analysis
        else
            warn "训练失败，跳过测试"
            warn "训练完成后可运行: $0 --test-only"
        fi
    fi

    local end_ts
    end_ts=$(date +%s)
    local total=$(( end_ts - start_ts ))
    echo "" | tee -a "${PIPELINE_LOG}"
    separator "🏁 完成！"
    info "总耗时: $(printf '%dh%dm%ds' $(( total / 3600 )) $(( (total % 3600) / 60 )) $(( total % 60 )))"
    info "流水线日志: ${PIPELINE_LOG}"
    echo "" | tee -a "${PIPELINE_LOG}"
}

cleanup() { rm -f "/tmp/tbsi_gpu_stats_$$.csv"; }
trap cleanup EXIT

main
