#!/usr/bin/env bash
#===============================================================================
# TBSI Fast Test — 使用 FP16 + inference_mode + 帧预加载 加速测试
#
# 基于 指南/复现.txt 第十章的测试加速方案：
#   ✅ 模板特征缓存（代码已有）   ✅ FP16 半精度推理（代码已有）
#   ✅ torch.inference_mode()      ✅ 预加载帧到 GPU（代码已有）
#   ✅ 消除 CPU-GPU 同步点        ✅ Checkpoint 模型缓存（代码已有）
#
# 用法:
#   ./run_test_fast.sh                          # 测试默认 config
#   ./run_test_fast.sh --config <name>          # 指定 config
#   ./run_test_fast.sh --epoch 10               # 指定 epoch
#   ./run_test_fast.sh --dataset lasher_test    # 指定数据集
#   ./run_test_fast.sh --analyze                # 测试后自动分析
#===============================================================================

set -euo pipefail

# ======================== 配置 ========================
CONFIG="vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_reproduce"
SCRIPT="tbsi_track"
SAVE_DIR="./output"
DATASET="lasher_test"
CONDA_ENV="tbsi"
TEST_EPOCH=""
DO_ANALYZE=0

LOG_DIR="${SAVE_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEST_LOG="${LOG_DIR}/test_${CONFIG}_${TIMESTAMP}.log"
CKPT_DIR="${SAVE_DIR}/checkpoints/train/${SCRIPT}/${CONFIG}"

parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --config)    CONFIG="$2"; shift 2 ;;
            --epoch)     TEST_EPOCH="$2"; shift 2 ;;
            --dataset)   DATASET="$2"; shift 2 ;;
            --analyze)   DO_ANALYZE=1; shift ;;
            -h|--help)
                echo "用法: $0 [选项]"
                echo "  --config <name>   Config 名称 (默认: ${CONFIG})"
                echo "  --epoch <N>       指定测试 epoch (默认: config 中 TEST.EPOCH)"
                echo "  --dataset <name>  数据集 (默认: ${DATASET})"
                echo "  --analyze         测试后自动分析结果"
                exit 0
                ;;
            *) echo "未知: $1"; exit 1 ;;
        esac
    done
}

parse_args "$@"

info()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${TEST_LOG}"; }
ok()    { info "✅ $*"; }
err()   { info "❌ $*"; }
warn()  { info "⚠️  $*"; }
separator() {
    echo "" | tee -a "${TEST_LOG}"
    echo "════════════════════════════════════════════════════" | tee -a "${TEST_LOG}"
    echo " $*" | tee -a "${TEST_LOG}"
    echo "════════════════════════════════════════════════════" | tee -a "${TEST_LOG}"
}

setup_env() {
    unset OMP_NUM_THREADS
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export PYTHONUNBUFFERED=1

    local conda_sh="/root/autodl-tmp/conda_envs/${CONDA_ENV}/etc/profile.d/conda.sh"
    [ -f "${conda_sh}" ] && source "${conda_sh}"
    conda activate "${CONDA_ENV}" 2>/dev/null || warn "conda env '${CONDA_ENV}' 无法激活"

    mkdir -p "${LOG_DIR}"
    touch "${TEST_LOG}"
}

find_checkpoint() {
    if [ -n "${TEST_EPOCH}" ]; then
        local ckpt="${CKPT_DIR}/TBSITrack_ep$(printf '%04d' ${TEST_EPOCH}).pth.tar"
        if [ -f "${ckpt}" ]; then
            echo "${ckpt}"
            return 0
        fi
        warn "指定 epoch ${TEST_EPOCH} 的 checkpoint 不存在"
    fi

    # 找最新
    local latest=$(ls -t "${CKPT_DIR}"/TBSITrack_ep*.pth.tar 2>/dev/null | head -1)
    if [ -z "${latest}" ]; then
        err "无 checkpoint 可用: ${CKPT_DIR}"
        return 1
    fi
    echo "${latest}"
}

main() {
    setup_env

    separator "🧪 TBSI Fast Test"
    info "Config:  ${CONFIG}"
    info "Dataset: ${DATASET}"
    info "Log:     ${TEST_LOG}"

    # ── 确认 checkpoint ──
    local ckpt
    ckpt=$(find_checkpoint) || exit 1
    local ckpt_name
    ckpt_name=$(basename "${ckpt}")
    local ckpt_size
    ckpt_size=$(du -h "${ckpt}" | cut -f1)
    info "Checkpoint: ${ckpt_name} (${ckpt_size})"

    # ── 清理残留进程 ──
    pkill -f "tracking/test.py" 2>/dev/null || true
    sleep 2

    # ── 运行测试 ──
    separator "运行测试 (FP16 + inference_mode + 帧预加载)"

    local TEST_CMD="python tracking/test.py \
        ${SCRIPT} \
        ${CONFIG} \
        --dataset_name ${DATASET} \
        --threads 0 \
        --num_gpus 1"

    info "运行: ${TEST_CMD}"

    local test_start
    test_start=$(date +%s)
    ${TEST_CMD} 2>&1 | tee -a "${TEST_LOG}"
    local exit_code=$?
    local test_duration=$(( $(date +%s) - test_start ))

    echo "" | tee -a "${TEST_LOG}"

    if [ "${exit_code}" -eq 0 ]; then
        ok "测试成功完成 (耗时: $(printf '%dh%dm%ds' $(( test_duration / 3600 )) $(( (test_duration % 3600) / 60 )) $(( test_duration % 60 ))))"
    else
        err "测试异常退出 (exit=${exit_code})"
    fi

    # ── 分析 ──
    if [ "${DO_ANALYZE}" -eq 1 ] || [ "${exit_code}" -eq 0 ]; then
        separator "📊 结果分析"

        local ANALYSIS_CMD="python tracking/analysis_results.py \
            --tracker_name ${SCRIPT} \
            --tracker_param ${CONFIG} \
            --dataset_name ${DATASET}"

        info "运行: ${ANALYSIS_CMD}"
        ${ANALYSIS_CMD} 2>&1 | tee -a "${TEST_LOG}"
        local ana_exit=$?
        if [ "${ana_exit}" -eq 0 ]; then
            ok "分析完成"
        else
            warn "分析返回非零 (exit=${ana_exit})"
        fi
    fi

    separator "🏁 完成！"
    info "日志: ${TEST_LOG}"
    echo ""
}

main
