#!/bin/bash
# ==============================================================================
# DGSFusion v4-diffproj 全量实验：训练 → 测试 → 分析
# 配置: v1.5.0-v4-diffproj-全量 (基于完美基线, 仅添加差异投影路由)
# 数据集: LasHeR 完整训练集 (979 seqs) + 测试集 (244 seqs)
# ==============================================================================
set -e

cd /root/autodl-tmp/TBSI_gai/TBSI
unset OMP_NUM_THREADS

CONFIG="v1.5.0-v4-diffproj-全量"
EXP_DIR="./output/experiments/${CONFIG}"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
MAIN_LOG="/tmp/full_v4_pipeline_${TIMESTAMP}.log"

echo "================================================" | tee -a "${MAIN_LOG}"
echo " DGSFusion v4 全量实验" | tee -a "${MAIN_LOG}"
echo " 配置: ${CONFIG}" | tee -a "${MAIN_LOG}"
echo " 开始: $(date)" | tee -a "${MAIN_LOG}"
echo "================================================" | tee -a "${MAIN_LOG}"

# ===== Step 1: 训练 =====
echo "" | tee -a "${MAIN_LOG}"
echo "[$(date)] === Step 1/3: 训练 ===" | tee -a "${MAIN_LOG}"
echo " 预估耗时: ~5.5h (LasHeR 全量训练集, 15 epoch)" | tee -a "${MAIN_LOG}"

python tracking/train.py \
  --script tbsi_track \
  --config "${CONFIG}" \
  --save_dir ./output \
  --mode single 2>&1 | tee -a "${EXP_DIR}/logs/train_full.log" "${MAIN_LOG}"

echo "[$(date)] 训练完成" | tee -a "${MAIN_LOG}"

# ===== Step 2: 测试 =====
echo "" | tee -a "${MAIN_LOG}"
echo "[$(date)] === Step 2/3: 测试 (LasHeR test, 244 seqs) ===" | tee -a "${MAIN_LOG}"
echo " 预估耗时: ~20min" | tee -a "${MAIN_LOG}"
echo " 注意: 测试使用单进程模式 (threads=0)" | tee -a "${MAIN_LOG}"

python tracking/test.py tbsi_track "${CONFIG}" \
  --dataset_name lasher_test --threads 0 --num_gpus 1 2>&1 | tee -a "/tmp/test_full_v4_${TIMESTAMP}.log" "${MAIN_LOG}"

echo "[$(date)] 测试完成" | tee -a "${MAIN_LOG}"

# ===== Step 3: 评估 =====
echo "" | tee -a "${MAIN_LOG}"
echo "[$(date)] === Step 3/3: 评估 ===" | tee -a "${MAIN_LOG}"

python tracking/analysis_results.py \
  --tracker_name tbsi_track \
  --tracker_param "${CONFIG}" \
  --dataset_name lasher_test 2>&1 | tee -a "${MAIN_LOG}"

echo "" | tee -a "${MAIN_LOG}"
echo "================================================" | tee -a "${MAIN_LOG}"
echo " 全部完成!" | tee -a "${MAIN_LOG}"
echo " 结束: $(date)" | tee -a "${MAIN_LOG}"
echo " 日志: ${MAIN_LOG}" | tee -a "${MAIN_LOG}"
echo "================================================" | tee -a "${MAIN_LOG}"
