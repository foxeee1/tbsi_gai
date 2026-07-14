#!/bin/bash
# TBSI v3时序令牌快速验证脚本
# ==============================
# 迭代1：①+②+③ 合并验证（3ep，100-seq快速测试）
# 迭代2：④ 6ep全量（如果迭代1 AUC >= 55.5）
#
# 改动清单（v3模块）：
#   ① 纯交叉注意力（消除冗余自注意力噪声）
#   ② 无瓶颈768维原生空间（消除信息丢失）
#   ③ 可学习残差缩放 init=0.3（释放有效信号）

set -e
cd /root/autodl-tmp/TBSI_gai/TBSI

export PYTHONPATH="/root/autodl-tmp/TBSI_gai/TBSI:$PYTHONPATH"
PYTHON="/root/miniconda3/bin/python"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="output/logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ========================================
# 快速测试函数（100-seq子集, 通过API）
# ========================================
fast_eval() {
    local config=$1
    log "快速测试 (100-seq) config=$config ..."
    $PYTHON -c "
from benchmark.bm_evaluate import evaluate
m = evaluate('tbsi_track', '$config', 'lasher_test',
             subset_size=100, threads=0, num_gpus=1, force_eval=True)
print('METRICS_AUC:', m.get('SR', 0))
print('METRICS_OP50:', m.get('OP50', 0))
print('METRICS_OP75:', m.get('OP75', 0))
print('METRICS_PR:', m.get('PR', 0))
print('METRICS_NPR:', m.get('NPR', 0))
" 2>&1 | grep "METRICS_" | tee -a "$LOG_DIR/v3_iteration.log"
}

# ========================================
# 全量测试函数（244-seq, 最终验证）
# ========================================
full_eval() {
    local config=$1
    log "全量测试 (244-seq) ..."
    rm -rf "output/test/tracking_results/tbsi_track/$config/"
    $PYTHON tracking/test.py tbsi_track "$config" \
        --dataset_name lasher_test --threads 0 --num_gpus 1 \
        > "$LOG_DIR/test_${config}_full.log" 2>&1
    $PYTHON tracking/analysis_results.py \
        --tracker_name tbsi_track --tracker_param "$config" \
        --dataset_name lasher_test 2>&1 | grep "vitb_256" \
        | tee -a "$LOG_DIR/v3_iteration.log"
}

# ========================================
# 提取AUC
# ========================================
extract_auc() {
    grep "METRICS_AUC:" "$LOG_DIR/v3_iteration.log" | tail -1 | sed 's/.*AUC: //'
}

# ========================================
# 迭代1：①+②+③ 合并验证 (3ep)
# ========================================
log "=========================================="
log "迭代1: ①+②+③ 合并验证 (pure cross-attn + 768-dim + rs=0.3)"
log "  3epoch, 快速100-seq测试"
log "=========================================="

CONFIG_V3="vitb_256_tbsi_ablation_v3"

# 创建3ep临时配置
TMP_CONFIG="experiments/tbsi_track/vitb_256_tbsi_ablation_v3_quick.yaml"
cp "experiments/tbsi_track/vitb_256_tbsi_ablation_v3.yaml" "$TMP_CONFIG"
sed -i 's/EPOCH: 6/EPOCH: 3/' "$TMP_CONFIG"
sed -i 's/TEST:\n  EPOCH: 6/TEST:\n  EPOCH: 3/' "$TMP_CONFIG" 2>/dev/null || true
# simpler sed
$PYTHON -c "
y = open('$TMP_CONFIG').read()
y = y.replace('EPOCH: 6', 'EPOCH: 3')
y = y.replace('EPOCH: 3\n  SEARCH_FACTOR', 'EPOCH: 3\n  SEARCH_FACTOR')  # test epoch already 6
open('$TMP_CONFIG','w').write(y)
"

# 清理旧检查点
rm -rf "output/checkpoints/train/tbsi_track/${CONFIG_V3}/"
rm -rf "output/test/tracking_results/tbsi_track/${CONFIG_V3}/"

# 训练
log "训练 3epoch ..."
$PYTHON tracking/train.py --script tbsi_track --config vitb_256_tbsi_ablation_v3_quick \
    --save_dir ./output --mode single > "$LOG_DIR/train_v3_quick.log" 2>&1

log "训练完成"

# 快速测试
fast_eval "${CONFIG_V3}"

AUC1=$(extract_auc)
log "迭代1 AUC (100-seq) = $AUC1"

# 输出对比
$PYTHON -c "
auc = $AUC1 if $AUC1 else 0
print()
print('=== 迭代1 结果 ===')
print(f'  v3 (①+②+③) 100-seq AUC: {auc:.2f}')
print(f'  vs bn128基线 (55.40): {auc-55.40:+.2f}')
print(f'  vs DA基线  (55.83): {auc-55.83:+.2f}')
print(f'  vs 噪声下限 (54.58): {auc-54.58:+.2f}')
" | tee -a "$LOG_DIR/v3_iteration.log"

# ========================================
# 迭代2：④ 6ep全量（条件触发）
# ========================================
AUC_THRESHOLD=55.3
if [ "$(echo "$AUC1 >= $AUC_THRESHOLD" | bc 2>/dev/null)" = "1" ] || [ "$(awk -v a="$AUC1" -v t="$AUC_THRESHOLD" 'BEGIN{print (a>=t)?1:0}')" = "1" ]; then
    log ""
    log "=========================================="
    log "AUC=$AUC1 >= $AUC_THRESHOLD ✅, 启动迭代2"
    log "迭代2: ④ 6ep全量训练 + 全量测试"
    log "=========================================="

    # 恢复6ep配置
    rm -rf "output/checkpoints/train/tbsi_track/${CONFIG_V3}/"
    rm -rf "output/test/tracking_results/tbsi_track/${CONFIG_V3}/"

    log "训练 6epoch ..."
    $PYTHON tracking/train.py --script tbsi_track --config vitb_256_tbsi_ablation_v3 \
        --save_dir ./output --mode single > "$LOG_DIR/train_v3_full.log" 2>&1
    log "训练完成"

    log "全量测试 ..."
    full_eval "${CONFIG_V3}"

    log ""
    log "=========================================="
    log "最终结果"
    log "=========================================="
    grep "vitb_256" "$LOG_DIR/v3_iteration.log" | tail -1
    grep -E "vs DA基线|vs bn128" "$LOG_DIR/v3_iteration.log" | tail -2
    log ""
    log "对比基线:"
    log "  sprint_da_ch_full (DA基线) = 55.83"
    log "  ablation_bn128 (bn128帧对) = 55.40"
    log "  4ep_stage2 (bn64帧对)     = 54.94"
    log "  4ep_post_fusion (噪声下限) = 54.58"
else
    log ""
    log "AUC=$AUC1 < $AUC_THRESHOLD, 跳过迭代2"
    log "分析原因后再决定下一步"
fi

log ""
log "完整日志: $LOG_DIR/v3_iteration.log"
