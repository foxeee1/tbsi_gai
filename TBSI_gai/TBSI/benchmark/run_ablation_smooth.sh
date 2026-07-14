#!/bin/bash
# TBSI 时序消融实验流水线
# 方案一: EMA权重平滑 (推理侧, 零训练)
# 方案二: 质量一致性正则 (训练3ep)
set -e
cd /root/autodl-tmp/TBSI_gai/TBSI
PYTHON="/root/miniconda3/bin/python"
LOG_DIR="output/logs"
mkdir -p "$LOG_DIR"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_50seq_test() {
    local config=$1 tag=$2
    log "50-seq测试: $tag"
    rm -rf "output/test/tracking_results/tbsi_track/$config/" 2>/dev/null
    $PYTHON -c "
import sys, warnings, torch; warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from lib.test.evaluation import get_dataset, Tracker
from lib.test.evaluation.running import run_dataset
from lib.test.analysis.extract_results import extract_results

dataset = get_dataset('lasher_test')[:50]
tracker = Tracker('tbsi_track', '$config', 'lasher_test', run_id=None)
run_dataset(dataset, [tracker], debug=0, threads=0, num_gpus=1)

report = '${tag}_50'
eval_data = extract_results([tracker], dataset, report)
valid = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)
sr = torch.tensor(eval_data['ave_success_rate_plot_overlap'])
sr_valid = sr[valid, 0, :]
auc = sr_valid.mean(dim=1).mean().item() * 100.0

thresh_p = torch.tensor(eval_data['threshold_set_center'])
idx20 = (thresh_p == 20.0).nonzero(as_tuple=True)[0]
pr = torch.tensor(eval_data['ave_success_rate_plot_center'])
pr_v = pr[valid, 0, :]
prec = pr_v[:, idx20].mean().item() * 100.0 if idx20.numel() > 0 else 0.0

print(f'RESULT_{tag}_AUC: {auc:.2f}')
print(f'RESULT_{tag}_PR: {prec:.2f}')
" 2>&1 | grep "RESULT_" | tee -a "$LOG_DIR/ablation_smooth.log"
}

# ===== 方案一: EMA平滑 (零训练, 直接推理) =====
log "========================================"
log "方案一: DA融合EMA权重平滑 (推理侧, 零训练)"
log "========================================"

CONFIG_SMOOTH="vitb_256_tbsi_ablation_smooth"
# Create checkpoint dir with symlink to baseline
mkdir -p "output/checkpoints/train/tbsi_track/$CONFIG_SMOOTH/"
BASELINE_CKPT="output/checkpoints/train/tbsi_track/vitb_256_tbsi_sprint_da_ch_full/TBSITrack_ep0004.pth.tar"
ln -sf "$(realpath $BASELINE_CKPT)" "output/checkpoints/train/tbsi_track/$CONFIG_SMOOTH/TBSITrack_ep0003.pth.tar"
log "基线检查点已链接"

run_50seq_test "$CONFIG_SMOOTH" "SMOOTH"
log "方案一完成"

# ===== 方案二: 质量一致性正则 (训练3ep) =====
log ""
log "========================================"
log "方案二: 跨帧质量一致性正则 (训练3ep)"
log "========================================"

# Remove symlink from方案一
rm -f "output/checkpoints/train/tbsi_track/$CONFIG_SMOOTH/TBSITrack_ep0003.pth.tar"
rm -rf "output/checkpoints/train/tbsi_track/$CONFIG_SMOOTH/"

log "训练3epoch..."
$PYTHON -W ignore tracking/train.py \
    --script tbsi_track --config "$CONFIG_SMOOTH" \
    --save_dir ./output --mode single > "$LOG_DIR/train_consistency.log" 2>&1
log "训练完成"

# Test
rm -rf "output/test/tracking_results/tbsi_track/$CONFIG_SMOOTH/"
run_50seq_test "$CONFIG_SMOOTH" "CONSIST"
log "方案二完成"

# ===== 结果汇总 =====
log ""
log "========================================"
log "消融实验结果汇总"
log "========================================"
echo ""
echo "方案一 (EMA平滑, 零训练):"; grep "RESULT_SMOOTH" "$LOG_DIR/ablation_smooth.log"
echo "方案二 (一致性正则, 3ep):"; grep "RESULT_CONSIST" "$LOG_DIR/ablation_smooth.log"
echo ""
echo "对比基线 (sprint_da_ch_full全量): AUC=55.83"
log "完整日志: $LOG_DIR/ablation_smooth.log"
