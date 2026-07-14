#!/bin/bash
# TBSI 三级流水线基线运行脚本 (SOT权重)
# 自动: L1 Sprint → 评测 → L2 Verify → 评测 → L3 Full → 评测
# 使用 benchmark API 评测 (自动处理 checkpoint 路径)

set -e

cd /root/autodl-tmp/TBSI_gai/TBSI
unset OMP_NUM_THREADS
export PYTHONPATH="/root/autodl-tmp/TBSI_gai/TBSI:$PYTHONPATH"
PYTHON="/root/autodl-tmp/conda_envs/tbsi/bin/python"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="benchmark/ledgers"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/pipeline_$TIMESTAMP.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGFILE"; }

log "=========================================="
log " TBSI 三级流水线基线 (SOT权重)"
log " 开始时间: $(date)"
log "=========================================="

# ======================================================
# LEVEL 1: SPRINT
# ======================================================
log ""
log "========== LEVEL 1: SPRINT =========="
log " 4ep x 12k smpl, bs32, 100seq test"
log " 预计: ~50min train + ~10min test"

START_L1=$(date +%s)

log "L1 训练开始..."
$PYTHON tracking/train.py \
    --script tbsi_track \
    --config vitb_256_tbsi_sprint \
    --save_dir ./output \
    --mode single 2>&1 | tee -a "$LOGDIR/train_l1.log"
log "L1 训练完成"

log "L1 评测 (100 seq, API)..."
$PYTHON -c "
import json, sys
sys.path.insert(0, '.')
from benchmark.bm_evaluate import evaluate_level
from benchmark.bm_config import LEVEL1
metrics = evaluate_level(LEVEL1, threads=4, num_gpus=1, force_eval=True)
print('L1_METRICS:', json.dumps(metrics))
" 2>&1 | tee -a "$LOGDIR/eval_l1.log"
log "L1 评测完成"

# Extract metrics
L1_METRICS=$(grep 'L1_METRICS:' "$LOGDIR/eval_l1.log" | sed 's/.*L1_METRICS: //')
log "L1 结果: $L1_METRICS"

END_L1=$(date +%s)
L1_TIME=$(( (END_L1 - START_L1) / 60 ))
log "L1 耗时: ${L1_TIME}min"

# Save to ledger
$PYTHON -c "
import json
from benchmark.bm_config import LEDGERS_DIR
metrics = json.loads('$L1_METRICS')
ledger = {
    'session_id': '$TIMESTAMP',
    'created_at': '$(date -Iseconds)',
    'current_level': 1,
    'baselines': {'level1': metrics},
    'iterations': [{
        'id': 'baseline_l1', 'level': 1,
        'timestamp': '$(date -Iseconds)',
        'description': '[baseline] L1 Sprint (SOT)',
        'changed_files': [], 'metrics': metrics, 'gate_passed': True,
    }],
    'advancements': [],
    'total_iterations': 1,
    'is_completed': False,
}
with open('$LOGDIR/bm_ledger.json', 'w') as f:
    json.dump(ledger, f, indent=2, default=str)
print('L1 baseline saved')
" 2>&1 | tee -a "$LOGFILE"

# ======================================================
# LEVEL 2: VERIFY
# ======================================================
log ""
log "========== LEVEL 2: VERIFY =========="
log " 8ep x 20k smpl, bs32, 245seq full test"
log " 预计: ~2.5h train + ~20min test"

START_L2=$(date +%s)

log "L2 训练开始..."
$PYTHON tracking/train.py \
    --script tbsi_track \
    --config vitb_256_tbsi_verify \
    --save_dir ./output \
    --mode single 2>&1 | tee -a "$LOGDIR/train_l2.log"
log "L2 训练完成"

log "L2 评测 (全量 245 seq, API)..."
$PYTHON -c "
import json, sys
sys.path.insert(0, '.')
from benchmark.bm_evaluate import evaluate_level
from benchmark.bm_config import LEVEL2
metrics = evaluate_level(LEVEL2, threads=6, num_gpus=1, force_eval=True)
print('L2_METRICS:', json.dumps(metrics))
" 2>&1 | tee -a "$LOGDIR/eval_l2.log"
log "L2 评测完成"

L2_METRICS=$(grep 'L2_METRICS:' "$LOGDIR/eval_l2.log" | sed 's/.*L2_METRICS: //')
log "L2 结果: $L2_METRICS"

END_L2=$(date +%s)
L2_TIME=$(( (END_L2 - START_L2) / 60 ))
log "L2 耗时: ${L2_TIME}min"

# Update ledger
$PYTHON -c "
import json
from benchmark.bm_config import LEDGERS_DIR
metrics = json.loads('$L2_METRICS')
with open('$LOGDIR/bm_ledger.json') as f:
    ledger = json.load(f)
ledger['baselines']['level2'] = metrics
ledger['current_level'] = 2
ledger['iterations'].append({
    'id': 'baseline_l2', 'level': 2,
    'timestamp': '$(date -Iseconds)',
    'description': '[baseline] L2 Verify (SOT)',
    'changed_files': [], 'metrics': metrics, 'gate_passed': True,
})
ledger['total_iterations'] = 2
with open('$LOGDIR/bm_ledger.json', 'w') as f:
    json.dump(ledger, f, indent=2, default=str)
print('L2 baseline saved')
" 2>&1 | tee -a "$LOGFILE"

# ======================================================
# LEVEL 3: FULL (原始 SOT 配置)
# ======================================================
log ""
log "========== LEVEL 3: FULL =========="
log " 15ep x 60k smpl, bs32, 245seq full test"
log " 预计: ~12h train + ~20min test"

START_L3=$(date +%s)

log "L3 训练开始..."
$PYTHON tracking/train.py \
    --script tbsi_track \
    --config vitb_256_tbsi_32x1_1e4_lasher_15ep_sot \
    --save_dir ./output \
    --mode single 2>&1 | tee -a "$LOGDIR/train_l3.log"
log "L3 训练完成"

log "L3 评测 (全量 245 seq, API)..."
$PYTHON -c "
import json, sys
sys.path.insert(0, '.')
from benchmark.bm_evaluate import evaluate_level
from benchmark.bm_config import LEVEL3
metrics = evaluate_level(LEVEL3, threads=6, num_gpus=1, force_eval=True)
print('L3_METRICS:', json.dumps(metrics))
" 2>&1 | tee -a "$LOGDIR/eval_l3.log"
log "L3 评测完成"

L3_METRICS=$(grep 'L3_METRICS:' "$LOGDIR/eval_l3.log" | sed 's/.*L3_METRICS: //')
log "L3 结果: $L3_METRICS"

END_L3=$(date +%s)
L3_TIME=$(( (END_L3 - START_L3) / 60 ))

# Final update
$PYTHON -c "
import json
from benchmark.bm_config import LEDGERS_DIR
metrics = json.loads('$L3_METRICS')
with open('$LOGDIR/bm_ledger.json') as f:
    ledger = json.load(f)
ledger['baselines']['level3'] = metrics
ledger['current_level'] = 3
ledger['is_completed'] = True
ledger['iterations'].append({
    'id': 'baseline_l3', 'level': 3,
    'timestamp': '$(date -Iseconds)',
    'description': '[baseline] L3 Full (SOT)',
    'changed_files': [], 'metrics': metrics, 'gate_passed': True,
})
ledger['total_iterations'] = 3
with open('$LOGDIR/bm_ledger.json', 'w') as f:
    json.dump(ledger, f, indent=2, default=str)
print('L3 baseline saved')
" 2>&1 | tee -a "$LOGFILE"

# ======================================================
# 最终报告
# ======================================================
TOTAL_TIME=$(( (END_L3 - START_L1) / 60 ))
TOTAL_H=$(( TOTAL_TIME / 60 ))
TOTAL_M=$(( TOTAL_TIME % 60 ))

log ""
log "=========================================="
log " TBSI 三级流水线基线 全部完成!"
log "=========================================="
log " L1 Sprint: ${L1_TIME}min"
log " L2 Verify: ${L2_TIME}min"
log " L3 Full:   ${L3_TIME}min"
log " 总计: ${TOTAL_H}h ${TOTAL_M}min"
log ""
log " 最终指标:"
log " L1: $L1_METRICS"
log " L2: $L2_METRICS"
log " L3: $L3_METRICS"
log ""
log " 日志: $LOGFILE"
log " 会话: $LOGDIR/bm_ledger.json"
log "=========================================="
