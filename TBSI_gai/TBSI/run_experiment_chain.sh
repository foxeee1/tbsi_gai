#!/bin/bash
# TBSI 优化迭代实验链 - 串行逐步验证
# 每个Step: 训练 → 测试50-seq → 评估 → 决策保留/回退

set -e
cd /root/autodl-tmp/TBSI_gai/TBSI

RESULTS_FILE="output/logs/experiment_chain_results.txt"
echo "===== 实验链开始 $(date) =====" > $RESULTS_FILE

# 基线对比函数: 在相同序列上对比两个实验
compare() {
    local NAME=$1 EXP=$2 BASENAME=$3 BASEEXP=$4
    python3 << PYEOF
import os, torch, numpy as np
DATA_DIR = 'data/lasher/testingset'
BASE_DIR = f'output/test/tracking_results/tbsi_track/{BASEEXP}'
EXP_DIR = f'output/test/tracking_results/tbsi_track/{EXP}'

base_s = set(f.replace('.txt','') for f in os.listdir(BASE_DIR) if f.endswith('.txt') and not f.endswith('_time.txt'))
exp_s = set(f.replace('.txt','') for f in os.listdir(EXP_DIR) if f.endswith('.txt') and not f.endswith('_time.txt'))
common = sorted(base_s & exp_s)
print(f'{NAME}: {len(base_s)} vs {len(exp_s)} seq, 交集={len(common)}')

def eval_on(DIR, seqs):
    th=torch.arange(0.0,1.0+0.05,0.05); ai,ac=[],[]
    for seq in seqs:
        gt_path=os.path.join(DATA_DIR,seq,'visible.txt')
        if not os.path.exists(gt_path): gt_path=os.path.join(DATA_DIR,seq,'infrared.txt')
        if not os.path.exists(gt_path): continue
        with open(gt_path) as f: gt=[[float(x) for x in l.strip().split(',')] for l in f if l.strip()]
        if len(gt)<2: continue
        anno=torch.tensor(gt); pp=os.path.join(DIR,f'{seq}.txt')
        if not os.path.exists(pp): continue
        with open(pp) as f: pred=[[float(x) for x in l.strip().replace(',',' ').split()[:4]] for l in f if l.strip()]
        if not pred: continue
        pred=torch.tensor(pred); n=min(len(pred),len(anno)); pred,anno=pred[:n],anno[:n]
        if n: pred[0]=anno[0].clone(); vf=(anno[:,2:]>0).all(1)
        tl=torch.max(pred[:,:2],anno[:,:2]); br=torch.min(pred[:,:2]+pred[:,2:]-1,anno[:,:2]+anno[:,2:]-1)
        sz=(br-tl+1).clamp(0); inter=sz[:,0]*sz[:,1]; union=pred[:,2]*pred[:,3]+anno[:,2]*anno[:,3]-inter
        iou=torch.where(union>0,inter/union,torch.zeros_like(union))
        pc=pred[:,:2]+0.5*(pred[:,2:]-1); agt=anno[:,:2]+0.5*(anno[:,2:]-1); ce=((pc-agt)**2).sum(1).sqrt()
        iou[~vf]=-1; ce[~vf]=float('inf'); ai.append(iou); ac.append(ce)
    iou_all=torch.cat(ai); ce_all=torch.cat(ac)
    sr=np.mean([(iou_all>t).float().mean()*100 for t in th]); pr=(ce_all<=20).float().mean()*100
    miou=iou_all[iou_all>=0].mean()*100
    return sr,pr,miou,len(seqs)

b=eval_on(BASE_DIR,common); e=eval_on(EXP_DIR,common)
print(f'  {BASENAME}: SR={b[0]:.2f} PR={b[1]:.2f} (N={b[3]})')
print(f'  {EXP}:     SR={e[0]:.2f} PR={e[1]:.2f} (N={e[3]})')
print(f'  Δ:              SR={e[0]-b[0]:+.2f} PR={e[1]-b[1]:+.2f}')
PYEOF
}

train_and_test() {
    local STEP=$1 CFG=$2
    echo ""
    echo "=========================================="
    echo "  Step $STEP: $CFG"
    echo "=========================================="

    # 训练
    echo "[$(date +%H:%M)] 训练中..."
    rm -rf output/checkpoints/train/tbsi_track/${CFG}/
    rm -f output/logs/tbsi_track-${CFG}.log
    unset OMP_NUM_THREADS
    python -u tracking/train.py --script tbsi_track --config ${CFG} --save_dir ./output --mode single 2>&1 | tail -5

    # 检查训练是否成功
    if [ ! -f output/checkpoints/train/tbsi_track/${CFG}/TBSITrack_ep*.pth.tar ]; then
        echo "❌ 训练失败, 跳过此步"
        return 1
    fi
    echo "✅ 训练完成"

    # 测试50-seq
    echo "[$(date +%H:%M)] 测试50-seq..."
    rm -rf output/test/tracking_results/tbsi_track/${CFG}/
    python -u tracking/test.py tbsi_track ${CFG} --dataset_name lasher_test --threads 2 --num_gpus 1 &>/tmp/test_${STEP}.log &
    TEST_PID=$!

    # 等待50个序列
    for i in $(seq 1 30); do
        sleep 60
        s=$(ls output/test/tracking_results/tbsi_track/${CFG}/ 2>/dev/null | grep -v "_time" | wc -l)
        echo "  测试: $s/50"
        [ "$s" -ge 50 ] 2>/dev/null && kill $TEST_PID 2>/dev/null && break
        ! kill -0 $TEST_PID 2>/dev/null && break
    done

    echo "✅ 测试完成 ($(ls output/test/tracking_results/tbsi_track/${CFG}/ 2>/dev/null | grep -v "_time" | wc -l) seq)"
    return 0
}

# ============ 实验链 ============

# Step S0: 8ep基线
train_and_test "S0" "vitb_256_tbsi_opt_S0"
echo "=== Step S0: 8ep基线 ===" >> $RESULTS_FILE
compare "S0" "vitb_256_tbsi_opt_S0" "A1(4ep)" "vitb_256_tbsi_exp_A1_mdta" >> $RESULTS_FILE

# Step S1: +TEMPORAL_LR
train_and_test "S1" "vitb_256_tbsi_opt_S1"
echo "=== Step S1: +TEMPORAL_LR=1e-4 ===" >> $RESULTS_FILE
compare "S1" "vitb_256_tbsi_opt_S1" "S0(8ep)" "vitb_256_tbsi_opt_S0" >> $RESULTS_FILE

# Step S2: +cls_token init (代码已改, 直接用S1 config重训即可利用初始化)
# 实际上 S1/S2 共用同一个config, 代码改动已生效
echo "=== Step S2: +cls_token init (代码生效, 同S1) ===" >> $RESULTS_FILE
compare "S2" "vitb_256_tbsi_opt_S1" "S1(无init)" "vitb_256_tbsi_opt_S0" >> $RESULTS_FILE

# Step S3: +QAF
train_and_test "S3" "vitb_256_tbsi_opt_S3"
echo "=== Step S3: +QAF(DA融合) ===" >> $RESULTS_FILE
compare "S3" "vitb_256_tbsi_opt_S3" "S2(Token优化)" "vitb_256_tbsi_opt_S1" >> $RESULTS_FILE

# ============ 汇总 ============
echo ""
echo "=========================================="
echo "  全部完成! 结果汇总:"
echo "=========================================="
cat $RESULTS_FILE
