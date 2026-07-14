#!/bin/bash
# ============================================================
# TBSI 基准环境激活脚本
# 用法: source benchmark/bm_env.sh
# ============================================================

# Python 环境
PYTHON="/root/autodl-tmp/conda_envs/tbsi/bin/python"
export PATH="/root/autodl-tmp/conda_envs/tbsi/bin:$PATH"
echo "[✓] Python: $($PYTHON --version 2>&1)"

# 验证 PyTorch
$PYTHON -c "import torch; print(f'[✓] Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.device_count()}')" 2>/dev/null

# 修复 OMP 错误
unset OMP_NUM_THREADS
echo "[✓] OMP_NUM_THREADS 已清除"

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[✓] 项目目录: $PROJ_DIR"

# 设置 PYTHONPATH
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"

echo "[✓] 环境就绪"

# 快捷命令别名
alias bm-train='$PYTHON tracking/train.py --script tbsi_track --save_dir ./output --mode single'
alias bm-test='$PYTHON tracking/test.py tbsi_track --dataset_name lasher_test --threads 6 --num_gpus 1'
alias bm-analyze='$PYTHON tracking/analysis_results.py --tracker_name tbsi_track --dataset_name lasher_test'
alias bm-run='$PYTHON benchmark/bm_run.py'
