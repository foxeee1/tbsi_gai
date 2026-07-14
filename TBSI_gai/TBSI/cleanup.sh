#!/bin/bash
# cleanup.sh — 清理残留进程和GPU缓存，每次训练测试前执行
# 用法: source cleanup.sh

echo "🔄 清理残留进程..."
kill -9 $(pgrep -f "multiprocessing.spawn\|run_training.py\|tracking/train.py\|tracking/test.py" 2>/dev/null) 2>/dev/null
sleep 2

echo "🔄 清理GPU缓存..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null

echo "🔄 检查GPU状态..."
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null

echo "✅ 环境干净"
