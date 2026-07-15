#!/bin/bash
# ==============================================================================
# MiniLasHeR 快速训练测试流水线
# ==============================================================================
# 功能: 一键完成 MiniLasHeR 训练 → 测试 → 基准采集
# 用法:
#   # 采集参考基准 (在 main 分支上)
#   bash tools/run_mini_pipeline.sh --prefix reference
#
#   # 采集实验基准 (在 fix/dev 分支上)
#   bash tools/run_mini_pipeline.sh --prefix exp_fix_detach
#
#   # 采集并对比
#   bash tools/run_mini_pipeline.sh --prefix exp_phase1 --compare ref.json
# ==============================================================================

set -e

# ---- Defaults ----
CONFIG="vitb_256_tbsi_32x4_4e4_miniLasHeR_15ep"
OUTPUT_DIR="./output"
PREFIX=""
COMPARE=""

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --compare) COMPARE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash tools/run_mini_pipeline.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config NAME    Config name (default: $CONFIG)"
            echo "  --prefix NAME    Benchmark save prefix (required)"
            echo "  --output DIR     Output dir (default: $OUTPUT_DIR)"
            echo "  --compare FILE   Compare against this baseline JSON after collection"
            echo "  -h, --help       Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$PREFIX" ]; then
    echo "ERROR: --prefix is required"
    echo "Usage: bash tools/run_mini_pipeline.sh --prefix NAME"
    exit 1
fi

echo ""
echo "============================================================"
echo "  TBSI MiniLasHeR Pipeline"
echo "============================================================"
echo "  Config:   $CONFIG"
echo "  Prefix:   $PREFIX"
echo "  Output:   $OUTPUT_DIR"
echo "  Compare:  ${COMPARE:-none}"
echo "============================================================"
echo ""

# ---- Step 0: Environment ----
cd "$(dirname "$0")/.."  # Move to TBSI/ root
export PYTHONPATH="$PWD:$PYTHONPATH"
unset OMP_NUM_THREADS

echo "Working directory: $(pwd)"
echo ""

# ---- Step 1: Kill residual processes ----
echo "--- Cleaning residual processes ---"
pkill -f "multiprocessing.spawn" 2>/dev/null || true
sleep 2

# ---- Step 2: Run benchmark collection (train + eval + diagnostics) ----
echo ""
echo "============================================================"
echo "  Step 1/2: Training + Benchmark Collection"
echo "============================================================"
echo ""

python tools/run_mini_benchmark.py \
    --config "$CONFIG" \
    --save_prefix "$PREFIX" \
    --output_dir "$OUTPUT_DIR"

BENCHMARK_FILE="$OUTPUT_DIR/benchmark/${PREFIX}_benchmark.json"

if [ ! -f "$BENCHMARK_FILE" ]; then
    echo ""
    echo "ERROR: Benchmark file not found at $BENCHMARK_FILE"
    echo "Training may have failed."
    exit 1
fi

echo ""
echo "============================================================"
echo "  Step 2/2: Results Summary"
echo "============================================================"
echo ""

# Print key metrics from benchmark
python3 -c "
import json
with open('$BENCHMARK_FILE') as f:
    data = json.load(f)

print('  Benchmark Summary:')
print(f'  Config:    {data[\"meta\"][\"config\"]}')
print(f'  Train:     {data[\"meta\"][\"train_elapsed_s\"]:.0f}s')
print(f'  Health:    {data[\"health\"][\"summary\"]}')
print()

# Loss
ls = data.get('loss_summary', {})
if ls:
    epochs = sorted(ls.keys())
    first = ls[epochs[0]]['Loss/total_mean']
    last = ls[epochs[-1]]['Loss/total_mean']
    print(f'  Loss:      {first:.4f} → {last:.4f} (epochs {epochs[0]}-{epochs[-1]})')

# Eval
em = data.get('eval_metrics', {})
if em:
    print(f'  Eval AUC:  {em.get(\"AUC\", \"N/A\")}')
    print(f'  Eval PR:   {em.get(\"Precision\", \"N/A\")}')
    print(f'  Eval NPR:  {em.get(\"Norm Precision\", \"N/A\")}')
    print(f'  OP50:      {em.get(\"OP50\", \"N/A\")}')
    print(f'  OP75:      {em.get(\"OP75\", \"N/A\")}')

# Grad health
gs = data.get('gradient_summary', {})
if gs.get('da_fusion_mean'):
    dm = gs['da_fusion_mean']
    print(f'  DaFusion grad: {dm.get(\"final\", \"N/A\")}')
if gs.get('degradation_mod_mean'):
    dm = gs['degradation_mod_mean']
    print(f'  DegradMod grad: {dm.get(\"final\", \"N/A\")}')
if gs.get('tbsi_layer_mean'):
    dm = gs['tbsi_layer_mean']
    print(f'  TBSILayer grad: {dm.get(\"final\", \"N/A\")}')
"

# ---- Step 3: Compare against baseline (optional) ----
if [ -n "$COMPARE" ]; then
    echo ""
    echo "============================================================"
    echo "  Step 3/2: Comparison with Baseline"
    echo "============================================================"
    echo ""

    python tools/compare_benchmark.py \
        --baseline "$COMPARE" \
        --experiment "$BENCHMARK_FILE"
fi

echo ""
echo "============================================================"
echo "  MiniLasHeR Pipeline Complete!"
echo "============================================================"
echo "  Benchmark: $BENCHMARK_FILE"
echo ""
echo "  Next steps:"
echo "    - Compare with baseline:"
    echo "      python tools/compare_benchmark.py \\"
    echo "        --baseline output/benchmark/reference_benchmark.json \\"
    echo "        --experiment $BENCHMARK_FILE"
echo ""
echo "    - Run full evaluation:"
echo "      python tracking/analysis_results.py \\"
echo "        --tracker_name tbsi_track \\"
echo "        --tracker_param $CONFIG \\"
echo "        --dataset_name mini_lasher_test"
echo "============================================================"
