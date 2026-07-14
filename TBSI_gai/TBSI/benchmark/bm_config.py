"""
TBSI 二级训练测试基准 - 配置文件
=================================

二级渐进式验证:
  L1 Sprint (≤1h) → 快速验证 + 严格门控
  L2 Full   (~6h) → 全量训练确认最终结果

门控原则 (L1→L2, 必须全部满足):
  1. 复合指标提升 ≥ 1.0% (防单指标投机)
  2. 每序列胜率 > 55% (过半序列提升)
  3. 无核心指标降幅 > 1.0% (防局部退化)

多指标体系:
  - Overlap: SR(AUC), OP50, OP75, MeanIoU
  - Precision: PR, NPR
  - Robustness: WinRate (每序列对比)
  - Composite: 加权融合 0.30×SR + 0.15×OP50 + 0.25×PR + 0.15×NPR + 0.15×MeanIoU
"""

import os
from collections import OrderedDict

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "tbsi_track")
BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "benchmark")
LEDGERS_DIR = os.path.join(BENCHMARK_DIR, "ledgers")
PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "pretrained_networks")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# ============================================================
# 复合指标权重
# ============================================================
COMPOSITE_WEIGHTS = OrderedDict()
COMPOSITE_WEIGHTS["SR"] = 0.30
COMPOSITE_WEIGHTS["OP50"] = 0.15
COMPOSITE_WEIGHTS["PR"] = 0.25
COMPOSITE_WEIGHTS["NPR"] = 0.15
COMPOSITE_WEIGHTS["MeanIoU"] = 0.15

def compute_composite(metrics: dict) -> float:
    score = 0.0
    total_w = 0.0
    for key, w in COMPOSITE_WEIGHTS.items():
        if key in metrics and metrics[key] is not None:
            score += w * metrics[key]
            total_w += w
    return round(score / total_w, 2) if total_w > 0 else 0.0

# ============================================================
# Level 1: Sprint — 快速验证 (≤1h)
# ============================================================
LEVEL1 = OrderedDict()
LEVEL1["name"] = "sprint"
LEVEL1["desc"] = "Sprint: 4ep x 12k smpl, bs32, 100seq test, ~50min"
LEVEL1["config_name"] = "vitb_256_tbsi_sprint_da"
LEVEL1["epochs"] = 4
LEVEL1["sample_per_epoch"] = 12000
LEVEL1["batch_size"] = 32
LEVEL1["val_epoch_interval"] = 1
LEVEL1["test_epoch"] = 4
LEVEL1["test_subset_size"] = 100

# ============================================================
# Level 2: Full — 全量训练确认 (~6h)
# ============================================================
LEVEL2 = OrderedDict()
LEVEL2["name"] = "full"
LEVEL2["desc"] = "Full: 15ep x 60k smpl, bs32, 245seq full test, ~6h"
LEVEL2["config_name"] = "vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_da"
LEVEL2["epochs"] = 15
LEVEL2["sample_per_epoch"] = 60000
LEVEL2["batch_size"] = 32
LEVEL2["val_epoch_interval"] = 5
LEVEL2["test_epoch"] = 15
LEVEL2["test_subset_size"] = None  # 全量

# ============================================================
# L1→L2 门控阈值 (全部满足才通过)
# ============================================================
GATE_COMPOSITE_DELTA = 1.0          # 复合指标提升 ≥ 1.0%
GATE_WIN_RATE = 55.0                # 胜率 > 55%
GATE_MAX_SINGLE_DEGRADE = -1.0      # 任一核心指标降幅 ≤ 1.0% (即不能跌超过1%)

# ============================================================
# Level 1.5: Medium — 更大数据量验证新模块 (10ep x 36k, ~3h)
# ============================================================
MEDIUM = OrderedDict()
MEDIUM["name"] = "medium"
MEDIUM["desc"] = "Medium: 10ep x 36k smpl, bs32, 100seq test, ~3h"
MEDIUM["epochs"] = 10
MEDIUM["sample_per_epoch"] = 36000
MEDIUM["batch_size"] = 32
MEDIUM["val_epoch_interval"] = 1
MEDIUM["test_epoch"] = 10
MEDIUM["test_subset_size"] = 100

# ============================================================
# 基准等级列表
# ============================================================
LEVELS = [LEVEL1, LEVEL2]

# ============================================================
# 路径工具
# ============================================================
def get_checkpoint_path(config_name: str, epoch: int, save_dir: str = None) -> str:
    if save_dir is None:
        save_dir = OUTPUT_DIR
    return os.path.join(
        save_dir, "checkpoints", "train", "tbsi_track", config_name,
        f"TBSITrack_ep{epoch:04d}.pth.tar"
    )

def get_config_yaml_path(config_name: str) -> str:
    return os.path.join(EXPERIMENTS_DIR, f"{config_name}.yaml")

LEDGER_FILENAME = "bm_ledger.json"
