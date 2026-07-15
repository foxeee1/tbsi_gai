"""
MiniLasHeR: A representative subset of LasHeR for rapid training/testing iteration.

Coverage:
  - 30 training sequences covering 5 challenge categories (6 each)
  - ~22K frames total (vs 734K full LasHeR = ~3%)
  - Training time: ~15 min (vs ~2.5h full LasHeR)

Usage:
  # In config YAML:
  DATA:
    TRAIN:
      DATASETS_NAME: ["MiniLasHeR_train"]
    VAL:
      DATASETS_NAME: ["MiniLasHeR_train"]

Design Principle:
  - Subclasses LasHeR to inherit all data augmentation and preprocessing.
  - Only difference: filters self.sequence_list to the MINI_SEQUENCES subset.
  - For reproducibility: each sequence's challenge category is documented.

Reference:
  - Full LasHeR: 979 sequences, ~734K frames
  - Mini LasHeR (train): 30 sequences, ~22K frames
  - Full training ~2.5h → Mini training ~15 min (single GPU)
"""

from .lasher import LasHeR

# =============================================================================
# MiniLasHeR Training Sequences: 30 from LasHeR training set
# =============================================================================
# Categories (6 sequences each, covering diverse tracking challenges):
#   1. Normal (易)       — Simple tracking scenarios with good conditions
#   2. Illumination (光照) — Night/dark/snow scenes (RGB degradation)
#   3. ThermalCross (热交叉) — IR whiteout/low-contrast (TIR degradation)
#   4. Occlusion (遮挡)   — Heavy/partial occlusion
#   5. FastMotion (快速)  — Fast motion / motion blur
# =============================================================================

MINI_TRAIN_SEQUENCES = [
    # ======== 1. Normal (易) — 基础跟踪场景 ========
    "blkboy",                  # 简单户外, 单人行走
    "whitegirl",               # 简单背景, 正常光照
    "redgirl",                 # 简单跟踪, 色彩鲜明
    "girlatleft",              # 简单场景, 人物移动
    "boy",                     # 简单, 基线场景
    "man",                     # 简单跟踪

    # ======== 2. Illumination (光照变化) — RGB 退化场景 ========
    "nightboy",                # 夜间场景, 低光照
    "nightrightboy1",          # 夜间场景, 右向移动
    "boyindarkwithgirl",       # 黑暗环境
    "boyinsnowfield2",         # 雪地/高反射, 过曝
    "boyinsnowfield_inf_white", # 雪地+IR白化
    "motolight",               # 光线变化环境

    # ======== 3. ThermalCross (热交叉) — TIR 退化场景 ========
    "boytoleft_inf_white",     # IR 全白/过曝
    "fogboyscoming1_quezhen_inf_heiying",  # 雾天+IR黑化
    "ab_hyalinepaperatground", # 透明物体 (IR难以分辨)
    "ab_leftfoam",             # 泡沫板 (低IR对比度)
    "ab_rightcupcoming_infwhite_quezhen",  # IR全白场景
    "ab_pingpongball",         # 小球+低纹理 (IR退化)

    # ======== 4. Occlusion (遮挡) — 目标被遮挡场景 ========
    "boybehindtrees",          # 树后遮挡
    "boybehindtrees2",         # 树后遮挡 (更复杂)
    "occludedmoto",            # 摩托车被遮挡
    "manaftercars",            # 车后行人
    "manaftertrees",           # 树后行人
    "boyunderthecolumn",       # 柱子遮挡

    # ======== 5. FastMotion (快速运动) — 运动模糊/高速目标 ========
    "boyrunning",              # 奔跑
    "leftrunningboy",          # 横向奔跑
    "motocross",               # 摩托车快速穿越
    "mototurn",                # 摩托车转弯
    "basketballatright",       # 篮球运动 (快速方向变化)
    "carleaves",               # 车辆驶离 (尺变+运动)
]

MINI_TEST_SEQUENCES = [
    # ======== 1. Normal (易) ========
    "1boycoming",              # 单人行走
    "2girl",                   # 简单多人
    "baggirl",                 # 正常跟踪
    "basketboy",               # 篮球场景
    "blackboy",                # 正常户外
    "bike",                    # 自行车

    # ======== 2. Illumination (光照) ========
    "belowdarkgirl",           # 暗光
    "boyfromdark",             # 从暗处走出
    "darkgirl",                # 暗光环境
    "girlfromlight_quezhen",   # 光照变化
    "carlightcome2",           # 车灯/光线变化
    "boyinsnowfield3",         # 雪地过曝

    # ======== 3. ThermalCross (热交叉) ========
    "ab_pingpongball2",        # 小球,低纹理,IR退化
    "ab_bolstershaking",       # 低对比度IR
    "ab_blkskirtgirl",         # 黑色裙子(IR困难)
    "ab_rightlowerredcup_quezhen",  # IR全白场景
    "ab_whiteboywithbluebag",  # 消融测试
    "ab_girlcrossroad",        # 路口,多干扰

    # ======== 4. Occlusion (遮挡) ========
    "boy2trees",               # 树后遮挡
    "boyaftertree",            # 树后穿过
    "carbehindtrees",          # 车在树后
    "boy2buildings",           # 建筑间遮挡
    "girlafterglassdoor",      # 玻璃门后
    "ab_bikeoccluded",         # 自行车被遮挡

    # ======== 5. FastMotion (快速运动) ========
    "10runone",                # 奔跑
    "boyruninsnow",            # 雪地奔跑
    "mototurneast",            # 摩托车转弯
    "bikeboyturntimes",        # 自行车转弯
    "runningcameragirl",       # 跟拍奔跑
    "boytakingbasketballfollowing",  # 篮球运动
]

# Total training frames: ~22,300
# Total test frames: ~18,000


class MiniLasHeR(LasHeR):
    """
    LasHeR subset for rapid training validation.

    Usage:
      MiniLasHeR(root=..., split='train')  → uses MINI_TRAIN_SEQUENCES
      MiniLasHeR(root=..., split='test')   → uses MINI_TEST_SEQUENCES

    Inherits all data loading, augmentation, and caching logic from LasHeR.
    """

    def __init__(self, root=None, image_loader=None, split=None, seq_ids=None, data_fraction=None,
                 mini_sequences=None):
        # Match LasHeR.__init__ signature so positional args (root, image_loader, split) are dispatched correctly
        super().__init__(root=root, image_loader=image_loader, split=split,
                         seq_ids=seq_ids, data_fraction=data_fraction)

        # Determine which mini set to use based on split
        user_provided = mini_sequences is not None
        if mini_sequences is None:
            is_test_split = split == 'test'
            mini_sequences = MINI_TEST_SEQUENCES if is_test_split else MINI_TRAIN_SEQUENCES
        else:
            is_test_split = split == 'test'

        # Retain only the mini subset
        available = set(self.sequence_list)
        self.sequence_list = [s for s in mini_sequences if s in available]
        missing = [s for s in mini_sequences if s not in available]
        if missing:
            print(f"[MiniLasHeR] WARNING: {len(missing)} requested sequence(s) "
                  f"not found in dataset: {missing}")

        # Rebuild class index and dir-listing cache for the subset
        self.seq_per_class = self._build_seq_per_class()
        self._pre_cache_dir_listings()

        # Log stats
        total_frames = sum(
            len(self._vis_files_cache.get(s, [])) for s in self.sequence_list
        )
        set_name = ("Test" if is_test_split else "Train") if not user_provided else "Custom"
        print(f"[MiniLasHeR/{set_name}] Initialized:")
        print(f"  Active sequences: {len(self.sequence_list)} / requested {len(mini_sequences)}")
        print(f"  Estimated frames: ~{total_frames}")
        print(f"  Categories: Normal, Illumination, ThermalCross, Occlusion, FastMotion")

    def get_name(self):
        return 'mini_lasher'
