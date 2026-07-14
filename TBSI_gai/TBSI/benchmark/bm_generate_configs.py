"""
TBSI 基准 - 生成各级别配置文件
================================
基于原始 SOT 配置生成：
  - L1 Sprint (快速验证)
  - L2 Full (全量)
  - Medium Channel (方向A, 10ep x 36k)
  - Medium Compensate (方向C, 10ep x 36k)
"""

import os, sys, copy
prj_path = os.path.join(os.path.dirname(__file__), "..")
if prj_path not in sys.path:
    sys.path.append(prj_path)

from benchmark.bm_config import EXPERIMENTS_DIR, LEVEL1, LEVEL2, MEDIUM, get_config_yaml_path
import yaml

ORIGINAL_SOT_CONFIG = "vitb_256_tbsi_32x1_1e4_lasher_15ep_sot"

def load_sot_config() -> dict:
    path = get_config_yaml_path(ORIGINAL_SOT_CONFIG)
    if not os.path.exists(path):
        raise FileNotFoundError(f"基础配置文件未找到: {path}")
    with open(path) as f:
        return yaml.safe_load(f)

def make_config(level_cfg: dict, da_mode: str = None, config_name: str = None,
                da_in_layer=False, use_attn_gate=False) -> dict:
    """从 SOT 配置派生，覆盖 level 参数和融合模式"""
    config = copy.deepcopy(load_sot_config())
    config["TRAIN"]["EPOCH"] = level_cfg["epochs"]
    config["TRAIN"]["BATCH_SIZE"] = level_cfg["batch_size"]
    config["TRAIN"]["VAL_EPOCH_INTERVAL"] = level_cfg["val_epoch_interval"]
    config["DATA"]["TRAIN"]["SAMPLE_PER_EPOCH"] = level_cfg["sample_per_epoch"]
    if level_cfg.get("test_epoch"):
        config["TEST"]["EPOCH"] = level_cfg["test_epoch"]
    config["MODEL"]["PRETRAIN_FILE"] = "TBSITrack_SOT_Pretrained.pth.tar"
    config["TRAIN"]["SOT_PRETRAIN"] = True
    if da_mode:
        config["MODEL"]["DEGRADATION_AWARE"] = True
        config["MODEL"]["DA_MODE"] = da_mode
    if da_in_layer:
        config["MODEL"]["DA_IN_LAYER"] = True
    if use_attn_gate:
        config["MODEL"]["ATTN_GATE"] = True
    return config

def write_config(config: dict, config_name: str) -> str:
    path = get_config_yaml_path(config_name)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  [✓] 已生成: {path}")
    return path

def generate_all_configs():
    print("=" * 60)
    print("TBSI 基准配置生成")
    print("=" * 60)
    generated = []

    # L1 Sprint (spatial mode)
    print(f"\n[L1 Sprint] {LEVEL1['desc']}")
    cfg = make_config(LEVEL1, da_mode='spatial')
    generated.append(write_config(cfg, LEVEL1["config_name"]))

    # L2 Full (spatial mode)
    print(f"\n[L2 Full] {LEVEL2['desc']}")
    cfg2 = make_config(LEVEL2, da_mode='spatial')
    generated.append(write_config(cfg2, LEVEL2["config_name"]))

    # Medium - Channel (方向A)
    print(f"\n[Medium-Channel] {MEDIUM['desc']}")
    cfg3 = make_config(MEDIUM, da_mode='channel',
                       config_name=f"vitb_256_tbsi_medium_channel")
    generated.append(write_config(cfg3, f"vitb_256_tbsi_medium_channel"))

    # Medium - Compensate (方向C)
    print(f"\n[Medium-Compensate] {MEDIUM['desc']}")
    cfg4 = make_config(MEDIUM, da_mode='compensate',
                       config_name=f"vitb_256_tbsi_medium_compensate")
    generated.append(write_config(cfg4, f"vitb_256_tbsi_medium_compensate"))

    # Sprint - LayerDA (实验1: 层内退化调制, L1 Sprint)
    print(f"\n[Sprint-LayerDA] {LEVEL1['desc']}")
    cfg5 = make_config(LEVEL1, da_mode='spatial', da_in_layer=True,
                       config_name=f"vitb_256_tbsi_sprint_layerda")
    generated.append(write_config(cfg5, f"vitb_256_tbsi_sprint_layerda"))

    # Sprint - PixelDA (实验2: 像素级质量估计, L1 Sprint)
    print(f"\n[Sprint-PixelDA] {LEVEL1['desc']}")
    cfg6 = make_config(LEVEL1, da_mode='pixel',
                       config_name=f"vitb_256_tbsi_sprint_pixelda")
    generated.append(write_config(cfg6, f"vitb_256_tbsi_sprint_pixelda"))

    # Sprint - Direction1 (stat): 特征统计自适应校准
    print(f"\n[Sprint-D1-stat] {LEVEL1['desc']}")
    cfg7 = make_config(LEVEL1, da_mode='stat',
                       config_name="vitb_256_tbsi_sprint_d1_stat")
    generated.append(write_config(cfg7, "vitb_256_tbsi_sprint_d1_stat"))

    # Sprint - Direction2 (gauss): 可学习高斯中心先验
    print(f"\n[Sprint-D2-gauss] {LEVEL1['desc']}")
    cfg8 = make_config(LEVEL1, da_mode='gauss',
                       config_name="vitb_256_tbsi_sprint_d2_gauss")
    generated.append(write_config(cfg8, "vitb_256_tbsi_sprint_d2_gauss"))

    # Sprint - Direction3 (agate): 注意力输出门控
    print(f"\n[Sprint-D3-agate] {LEVEL1['desc']}")
    # D3 doesn't use DegradationAwareFusion, it uses attn gate in CASTBlock
    cfg9 = copy.deepcopy(load_sot_config())
    cfg9['TRAIN']['EPOCH'] = LEVEL1['epochs']
    cfg9['TRAIN']['BATCH_SIZE'] = LEVEL1['batch_size']
    cfg9['TRAIN']['VAL_EPOCH_INTERVAL'] = LEVEL1['val_epoch_interval']
    cfg9['DATA']['TRAIN']['SAMPLE_PER_EPOCH'] = LEVEL1['sample_per_epoch']
    cfg9['TEST']['EPOCH'] = LEVEL1['test_epoch']
    cfg9['MODEL']['PRETRAIN_FILE'] = 'TBSITrack_SOT_Pretrained.pth.tar'
    cfg9['TRAIN']['SOT_PRETRAIN'] = True
    cfg9['MODEL']['ATTN_GATE'] = True
    generated.append(write_config(cfg9, "vitb_256_tbsi_sprint_d3_agate"))

    print(f"\n共 {len(generated)} 个配置文件")
    return generated

if __name__ == "__main__":
    generate_all_configs()
