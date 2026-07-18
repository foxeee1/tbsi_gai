from easydict import EasyDict as edict
import yaml

"""
Add default config for TBSITrack.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = "mae_pretrain_vit_base.pth"
cfg.MODEL.EXTRA_MERGER = False

cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = [2, 5, 8, 11]

# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.MID_PE = False
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = 'ignore'

cfg.MODEL.BACKBONE.CE_LOC = []
cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []
cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = 'ALL'  # choose between ALL, CTR_POINT, CTR_REC, GT_BOX

# RGBT.BACKBONE
cfg.MODEL.BACKBONE.TBSI_LOC = []
cfg.MODEL.BACKBONE.RGB_ONLY = False
cfg.MODEL.BACKBONE.RGBT_UNSHARE = False

# Temporal Token Module (Section 3.1)
cfg.MODEL.TEMPORAL_TOKENS = False
cfg.MODEL.NUM_TEMPORAL_TOKENS = 4

# Temporal Token Module
cfg.MODEL.TEMPORAL_TOKENS = False
cfg.MODEL.NUM_TEMPORAL_TOKENS = 4
cfg.MODEL.STAGE2_BASELINE = ""  # path to baseline checkpoint for stage 2 fine-tuning

# Quality-Aware Fusion (spatial + channel gate + MADC + CSR)
cfg.MODEL.DEGRADATION_AWARE = False
cfg.MODEL.DA_MODE = 'channel'  # 'spatial' | 'channel'
cfg.MODEL.DA_MADC = False  # MADC: Modal Adaptive Distribution Calibration
cfg.MODEL.DA_CSR = False  # CSR: Complementary-Aware Spatial Reweighting
cfg.MODEL.DA_V2 = False   # DaFusionV2: 真正的质量感知融合 (多分辨率退化编码 + MoE原型路由)
cfg.MODEL.DA_FUSION_MODE = "residual"
cfg.MODEL.DA_FUSION_SCALE = 0.5
cfg.MODEL.DA_IN_LAYER = False  # Per-layer degradation-aware cross-attention modulation
cfg.MODEL.DGSFUSION = False   # DGSFusion: Divergence-Gated Specialized Fusion (内嵌TBSILayer)
cfg.MODEL.DGS_MODE = "v1"     # DGSFusion router mode: "v1"(6→1硬编码α) | "v2"(6→2自由α) | "v3"(attention熵) | "v4"(差异投影) | "v5"(自质量+差异投影) | "v6"(差异+模板对齐)

# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FREEZE_LAYERS = [0, ]
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False

cfg.TRAIN.CE_START_EPOCH = 20  # candidate elimination start epoch
cfg.TRAIN.CE_WARM_EPOCH = 80  # candidate elimination warm up epoch
cfg.TRAIN.DROP_PATH_RATE = 0.1  # drop path rate for ViT backbone

# TBSI
cfg.TRAIN.TBSI_DROP_RATE = 0.  # dropout rate for TransformerDecoderLayer
cfg.TRAIN.TBSI_DROP_PATH = []  # drop_path rate for TBSI Attention_st
cfg.TRAIN.SOT_PRETRAIN = False  # SOT pretraining with shared backbones
cfg.TRAIN.TEMPORAL_LR = None  # Independent LR for temporal token params (if set, overrides default grouping)
cfg.TRAIN.ROUTER_LR = None    # Independent LR for dgs_router params

cfg.TRAIN.BN_MOMENTUM = None  # BN momentum override (None = use default 0.1)
cfg.TRAIN.GRAD_ACCUM_STEPS = 1  # gradient accumulation steps
cfg.TRAIN.USE_CHECKPOINT = False  # gradient checkpointing for ViT blocks
cfg.TRAIN.TORCH_COMPILE = False  # torch.compile for ~20-30% speedup
cfg.TRAIN.PREFETCH_FACTOR = 4  # DataLoader prefetch factor
cfg.TRAIN.PERSISTENT_WORKERS = True  # Keep DataLoader workers alive between epochs
cfg.TRAIN.CHANNELS_LAST = False  # NHWC memory format for Tensor Cores
cfg.TRAIN.FUSED_OPTIMIZER = False  # Fused AdamW optimizer
cfg.TRAIN.SAVE_EPOCHS = []  # List of epochs to save checkpoints at (e.g. [3])

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"  # sampling methods
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.NUMBER = 1
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 500


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)
