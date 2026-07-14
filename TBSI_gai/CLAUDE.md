# TBSI: Bridging Search Region Interaction With Template for RGB-T Tracking (CVPR 2023)

## Project Overview
TBSI is a dual-stream RGB-T tracking framework that bridges search region interaction with template features.
- Published at CVPR 2023: [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Hui_Bridging_Search_Region_Interaction_With_Template_for_RGB-T_Tracking_CVPR_2023_paper.pdf)
- Based on OSTrack foundation model
- Core innovation: TBSILayer with CASTBlock cross-attention between visible and infrared modalities
- Dual-stream ViT backbone with weight sharing between RGB and TIR modalities

## GPU Environment
- **Direct GPU access** (no SSH needed)
- **Conda/Python**: `/root/autodl-tmp/conda_envs/tbsi` (Python 3.8, PyTorch 1.9+cu111)
- **Code**: `/root/autodl-tmp/TBSI_gai/TBSI/`
- **Data**: `/root/autodl-tmp/TBSI_gai/data/lasher/`
- **Pretrained models**: `/root/autodl-tmp/TBSI_gai/pretrained/`
- **Note**: If `OMP_NUM_THREADS` error occurs, run `unset OMP_NUM_THREADS` first

## Training Commands

### ImageNet-1k pretrained backbone (default)
```bash
unset OMP_NUM_THREADS
python tracking/train.py --script tbsi_track --config vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --save_dir ./output --mode multiple --nproc_per_node 2
```

### SOT pretrained backbone
```bash
unset OMP_NUM_THREADS
python tracking/train.py --script tbsi_track --config vitb_256_tbsi_32x1_1e4_lasher_15ep_sot --save_dir ./output --mode multiple --nproc_per_node 2
```

### Single GPU (debug mode)
```bash
python tracking/train.py --script tbsi_track --config vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --save_dir ./output --mode single
```

## Evaluation Commands

```bash
# Test with ImageNet-1k pretrained model
python tracking/test.py tbsi_track vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --dataset_name lasher_test --threads 6 --num_gpus 1

# Test with SOT pretrained model
python tracking/test.py tbsi_track vitb_256_tbsi_32x1_1e4_lasher_15ep_sot --dataset_name lasher_test --threads 6 --num_gpus 1

# Analyze results
python tracking/analysis_results.py --tracker_name tbsi_track --tracker_param vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --dataset_name lasher_test
```

## Core Metrics
- **SR** (Success Rate): AUC of overlap success plot
- **PR** (Precision Rate): % frames within 20px center error
- **NPR** (Normalized Precision): Normalized precision

## TBSI Architecture

### Key Components
| Component | File | Description |
|-----------|------|-------------|
| **TBSITrack** | `lib/models/tbsi_track/tbsi_track.py` | Main model: ViT backbone + fusion conv + prediction head |
| **ViT-TBSI (CARE)** | `lib/models/tbsi_track/vit_tbsi_care.py` | Dual-stream ViT with TBSILayer insertion at specified layers |
| **TBSILayer** | `lib/models/layers/tbsi_layer.py` | Core innovation: 6 CASTBlock cross-attention modules |
| **CASTBlock** | `lib/models/layers/attn_blocks.py` | Cross-attention block for template-search interactions |
| **CenterPredictor** | `lib/models/layers/head.py` | Prediction head: center heatmap + size + offset regression |
| **BaseBackbone** | `lib/models/tbsi_track/base_backbone.py` | Abstract backbone with position embedding resizing |

### CASTBlock Cross-attention Patterns (TBSILayer)
1. `s2t_i2f` — Search-to-Template: infrared -> fused
2. `t2s_f2v` — Template-to-Search: fused -> visible
3. `s2t_v2f` — Search-to-Template: visible -> fused
4. `t2s_f2i` — Template-to-Search: fused -> infrared
5. `t2t_f2v` — Template-to-Template: fused -> visible
6. `t2t_f2i` — Template-to-Template: fused -> infrared

## Dataset: LasHeR
- **Location**: `/root/autodl-tmp/TBSI_gai/data/lasher/`
- **Training set**: ~979 sequences (in `trainingset/`)
- **Testing set**: ~245 sequences (in `testingset/`)
- **Sequence structure**: `sequence/visible/`, `sequence/infrared/`, `init.txt`

### Experiment Configurations

| Config | Pretrain | LR | Batch | Epochs | Search Size |
|--------|----------|-----|-------|--------|-------------|
| `vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k` | ImageNet-1k (deit_base) | 4e-4 | 32 | 15 | 256 |
| `vitb_256_tbsi_32x1_1e4_lasher_15ep_sot` | SOT (OSTrack) | 1e-4 | 32 | 15 | 256 |

## TBSI 两阶段流水线

一键跑完训练→测试→分析全流程（取代手动 Phase A+B）：

```
/tbsi-pipeline "<config_name>"
```

详见 [TBSI/TESTING_PIPELINE.md](TBSI/TESTING_PIPELINE.md)。核心原则：
- **一次只跑一个测试**，单进程模式（`--threads 0`），避免 `multiprocessing.spawn` 残留
- 训练/测试严格分离，训完再测
- 测试前自动清理残留进程，设置 `OMP_NUM_THREADS=4`

### 下游流程

```bash
# Phase C: ARIS Auto-Review
/auto-review-loop "TBSI CASTBlock cross-attention RGB-T tracking"

# Phase D: Paper Writing & Improvement
/paper-writing "NARRATIVE_REPORT.md" -- venue: CVPR
/auto-paper-improvement-loop "paper/" -- venue: CVPR
```

## 🛑 关键保护规则：严禁擅自删除实验记录与权重文件

**任何AI智能体（包括当前会话）在未获得用户明确手动确认前，严禁执行以下操作：**

### 禁止删除/覆盖/清理的目录与文件

| 类别 | 路径模式 | 说明 |
|------|----------|------|
| **训练权重/检查点** | `output/checkpoints/**`、`output/snapshots/**`、`*.pth`、`*.pt`、`*.ckpt` | 所有模型权重和训练中间检查点 |
| **实验日志** | `output/logs/**`、`logs/**`、`*.log`、`tensorboard/**` | 训练和测试日志文件 |
| **测试结果** | `output/test/**`、`test/**`、`results/**`、`analysis/**` | 评估结果和分析输出 |
| **实验记录** | `output/**`、`experiments/**`、`runs/**` | 整个输出目录树下的任何内容 |
| **配置文件** | `experiments/**/*.yaml`、`config/**/*.yaml` | 实验配置 |
| **预训练模型** | `pretrained/**`、`pretrain_models/**` | 下载或转换的预训练权重 |
| **数据** | `data/**` | 任何数据集文件 |

### 允许的操作（需谨慎）
- ✅ **读取/查看** 上述文件——始终允许
- ✅ **写入/追加** 新文件到现有目录——允许，不会覆盖已有内容即可
- ✅ **移动/重命名** ——必须先询问用户
- ❌ **删除/覆盖/清空** 上述任何路径——**绝对禁止**，除非用户逐条手动确认

### 执行原则
1. 当智能体不确定某操作是否涉及上述受保护文件时，**必须停下来询问**
2. 任何带有 `rm`、`rm -rf`、`mv`（覆盖模式）、清空目录、覆盖文件的命令，必须先列出受影响的文件并等待用户确认
3. 如果用户要求"清理"或"删除"，智能体必须先列出具体哪些文件会被影响，并逐条请用户确认
4. 此规则是**最高优先级**，覆盖任何其他优化、清理或自动化指令

## Available ARIS Skills
| Skill | Purpose | When |
|-------|---------|------|
| `/tbsi-pipeline` | 两阶段训练→测试流水线 | Train + Test cycle |
| `/auto-review-loop` | Autonomous review + fix loop | After training |
| `/auto-paper-improvement-loop` | Paper polish | After full paper |
| `/paper-writing` | Narrative.md → PDF | Final stage |
| `/analyze-results` | Experiment result analysis | After evaluation |
| `/research-lit` | Literature search | Research phase |
| `/run-experiment` | Deploy experiment to GPU | Training phase |
