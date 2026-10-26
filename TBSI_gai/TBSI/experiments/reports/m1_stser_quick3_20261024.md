# M1 STSER real T2S quick3

## 目的与唯一变量

本实验验证候选一的核心信息流：在 `tbsi_e` 完成模板联合 MHSA 后，是否存在一个真实的 `256 search x 64 template` template-to-search 路径，能被 CENTER 训练目标驱动。parent 是受保护的 M1 epoch15 checkpoint；两臂只改变 T2S 算子，训练数据、seed、进度和评测清单一致。

- Plain arm: `M1_STSER_PLAIN_T2S_QUICK3_20261024`
- Candidate arm: `M1_STSER_QUICK3_20261024`
- Parent: `TBSI_E_MHSA_FULL_4090D_seed42` epoch15
- Training: RPP-LasHeR train 176 sequences, 10,112 samples/epoch, 3 epochs, 237 optimizer updates, batch 16 x accumulation 8, seed 42
- Test: locked RPP-v2 60 sequence manifest, FP32, `threads=0`, template update off
- Manifest SHA256: `ab587f801525c2e7e0fc7a8465fb90f3a2d99b67764eb2bc19cac2b422a938a8`

## 算子与合同

两臂均在 `TBSILayer` 的 `tbsi_e` 分支中、`V2TemplateFusion` 之后和 `tbsi_fuse_search` 之前执行。RGB search query 读取 TIR template key/value，TIR search query 读取 RGB template key/value；key/value 轴保留 64 个模板 token，并增加一个 null token。输出是有界的 search residual，外层 `gamma` 零初始化。

STSER 在 plain T2S 响应上增加 shared/private experts 和 null route，由每个 search candidate 的 query、跨模态消息和模态一致性产生三路 soft routing。该路由不改变 CENTER、size、offset 或 IoU loss。

已通过的前置合同：shape/device/dtype、零初始化 identity、单 batch forward/backward、`gamma` 非零梯度、完整模型随机输入 forward、M1 checkpoint 非严格加载和独立输出目录。M1 默认配置关闭新开关时未改变。

训练后的 residual strength 也确实被驱动：plain 臂 gamma=`[-1.44e-5, 8.86e-4]`，STSER 臂 gamma=`[-1.20e-3, -2.97e-4]`（RGB/TIR）。因此本次结果不是“模块未执行”或“只加载了 M1”的假阳性。

训练启动的第一次尝试因没有设置 `TBSI_RPP_TRAIN_LIST` 在数据加载前退出，无训练样本和 checkpoint；随后按既有 RPP quick3 runner 补齐清单后成功完成，失败日志和修正后的 provenance 均保留。

## 配对结果

| metric | plain mean | STSER mean | STSER - plain | 95% sequence bootstrap CI | wins/losses/ties |
|---|---:|---:|---:|---:|---:|
| AUC | 51.787 | 51.991 | +0.205 | [-0.503, +0.846] | 32/28/0 |
| Precision | 63.178 | 63.477 | +0.299 | [-0.741, +1.223] | 30/22/8 |
| Norm Precision | 60.344 | 60.468 | +0.124 | [-0.870, +0.951] | 32/26/2 |
| OP50 | 61.505 | 61.788 | +0.283 | [-0.596, +1.095] | 30/18/12 |
| OP75 | 43.824 | 43.720 | -0.105 | [-0.648, +0.386] | 27/29/4 |
| mean IoU | 0.52315 | 0.52512 | +0.00197 | [-0.00532, +0.00853] | 34/26/0 |

## 判断

结论为 `PROXY-COMPLETE / CONTRACT-PASSED / LEARNABILITY-PASSED / INITIAL-POSITIVE-DIRECTION / CI-CROSSES-ZERO`。STSER 相对 plain 的小幅正向均值说明 shared/private/null 路由有可用信号，但 AUC 未达到预设的 +0.5 点准入线，置信区间覆盖零，OP75 略负，不能称为稳定提升或 full-data 结论。

这次实验支持继续做一个单变量细化：保留真实 T2S 位置和 search candidate 路由，降低参数量并固定一个可解释的 route 约束，再做同一 60 序列 matched quick；不支持直接启动 full training，也不支持把 STSER 的小幅均值写成论文主结果。

## 证据路径

- Configs: `experiments/tbsi_track/M1_STSER_PLAIN_T2S_QUICK3_20261024.yaml`, `experiments/tbsi_track/M1_STSER_QUICK3_20261024.yaml`
- Module: `lib/models/layers/search_conditioned_t2s_router.py`
- Runner: `tools/run_m1_stser_quick3_20261024.sh`
- Analyzer: `tools/analyze_m1_stser_quick3_20261024.py`
- Provenance: `experiments/snapshots/m1_stser_quick3_20261024/provenance.txt`
- Summary: `output/experiments/M1_STSER_QUICK3_20261024/paired_rpp60_summary.json`
- Per-sequence: `output/experiments/M1_STSER_QUICK3_20261024/paired_rpp60_per_sequence.csv`
- Checkpoints: `output/experiments/M1_STSER_PLAIN_T2S_QUICK3_20261024/checkpoints/TBSITrack_ep0003.pth.tar`, `output/experiments/M1_STSER_QUICK3_20261024/checkpoints/TBSITrack_ep0003.pth.tar`
