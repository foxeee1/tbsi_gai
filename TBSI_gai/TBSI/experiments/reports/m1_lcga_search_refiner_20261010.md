# M1 LCGA search refiner 预实验（2026-10-10）

## 实验目标

在受保护 M1 `TBSI_E_MHSA_FULL_4090D_seed42` 的两路最终 search token 上、原始 `tbsi_fuse_search` 之前加入独立 RGB/TIR LCGA 局部细化，验证接口契约和训练可行性。M1 的 MHSA、TBSI-E、原始 search fusion、CENTER head、loss、配置和 checkpoint 均未覆盖。

## 接入方式

真实节点为 `TBSITrack.forward_fusion_only()`：从 `[B,640,768]` 提取 RGB/TIR 的 `[B,256,768]` search token，执行

```text
RGB -> LCGA_RGB ->
                     原 tbsi_fuse_search -> CENTER head
TIR -> LCGA_TIR ->
```

残差形式为 `x + alpha * (LCGA(x) - x)`，默认 `alpha=0`。两路不互相注意，不替换原融合层。新配置为 `M1_LCGA_SEARCH_PROXY_seed42.yaml`，输出目录为 `M1_LCGA_SEARCH_PREEXPERIMENT_20261010`。

用户提供的 F 盘论文源码在远程仓库不可访问；当前 `lcga.py` 是依据给定接口实现的紧凑 LCGA-style 局部窗口注意力和动态位置偏置版本，已保留 shape、分窗、shift 和残差契约，但不是论文源码的逐字复制。因此本阶段只能称为工程接入预实验，短训前仍应对照原始 LCGA 文件核查算子语义。

## T0-T4 结果

CUDA 和 CPU 两次串行检查均通过：

- shape：输入 batch 2、`[B,256,768]`，输出 shape 不变；
- zero-alpha：refiner tensor 和完整 `forward_fusion_only` 输出的 `max_abs_diff = 0.0`；
- 非零梯度：alpha、QKV、projection、dynamic position bias 梯度均有限且非零；CUDA alpha 梯度约 `2.00`，QKV 梯度约 `1.72e-5`；
- checkpoint：M1 权重加载时仅缺 16 个 `search_refiner.*` 新参数，无 unexpected key；
- 参数量：`4,724,370`；
- optimizer：`search_refiner` 与 `tbsi_fuse_search` 都在 optimizer，学习率均为 `1e-4`，没有被 `BACKBONE_MULTIPLIER=0.1` 降低。

CUDA 与 CPU 结果分别保存于 `output/experiments/M1_LCGA_SEARCH_PREEXPERIMENT_20261010/t0_t4_cuda.json` 和 `t0_t4_cpu.json`，串行日志合并在 `preexperiment.log`。

## T0-T4 阶段结论（A0/A1 启动前）

预实验支持继续做 matched short proxy，但**尚未证明 LCGA 能改善 M1**。目前只证明了代码可接入、零初始化不改变 M1、梯度可训练、optimizer 归属正确。尚未执行 A0/A1 的 3 epoch 训练、proxy 测试或逐序列比较，因此不能报告 AUC/PR/NPR、中心误差、响应峰距离或恢复率的改善。

当时计划串行执行同一 proxy manifest、seed=42、3 epoch 的：

1. `A0_M1_CONT`：legacy M1 matched continuation；
2. `A1_M1_LCGA_BOTH`：当前 LCGA 配置。

只有 A1 在局部定位指标和总指标上同时满足预注册门槛，才考虑 RGB-only/TIR-only 或 after-fusion 对照；若局部指标不改善或总体退化，应停止 full training。

## 代码和证据

- `lib/models/layers/lcga.py`
- `lib/models/layers/lcga_search_refiner.py`
- `tools/test_lcga_search_refiner.py`
- `tools/run_lcga_search_preexperiment_20261010.sh`
- `experiments/tbsi_track/M1_LCGA_SEARCH_PROXY_seed42.yaml`
- M1 checkpoint SHA256：`1302debc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`

当时状态：`PREEXPERIMENT-PASS / TRAINING-UNCONFIRMED`；后续 A0/A1 完成情况见下节。本阶段不启动 full training/full test，也不修改受保护 M1 资产。

## A0/A1 串行短训与训练侧留出评测（2026-10-10）

已运行 `bash tools/run_lcga_a0_a1_20261010.sh`，按 A0 训练/评测、A1 训练/评测、配对分析的顺序完成，终态为 `ALL_A0_A1_COMPLETE`。A0 关闭 `SEARCH_REFINER.ENABLE`，A1 开启；两份 YAML 的结构化配置除此字段外一致。两臂均从受保护 M1 epoch-15 checkpoint 继续 3 epoch，seed 42、batch 16、累积 8、worker 1、FP32，每臂 2016 个训练 batch；训练集是固定 RPP-v1 176 条序列，评测是无成员交叉的 LasHeR 训练侧留出 30 条序列。实际环境为 1× RTX 4090 D 24 GiB、Python 3.8.20、PyTorch 1.9.0+cu111、CUDA 11.1。由于未独立采集 GPU 活跃时长，GPUHours 为 NA。

| 指标 | A0 | A1 | A1-A0 |
|---|---:|---:|---:|
| AUC | 70.4966 | 71.6902 | +1.1936 |
| PR | 86.0304 | 87.5710 | +1.5407 |
| NPR | 82.6776 | 84.1569 | +1.4793 |
| OP50 | 84.1595 | 85.7012 | +1.5418 |
| OP75 | 69.2650 | 70.3173 | +1.0522 |
| FPS | 73.0440 | 57.4670 | -21.3% |

逐序列 AUC 为 13 胜、16 负、1 平；中位差约 -0.016。按序列重抽样的均值差 95% 区间为 [-0.743, +4.532]，跨零。`redgirl1497` 单条 AUC +44.228，是主要增益来源；去掉它，余下 29 条平均差为 -0.290。其他较大增益为 `rightgirlplayingphone` +6.713；较大退化为 `boysback` -5.058、`rightgirlbike` -2.657、`leftbasketball` -2.371 和 `leftcup` -2.309。A1 的 `alpha_rgb`/`alpha_tir` 从零更新至 -0.000378/+0.000290，且梯度诊断没有新增持续零梯度项，说明模块参与了优化；这不等于证明普遍有效。

**决策：不启动 full training/full test，局部空间建模的普遍改善尚未成立。** 这次 A0/A1 是可训练新模块的代理筛选，观察到的均值改善被单序列驱动且有明显速度代价。两臂同时更新了 backbone、原融合和 CENTER head，因而本比较只能评估“加入 LCGA-style refiner 后联合续训”的整体效果，**不能区分局部特征改善与 head 对新特征的适配**。若要回答该机制问题，下一步需在同一冻结特征/同一训练清单下设置局部模块与 head-only 的单变量对照，并预注册困难序列之外的判定标准；目前没有 head-only 对照结果。原论文 F 盘源码不可用，当前实现仅为 LCGA-style，语义等价性未确认。

代码快照 `output/experiments/M1_LCGA_A0_A1_PROXY_20261010/provenance/source_before_launch.tar.gz` SHA256 `afc44883d6a600a0c7d9769ad1815e8756b4f3e5c935fae87d744bf2d2962491`，Git HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`。A0/A1 配置 SHA256 分别为 `800ddeff3626a896ff3fab799f52e8357fbdf5e8c7d0d9c0a1d7c7d21162971d` 和 `25bcff9ba1ebb1a7b934e0ba54029a8eab1e3f036924b229c40d0369568b755c`；epoch-3 权重 SHA256 分别为 `98de0c2ab2c3ed21ddc48629cef4beaf2562b37198052c7f48e9b843d3fc99d2` 和 `584cc2ce405e360fc65ebc671805c919ffaa31fcea9c9ae557942e3180943696`。完整训练日志在各臂 `logs/lcga_a0_a1_train.log`，评测日志在 `output/experiments/M1_LCGA_A0_A1_PROXY_20261010/logs/`；原始 30 条结果在各臂独立的 `output/test/tracking_results/tbsi_track/` 目录，完整分析在 `output/experiments/M1_LCGA_A0_A1_PROXY_20261010/metrics/` 和 `comparison.json`。没有覆盖 M1 权重、配置或 test-244 归档。
