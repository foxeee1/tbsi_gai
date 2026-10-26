# M1 AMCM 表示对比监督 full-244 实验（2026-10-18）

## 实验合同

- Parent：`M1-AMCM-HANN-FAILURE-PAIRWISE-20261018`；唯一新增变量是 AMCM search representation 的 target/background margin loss。
- 配置：`experiments/tbsi_track/M1_AMCM_DR_REPR_CONTRASTIVE_PROXY_20261018.yaml`，seed 42，LasHeR train 256 samples/epoch，3 epochs，FP32，batch 2，worker 1。
- 检测：full `lasher_test` 244/244 序列，checkpoint `TBSITrack_ep0003.pth.tar`，代码 commit `09f3d57c0761b01de1fa3aa29ff296c20ea5e9`。
- 目标：检验 feature-level target/background supervision 是否直接改善目标表示并减少背景峰误选。

## 全量结果

| 方法 | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| M1 fixed full-244 | 56.21 | 68.33 | 47.13 | 70.21 | 66.20 |
| 表示对比监督候选 full-244 | 56.41 | 68.64 | 47.50 | 70.44 | 66.60 |
| 相对 M1 | +0.20 | +0.31 | +0.37 | +0.23 | +0.40 |

该增益属于单 seed、3 epoch 的小幅观察，不能表述为稳定提升。当前最佳 Hann-aware pairwise parent 的 `66.84` AUC 是 difficult5 proxy 结果，未做同口径 full-244，不能与本表直接混合比较。

## 梯度、通路和特征变化

- `Loss/amcm_repr` 在训练/验证日志中持续出现，约 `0.066--0.073`，说明表示损失被计算并回传。
- `amcm_gradient_flow.csv`：24 个记录点，平均 AMCM 梯度范数 `0.09544`，范围 `0.05150--0.099999`；16/16 参数组均有非零梯度。
- `feature_changes.csv`：平均 RGB residual norm `6.282`、TIR `5.721`；平均输入 norm 分别 `224.693`、`204.663`，对应约 `2.80%`、`2.80%` 的相对变化。
- 平均 reliability 为 RGB `0.4596`、TIR `0.5404`，只表现为轻微模态偏置；日志中的 gamma 为零，未形成额外后置门控通路。
- 激活的实际链路是 `search RGB/TIR -> AMCM directed residual(pre_fusion) -> tbsi_fuse_search -> CENTER head`，template prototype 仅用于训练期表示损失；推理时没有新增状态更新或 reranking。

## 归因

已支持：新表示损失实现有效，梯度和参数更新存在，残差确实进入融合链路；full-244 相对 M1 的五项指标均为正。

尚未支持：目标/背景表示分离已经改善峰值选择。此前 fixed-state trace 的离散 rank 几乎不变，本实验尚未记录出稳定的 feature target/background margin 增长；因此 +0.20 AUC 更可能来自连续回归和闭环轨迹的微调，而不是已证实的背景峰翻转。

## 逐序列变化

按 full-244 average-overlap（AO）比较 M1 与候选，低性能集合（任一方法 AO < 45）中，明显改善的是 `ab_bikeoccluded`（10.27 -> 77.58）、`3bike1`（10.72 -> 68.57）、`leftfarboycomingpicktheball`（19.74 -> 62.82）、`girlrightthewautress`（25.08 -> 54.94）和 `whiteboyrightcoccergoal`（39.89 -> 59.82）。明显退化的是 `whiteridingbike`（40.75 -> 14.10）、`rightblkfatboyleftwhite`（69.69 -> 43.73）、`shinycarcoming`（39.19 -> 13.27）、`leftexcersicebookyellow`（58.20 -> 40.45）和 `redmidboy`（31.42 -> 17.58）。这说明增益和退化都集中在低 AO/背景相似的序列，不能归因于全局稳定的峰值校正。

此前 difficult5 的逐帧峰值审计仍是失败机理证据：`manoncall` 目标峰 0.2293、背景峰 0.2904；`boyss` 0.2533/0.4654；`girldownstairfromlight` 0.1108/0.3384；`bike2left` 0.0607/0.3385。表示监督候选尚未补出这些序列的 feature margin 与 rank 轨迹，因此这些峰值只能作为后续 target-conditioned prototype 实验的诊断基线。

主要限制：prototype 使用 template token 均值，正样本为 GT overlap cells，负样本为 GT 外最高相似度 cell；该监督没有显式建模 hard background 的空间/时序身份，且 3 epoch proxy 可能不足以改变 backbone 表示。full-244 结果只支持“在当前合同下有小幅改善”，不支持直接推广到多 seed 或正式训练。

## 下一步判定

`SUPPORTS_MINIMAL_REPRESENTATION_EFFECT / TARGET-BACKGROUND-SEPARATION-UNCONFIRMED`。下一步应保持注入位置和 CENTER/state 不变，只替换 prototype/负样本构造为 target-conditioned token prototype，并记录每帧 target/background cosine margin、Hann rank 和 residual ratio；若 margin 仍不变，应回到监督与 prototype 信息瓶颈，而不是继续增加融合分支。

证据目录：

- 训练日志与 CSV：`output/experiments/M1_AMCM_DR_REPR_CONTRASTIVE_PROXY_20261018/`
- full-244 tracking：`output/test/tracking_results/tbsi_track/M1_AMCM_DR_REPR_CONTRASTIVE_PROXY_20261018/`
- 汇总命令：`python tracking/analysis_results.py --tracker_name tbsi_track --tracker_param M1_AMCM_DR_REPR_CONTRASTIVE_PROXY_20261018 --dataset_name lasher_test`
