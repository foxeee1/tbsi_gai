# M1-S1 接入判断方法 full-244 对照（2026-10-14）

## 问题与实验合同

检验两种决策逻辑能否在保留 S1 offset correction 收益的同时，减少其在未筛选序列上的退化。三臂均使用受保护 M1 epoch15 checkpoint、冻结 S1 offset predictor、`rho=0.5`、utility threshold `0.08`、LasHeR official testingset 244 序列、FP32 和串行 tracker。唯一方法变量是 offset action 的 admission 逻辑：同代码 S1 control、train-side selective-risk gate、constant-velocity motion-innovation gate。没有重新训练 M1、S1 predictor 或 backbone；这是 train-side gate calibration + full-244 tracking。

72 条 LasHeR train-side 序列的 144 个固定状态用于校准。测试集没有参与拟合。两种 gate 的序列留出 OOF 有害率均为 18.18%；selective-risk 放行 22/144 个状态（15.28%），motion-innovation 放行 66/144（45.83%）。

## 全量结果

官方 evaluator 对 244 条序列汇总的指标如下。S1 control 与历史 S1 run1031 五项完全一致，决策分支关闭时 parity 成立。

| 方法 | AUC / Success | Precision | Norm Precision | OP50 | OP75 | AUC 相对 control（胜/负/平） |
|---|---:|---:|---:|---:|---:|---:|
| M1 固定 parent | 56.2077 | 70.2073 | 66.1980 | 68.3339 | 47.1349 | - |
| S1 control | 56.4599 | 70.5443 | 66.5330 | 68.7700 | 47.1488 | - |
| S1 + selective risk | 55.8636 | 69.7881 | 65.8215 | 67.8977 | 46.9464 | 115 / 128 / 1 |
| S1 + motion innovation | 55.8404 | 69.7540 | 65.8606 | 67.9598 | 46.6718 | 102 / 112 / 30 |

相对 S1 control，selective-risk 五项变化为 `-0.5963/-0.7561/-0.7115/-0.8722/-0.2023`；motion-innovation 为 `-0.6196/-0.7903/-0.6724/-0.8101/-0.4770`。两种 gate 的五项总指标都退化，且都低于 M1。故它们没有达到减少全量净损失的目标。

三臂都处理 219,048 帧。S1 control 的 offset/no-op 数为 87,212/131,836；selective-risk 为 16,900/202,148；motion-innovation 为 69,059/149,989。风险 gate 大幅减少介入，motion gate 保留较多介入，但两者的指标变化方向近似，说明单纯压低介入覆盖并不足以保证动作正确。

## Cohort 对照

以下为逐序列等权的平均指标差，单位为百分点。每格格式为 `AUC / PR / NPR / OP50 / OP75`；`vs M1` 衡量全量绝对收益，`vs S1` 衡量 gate 对 S1 control 的净作用。成员清单沿用 2026-10-12 冻结的 low-AUC62 与 normal20 sentinel，remaining162 是其余测试序列。

| Cohort | S1 control vs M1 | Selective risk vs M1 | Selective risk vs S1 | Motion innovation vs M1 | Motion innovation vs S1 |
|---|---|---|---|---|---|
| low-AUC62 (62) | +3.7274 / +5.0960 / +4.7513 / +5.0720 / +3.0750 | +1.1105 / +1.6333 / +1.5563 / +1.4085 / +1.1054 | -2.6169 / -3.4627 / -3.1949 / -3.6634 / -1.9696 | +1.1469 / +1.7395 / +1.8241 / +1.6823 / +0.9735 | -2.5805 / -3.3564 / -2.9271 / -3.3897 / -2.1015 |
| normal20 sentinel (20) | +0.0030 / +0.0205 / +0.0125 / +0.0059 / -0.0426 | -0.1308 / -0.0755 / -0.1543 / -0.1787 / -0.0602 | -0.1338 / -0.0961 / -0.1668 / -0.1846 / -0.0176 | +0.0153 / -0.0459 / +0.0407 / -0.0153 / -0.0578 | +0.0123 / -0.0665 / +0.0282 / -0.0212 / -0.0152 |
| remaining162 (162) | -1.0470 / -1.4453 / -1.3153 / -1.2850 / -1.1507 | -0.9271 / -1.2471 / -1.1436 / -1.1739 / -0.6995 | +0.1199 / +0.1982 / +0.1717 / +0.1111 / +0.4512 | -0.9941 / -1.3428 / -1.2113 / -1.2053 / -1.0630 | +0.0529 / +0.1025 / +0.1040 / +0.0797 / +0.0877 |
| all244 | +0.2522 / +0.3370 / +0.3351 / +0.4361 / +0.0139 | -0.3441 / -0.4192 / -0.3764 / -0.4362 / -0.1885 | -0.5963 / -0.7561 / -0.7115 / -0.8722 / -0.2023 | -0.3673 / -0.4533 / -0.3374 / -0.3741 / -0.4631 | -0.6196 / -0.7903 / -0.6724 / -0.8101 / -0.4770 |

Selective-risk 与 motion-innovation 相对 S1 control 均略微改善 remaining162 的序列均值，但仍保留约 `-0.93` 和 `-0.99` 的 remaining162 AUC 差（相对 M1）。它们同时从 low-AUC62 收益中分别削去约 `2.62` 和 `2.58` 个 AUC 点。normal20 只有 20 条预选 sentinel，不能代表其余正常/未筛选序列。

## 逐序列变化

相对 S1 control，selective-risk 的全量 AUC 最大增益为 `umbrellawillopen` `+64.11`、`3rdfatboy` `+54.19`、`bluegirlbiketurn` `+29.69`、`right5thflag` `+23.93`、`whiteridingbike` `+22.80`；最大退化为 `ab_bikeoccluded` `-60.56`、`bikefromlight` `-59.77`、`3bike1` `-56.49`、`runningcameragirl` `-31.70`、`4thboywithwhite` `-29.01`。

Motion-innovation 的最大增益为 `umbrellawillopen` `+64.23`、`AQgirlwalkinrain` `+14.13`、`rightbottlecomes` `+10.75`、`large` `+10.39`、`7rightorangegirl` `+8.31`；最大退化为 `bikefromlight` `-59.88`、`ab_bikeoccluded` `-54.57`、`runningcameragirl` `-31.38`、`right2ndflagformath` `-27.81`、`rightboy504` `-21.32`。完整 244 序列、五项指标及各候选相对 M1/control 的逐条变化见 `evaluation/per_sequence_official.csv`，不会用极端序列例子代替总体统计。

## 归因与路线

- **已排除**：这两种单独接入的 admission gate 能把 S1 变成全量优于 M1 的方法；五项总指标均低于 S1 control 和 M1。
- **已支持**：S1 的 full-set 退化不只是“介入次数太多”。Selective-risk 把 offset coverage 从 39.82% 降到 7.72%，motion-innovation 降到 31.53%，但两者总体仍明显退化。
- **暂不支持**：训练状态上的风险预测或恒速运动残差能可靠识别闭环中有害的 offset。remaining162 的相对 S1 小幅改善不足以恢复其相对 M1 的负收益，且 low-AUC62 收益明显变小。
- **尚未区分**：失败主要来自 admission 特征分布外推、offset 方向/幅度错误，还是动作改变后闭环状态漂移。full-244 序列指标本身不能给出固定状态的因果归因。

**建议停止扩展这两种 gate，也暂不再叠加门控特征。**下一步做轻量 train-side fixed-state 审计：在有代表性的 S1 负效序列与正效序列上，对相同状态/相同 crop 配对测量 no-op 和 S1 offset 的真实单步 IoU utility，同时记录两种 gate 的 admission score 与动作方向/幅度。关键问题是 gate score 是否能在序列留出条件下区分正负 utility。如果不能，停止改 admission；如果能而 full闭环仍退化，再把焦点转向状态漂移与闭环恢复，而不是融合算子复杂度。测试集 full-244 结果只用于这次候选判别，不应再据此拟合阈值或挑选下一 gate。

## 可比性与证据

- Parent 与固定权重：受保护 `TBSI_E_MHSA_FULL_4090D_seed42` epoch15，config/checkpoint SHA256 已核验仍为登记值。
- 数据/评测：同一 LasHeR official testingset 244/244、同一 tracker hash、FP32、串行；S1 control 与历史 run1031 五项完全 parity。
- 硬件/软件：1× NVIDIA GeForce RTX 4090 D 24 GiB；Python 3.8.20、PyTorch 1.9.0+cu111、CUDA 11.1。
- Git：`09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`；GPU-hours 未记录，记 `NA`。
- 全量官方汇总：`output/experiments/M1_S1_ADMISSION_METHODS_FULL244_20261014/evaluation/official_summary.json`。
- 逐序列全指标：`output/experiments/M1_S1_ADMISSION_METHODS_FULL244_20261014/evaluation/per_sequence_official.csv`。
- cohort 汇总：`output/experiments/M1_S1_ADMISSION_METHODS_FULL244_20261014/evaluation/cohort_summary.json`。
- run manifests、日志、policy 与动作计数位于 `output/experiments/M1_S1_ADMISSION_METHODS_FULL244_20261014/`。
