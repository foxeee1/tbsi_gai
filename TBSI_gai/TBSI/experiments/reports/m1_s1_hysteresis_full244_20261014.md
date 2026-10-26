# M1-S1 时间滞回单变量实验

日期：2026-10-14  
状态：`DIAGNOSTIC-COMPLETE / HYSTERESIS-REJECTED`  
父级：`M1-S1-FULL244-CONFIRM-20261014`；固定 M1 parent 为 `TBSI_E_MHSA_FULL_4090D_seed42`

## 问题与假设

检验 S1 的 full-244 退化是否部分来自短时误触发：在不改 offset predictor 和修正幅度的情况下，加入 2 帧确认、2 帧释放滞回。若错误动作主要是短时孤立触发，滞回应降低干预频率，并在保留 low-AUC62 收益的同时减少 remaining162 退化。

## 单变量合同

- 唯一改变：动作策略的 `confirm_frames=2`、`release_frames=2`。
- 固定：train-side 拟合并冻结的 S1 predictor、`rho=0.5`、utility threshold `0.08`、M1 epoch15 checkpoint、tracker 代码、FP32 推理和官方评估实现。
- 数据：LasHeR 官方 testingset，同一 manifest 的 244/244 序列；cohort 沿用 S1 full-244 报告中的 low-AUC62、normal20 sentinel、remaining162 划分。
- 比较：M1 run1030、S1 run1031、滞回候选 run1032。候选是在 full-244 结果已知后开展的机制筛选，属于探索性复测，不是独立确认。
- 训练：无。本轮没有重新训练 predictor 或模型。

## 全量结果

| 指标 | M1 | S1 | S1 + 滞回 | 滞回 - M1 | 滞回 - S1 |
|---|---:|---:|---:|---:|---:|
| AUC / Success | 56.2077 | 56.4599 | 55.9134 | -0.2943 | -0.5465 |
| Precision | 70.2073 | 70.5443 | 69.8738 | -0.3335 | -0.6705 |
| Norm Precision | 66.1980 | 66.5330 | 65.9465 | -0.2514 | -0.5865 |
| OP50 | 68.3339 | 68.7700 | 68.0840 | -0.2499 | -0.6859 |
| OP75 | 47.1349 | 47.1488 | 46.6160 | -0.5189 | -0.5328 |

逐序列 AUC 胜/负/平为相对 M1 `107/129/8`、相对 S1 `106/136/2`。因此全量下降不是少数序列造成的平均值假象。

## Cohort 结果

下表均为相对 M1 的变化，单位为指标百分点。

| Cohort | 序列数 | AUC Δ | PR Δ | NPR Δ | OP50 Δ | OP75 Δ |
|---|---:|---:|---:|---:|---:|---:|
| low-AUC62 | 62 | +1.9917 | +2.6546 | +2.5646 | +2.6536 | +1.6831 |
| normal20 sentinel | 20 | -0.2159 | -0.1830 | -0.1539 | -0.2135 | -0.2893 |
| remaining162 | 162 | **-1.1788** | **-1.4957** | **-1.3412** | **-1.3656** | **-1.3900** |

滞回保留了 low-AUC62 的部分收益，但 remaining162 的 AUC 退化比 S1 full-244 的 `-1.0470` 更大。normal20 也由 S1 对 M1 的近零变化转为小幅下降。全量净收益目标未达到，且未解决未筛选序列上的负收益。

## 序列级与动作审计

相对 M1，AUC 最大增益序列为 `ab_bikeoccluded` `+56.91`、`leftfarboycomingpicktheball` `+46.80`、`runningcameragirl` `+22.59`、`AQgirlwalkinrain` `+14.12`、`whiteboyrightcoccergoal` `+11.39`。最大退化为 `bluegirlbiketurn` `-33.36`、`midof3girls` `-33.21`、`whiteridingbike` `-26.76`、`4thboywithwhite` `-25.52`、`rightblkfatboyleftwhite` `-25.15`。逐序列完整值见证据 CSV；这些例子仅用于定位风险，不能替代总体配对统计。

候选共处理 219,048 帧，其中 offset 87,054 帧（39.74%）、no-op 131,994 帧（60.26%）；S1 offset 为 87,212/219,048（39.82%）。候选与 S1 的逐帧动作一致率为 86.96%。滞回确实改变了部分动作时序，但几乎没有改变总体介入率。

持久失败审计中，候选序列等权 IoU<0.5 比例为 0.3191，M1 为 0.3166；帧加权比例均为 0.3737。出现持续失败的序列数候选 138、M1 136；失败段数候选 420、M1 430；最长段均为 1494 帧。失败段计数略降，但其他失败统计未改善，不能据此宣称恢复能力增强。

## 归因与决定

- **已排除**：当前 2 帧确认/2 帧释放滞回能够减少 remaining162 退化。动作总覆盖基本不变，overall 五项指标均低于 M1。
- **暂不支持**：滞回可改善 S1 的全量部署表现。low-AUC62 仍有收益，但被 remaining162 的损失抵消。
- **尚未区分**：remaining162 的伤害究竟来自错误 admission 信号、持续错误 offset 方向，还是闭环状态漂移。单纯改变时间确认规则没有区分这些机制。
- **下一步路线**：不继续叠加时间规则或扩大融合算子。回到 train-side 观测，审计 S1 在负收益序列上的 admission 信号与实际动作效用是否校准；按固定状态做一个可部署信号对真实 offset utility 的配对诊断。只有证实 admission 错误是主因后，才单独改 gate 并先做闭环 proxy。

## 运行与可比性

候选使用 run1032；官方测试覆盖 244/244 序列，FP32、单 tracker、串行调度。环境为 Python 3.8.20、PyTorch 1.9.0+cu111、CUDA 11.1、1× NVIDIA GeForce RTX 4090 D。Git commit 为 `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`。M1 YAML SHA256 为 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`，checkpoint SHA256 为 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`，与保护登记一致。GPU-hours 未采集，记录为 NA。

首次 evaluator 启动因 runner 未设置 `TBSI_RPP_TEST_LIST` 而失败；原日志保留。随后从已完成的预测结果运行恢复 evaluator 并完成官方评估，没有重跑 tracking。完整命令、策略/predictor 哈希、预测路径与评估状态见 manifest。

## 证据路径

- 合同、命令、环境和各阶段状态：`output/experiments/M1_S1_HYSTERESIS_FULL244_20261014/manifest.json`
- 官方全量指标：`output/experiments/M1_S1_HYSTERESIS_FULL244_20261014/official_summary.json`
- 244 条逐序列结果：`output/experiments/M1_S1_HYSTERESIS_FULL244_20261014/per_sequence_official.csv`
- 动作覆盖：`output/experiments/M1_S1_HYSTERESIS_FULL244_20261014/action_audit/`
- 持久失败统计：`output/experiments/M1_S1_HYSTERESIS_FULL244_20261014/persistence_audit/`
- runner 与 evaluator 恢复脚本：`tools/run_m1_s1_hysteresis_full244_20261014.py`、`tools/finalize_m1_s1_hysteresis_full244_20261014.py`
