# M1-S1 方向特征单变量对照（2026-10-15）

## 问题与假设

上一轮真实动作效用审计显示，S1 的实际 offset 向量方向信息较弱：GT correction 方向 cosine 均值为 0.089，正向率为 56.25%。本实验只改变 ridge predictor 输入，检验当前 CENTER 输出中可直接取得的空间方向信号能否改善 held-sequence-out 修正。

## 协议

- Parent：受保护 M1 `TBSI_E_MHSA_FULL_4090D_seed42` epoch 15；配置与权重 SHA256 按 manifest 记录并与保护值一致。
- 样本：LasHeR train-side 预选 72 序列、144 固定状态；同一缓存融合特征、crop、state、CENTER head 和 GT。
- 拟合：按序列留一 ridge，alpha=10；三臂使用同一目标向量、`rho=0.5`、S1 原 utility threshold `0.08` 以及 no-op fallback。
- 唯一变量：predictor 输入特征。A=`peak_margin`；B1=A 加 top-1 response peak 处 CENTER offset map 的 x/y；B2=A 加 response top-1 到 top-2 峰的相对 x/y。
- 主指标：实际预测动作的 decoded IoU delta，所有 144 状态含 no-op；另报告放行动作效用、有害率、GT correction 方向 cosine，以及按序列配对 bootstrap。
- 环境：1×NVIDIA GeForce RTX 4090 D 24 GiB；Python 3.8.20、PyTorch 2.0.0+cu118、CUDA 11.8；FP32，单进程、0 workers。

## 结果

| Arm | 含 no-op 每状态 mean ΔIoU | 序列 bootstrap 95% CI | 放行状态 mean ΔIoU | 放行动作有害率 | 方向 cosine 均值 / 正向率 |
|---|---:|---:|---:|---:|---:|
| A: peak margin | +0.000096 | [-0.001374, +0.001173] | +0.000162 | 34.1% | 0.089 / 56.3% |
| B1: +peak offset map x/y | -0.001217 | [-0.004721, +0.001917] | -0.002062 | 43.5% | 0.022 / 50.7% |
| B2: +top-2 relative x/y | -0.003369 | [-0.004911, +0.000072] | -0.005708 | 44.7% | -0.075 / 45.8% |

三臂均放行 85/144 个状态。A 的 85 个放行动作与前一轮审计逐状态完全一致，最大 ΔIoU 差为 0；no-op 对缓存 baseline 的最大 IoU 差为 0。相对 A，B1 的序列宏平均差为 -0.001077，95% CI [-0.004420, +0.001984]，18 条序列改善、17 条退化；B2 差为 -0.002202，95% CI [-0.004754, +0.000066]，14 条改善、22 条退化。其余序列没有 A 放行动作或差值为零。

汇总脚本初次运行时将“放行动作均值”误标为“含 no-op 每状态均值”。原始逐状态结果未变；已从逐状态 CSV 重算并更正 `summary.json`，并在 manifest 中记录重算脚本哈希。表中含 no-op 数字以更正后的 summary 为准。

## 结论

**已排除**：这两种直接空间特征接入方式，在当前 S1 predictor、ridge 容量、固定状态与放行规则下没有改善方向或单步效用；点估计均低于 A，方向 cosine 与有害率也变差。**暂不支持**将其扩展成融合模块或进入全量训练/测试。

该结果没有排除所有方向信息，也不是闭环跟踪结论。它支持暂停这条 readout 特征扩展线，下一步做同 crop/state 的 RGB、TIR 与融合信息互补性审计：检验有用定位信号是否存在于单模态或跨模态差异中、以及融合表示是否丢失该信号；只有证实互补定位信息可恢复，再设计融合算子。

## 证据文件

- 主实验脚本：`tools/compare_m1_s1_direction_features_20261015.py`
- 汇总重算脚本：`tools/recompute_m1_s1_direction_feature_metrics_20261015.py`
- Manifest、汇总及逐状态结果：`output/experiments/M1_S1_DIRECTION_FEATURES_20261015/`
- 完整逐序列对照：`output/experiments/M1_S1_DIRECTION_FEATURES_20261015/per_sequence_direction_features.csv`
- Parent 审计：`experiments/reports/m1_s1_true_action_utility_audit_20261015.md`

所有结论限制在预选 train-side 固定状态、单一 M1 checkpoint/seed 和单步 decoded IoU；没有训练、闭环跟踪或 official test。
