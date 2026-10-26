# M1 候选级 response evidence selector OOF（2026-10-15）

## 结果

在同一 72 条序列、144 个固定状态上，冻结 fused/RGB-only/TIR-only 候选，加入每个候选的 peak score、top-2 margin、归一化熵、peak cell 坐标和候选间 response 差异，使用 sequence leave-one-out ridge selector。

| 方法 | 平均 IoU | 中位数 IoU | 相对 M1 fused | 序列聚类 95% CI |
|---|---:|---:|---:|---:|
| M1 fused | 0.64382 | 0.69920 | 0 | [0, 0] |
| 几何 + response selector | 0.60609 | 0.66489 | **-0.03774** | **[-0.07081, -0.01292]** |
| Best-of-three GT oracle | 0.68778 | 0.71826 | +0.04396 | [+0.02014, +0.05297] |

selector 选择 fused/RGB/TIR = `110/27/7`，与 oracle 一致率 `50.0%`；改善 `11/144` 状态，退化 `22/144` 状态。

## 归因

结论标签：**已排除**当前几何加 response evidence 的候选 selector；**已排除**继续堆叠简单 admission 特征作为主要路线。

候选 oracle 仍有 `+0.04396` 的上限，但两类 selector 都无法利用它。说明候选的峰值强度、熵和几何关系没有提供稳定的候选质量排序。当前主要瓶颈更可能在：

- crop 内 center-cell 选错后，三个候选共享相同的错误搜索位置；
- 单模态候选虽然偶尔更好，但其响应质量与真实目标质量不一致；
- 正确目标可能不在当前候选生成合同覆盖的可解码位置。

因此，继续增加 selector 输入或调阈值没有干净的因果依据。

## 下一步

停止候选 selector 扩展，不接入闭环。下一项实验转为 center/crop 可达性审计：在相同固定 crop/state 上统计 GT cell 与三个候选 peak cell 的距离、GT cell 是否被候选响应的 top-k 覆盖，以及 oracle center-cell 替换后的 IoU 上限。若 center-cell oracle 上限足够大，改造中心选择/候选生成；若上限仍小，则后融合 readout 修正没有足够信息，应停止该路线。

## 产物

- [response evidence](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/M1_S1_CANDIDATE_RESPONSE_EVIDENCE_20261015/response_evidence.csv)
- [selector metrics](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/M1_S1_CANDIDATE_SELECTOR_RESPONSE_OOF_20261015/metrics.json)
- [逐状态结果](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/M1_S1_CANDIDATE_SELECTOR_RESPONSE_OOF_20261015/per_state_selector.csv)
- [逐序列结果](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/M1_S1_CANDIDATE_SELECTOR_RESPONSE_OOF_20261015/per_sequence_selector.csv)
