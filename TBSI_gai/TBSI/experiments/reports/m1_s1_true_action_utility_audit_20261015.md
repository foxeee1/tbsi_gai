# M1-S1 真实 offset 动作效用审计（2026-10-15）

## 研究问题与合同

前一轮 admission gate 的训练标签来自 GT 导出的理想 offset 修正效用。本审计检验这些部署可见分数，能否识别**序列留出 S1 predictor 实际输出的 offset 动作**是否有害。

- Parent：受保护 M1 `TBSI_E_MHSA_FULL_4090D_seed42` epoch15；checkpoint/config SHA256 与登记值一致。
- 数据：LasHeR train-side 72 序列、144 个预选固定状态；使用已保存的融合特征、trace crop、tracker state 和 GT。固定状态、crop、CENTER head 和 decoder 不变。
- 动作：逐序列留一拟合二维 offset correction（peak-margin 输入、ridge alpha=10），在留出的序列上输出 offset 向量；在相同固定状态上比较 no-op 与该预测向量，rho=0.5。没有闭环状态更新，没有训练 M1 网络。
- gate：沿用已冻结的 S1 eligibility (`predicted utility > 0.08`)、selective-risk 分数/阈值和 motion-innovation 分数/阈值；本审计不重调阈值。
- 效用：实际预测修正后的 decoded bbox IoU 减 no-op decoded bbox IoU。另列 GT 导出的理想 offset correction IoU delta，不能将后者当作 S1 实际动作效用。
- 环境：1× NVIDIA GeForce RTX 4090 D 24 GiB；Python 3.8.20、PyTorch 2.0.0+cu118、CUDA 11.8；FP32，单进程、0 DataLoader workers。GPU-hours 与单独 wall time 未采集。

## 主结果

No-op 重放与存储的固定状态基线完全一致，最大绝对 IoU 差为 `0`。144/144 状态的 GT 均完全落在记录的 current crop 内，因此这组样本上的动作误差不能归因于目标已经在 crop 外。

| 决策 | 放行动作数/144 | 放行率 | 放行动作有害率 | 含 no-op 的每状态平均实际 IoU delta | 按序列 bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| S1 utility threshold only | 85 | 59.03% | 34.12% | +0.000096 | [-0.001374, +0.001173] |
| + selective-risk gate | 22 | 15.28% | 27.27% | +0.000142 | [-0.000307, +0.000490] |
| + motion-innovation gate | 66 | 45.83% | 36.36% | +0.000062 | [-0.001344, +0.001134] |

三者的实际单步效用区间都跨零。Selective-risk 只留下22个动作，样本少且区间较宽；不据此宣称它的正均值是真实收益。

## 动作方向与理想效用

GT 导出的理想修正平均 IoU delta 为 `+0.0755`，79.17% 状态为正；留出序列上 S1 预测动作的实际效用均值接近零。两种效用正负号在 `34.03%` 状态不一致。

预测的 offset correction 与 GT 导出修正向量平均 cosine 为 `0.089`，中位数 `0.158`，方向 cosine 为正的状态占 `56.25%`。实际施加向量的平均模长为 `0.055` 个 offset grid 单位；GT 修正向量平均模长为 `0.690`。它们并非完全等价的“最佳位移”，但方向一致性和幅值差都显示当前预测没有可靠复现理想方向。

为核实留一拟合合同，用同一批输入和 offset target 重拟合全量 ridge predictor；与冻结 S1 artifact 的 bias/scale 一致，二维权重最大绝对差为 `8.74e-6`。该差异来自输入 peak-margin 的微小精度差，不改变留一评估定义。

部署可见分数对实际留出动作有害性的判别也弱：selective-risk OOF 概率 AUROC `0.591`，motion innovation AUROC `0.461`（以分数越高越可能有害为正类方向）。这些是144状态上的探索性点估计，不能视为稳健判别能力。

按既有固定状态标签描述：`good_only` 35 状态的实际动作平均 delta `-0.00869`、有害率60.0%；`matched_good` 37 状态为 `+0.00538`、有害率24.3%；`pre_onset` 35 状态为 `+0.00118`、有害率31.4%；`severe_onset` 37 状态为 `+0.00109`、有害率29.7%。样本由预先诊断规则选取，各切片只作机制线索，不外推全数据集。

## 归因

- **已支持**：前一轮 risk/motion gate 标签和实际执行动作之间有合同差异。标签衡量朝 GT offset target 移动的理想修正，而不是 predictor 实际输出向量的收益。
- **暂不支持**：当前 gate 能识别有害的 S1 预测动作。选择动作后的有害率仍为27%–36%，实际效用均值没有离开零的不确定区间。
- **支持作为当前主假设**：offset predictor 只以一个 peak-margin 标量回归二维 correction；该输入没有显式提供移动方向。序列留出向量对齐弱，符合方向信息不足的预期。当前数据不证明这是所有可能输入下的唯一原因。
- **尚未区分**：真实在线闭环中的增益/退化是否由后续状态漂移放大。固定状态回放不更新 tracker state，不能回答轨迹恢复和误差累积。

## 下一步决定

先暂停增加 admission 特征和融合算子。下一项只改变 offset predictor 的输入表示：在保持 predictor 容量、rho、utility threshold、no-op fallback、数据成员和固定状态评估不变时，增加推理可见的二维 offset 方向信息（优先使用原 CENTER offset map 在响应峰处的 x/y 分量），再按序列留一拟合和评估。预设通过条件是：方向 cosine 与实际 decoded-IoU delta 都在留出序列上改善，且序列 bootstrap 的每状态效用差不再跨零；否则停止当前 S1 向量修正路线。该诊断仅允许进入另一个固定状态短实验，不授权 full training 或 full-244。

## 证据与限制

- Per-state actual action and gate scores: `output/experiments/M1_S1_TRUE_ACTION_UTILITY_AUDIT_20261015/per_state_true_action_utility.csv`.
- Summary and run provenance: `output/experiments/M1_S1_TRUE_ACTION_UTILITY_AUDIT_20261015/summary.json` and `manifest.json`.
- Reproduction script: `tools/audit_m1_s1_true_action_utility_20261015.py`.
- Command from repository root: `python3.8 TBSI_gai/TBSI/tools/audit_m1_s1_true_action_utility_20261015.py`.
- Inputs are the fixed feature archive, matching capture rows, sequence-held-out gate outputs, and three train-side trace directories; hashes are recorded in the manifest.
- This is one-seed, preselected train-side fixed-state diagnosis. It measures one-step utility, not tracking AUC or closed-loop performance. Thresholds were not retuned against actual action utility.
