# M1 utility/no-op gate signal screen（2026-10-14）

## 目的

在不修改模型、不训练 checkpoint 的条件下，验证当前可部署特征信号是否能在 sequence-held-out fixed states 上选择 RGB/TIR intervention 或 no-op，并降低 harmful intervention。该实验使用已有 72 条 train-side 序列、144 个固定状态的目标区衰减 utility 代理，不能直接代表融合 residual 的闭环收益。

## 结果

| 信号模型 | 阈值 | sequence-macro utility | oracle fraction | coverage | harmful rate |
|---|---:|---:|---:|---:|---:|
| rank + margin | 0.0 | 0.0119 | 0.5190 | 1.000 | 25.7% |
| all signals | 0.0 | 0.0155 | 0.7193 | 1.000 | 20.8% |
| all signals | 0.01 | 0.0146 | 0.6758 | 0.938 | 19.4% |
| all signals | 0.02 | 0.0062 | 0.2891 | 0.250 | 6.3% |

其中 all signals 包含 base rank/margin、局部 energy gap、template support gap、template PSR gap 和 conflict。所有预测均按 leave-one-sequence-out 拟合，测试状态未参与对应模型训练。

## 判定

当前信号组合相对 rank/margin 有一定筛选能力：sequence-macro utility 提升，harmful rate 从 25.7% 降到 20.8%。但 no-op gate 仍不能把 harmful intervention 压到可接受水平；阈值升到 0.02 后 coverage 仅 25%，收益也明显下降。该结果支持“信号存在弱指导能力”，不支持直接训练融合 residual 或将该 gate 作为部署模块。

更关键的限制是 utility 标签来自目标区特征衰减与等面积背景控制，测量的是固定 CENTER margin 代理，不是 Cdiag residual 的真实效用，也不是闭环 IoU。因此本实验只能作为信号筛选诊断。

## 下一步

要进入联合算子训练，必须先把 utility 标签改成真实 decoded box IoU 或 center/readout action utility，并在 sequence-held-out 上复核 no-op 风险；当前 gate 结果不足以授权 full training。下一候选应优先围绕 readout candidate ranking 和 state recovery 设计，并保留 no-op/abstain。

## 证据

- 输入：`output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/utility_replay144/per_state_utility.csv`
- 汇总：`output/experiments/M1_UTILITY_NOOP_GATE_SCREEN_20261014/summary.json`
- 逐状态选择：`output/experiments/M1_UTILITY_NOOP_GATE_SCREEN_20261014/per_state_selection.csv`
- 脚本：`tools/screen_m1_utility_noop_gate_20261014.py`
