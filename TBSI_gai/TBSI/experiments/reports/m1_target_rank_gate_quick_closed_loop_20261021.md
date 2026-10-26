# M1 TargetRankGate 最小闭环快速实验

日期：2026-10-21  
实验 ID：`M1-TARGET-RANK-GATE-QUICK-CLOSED-LOOP-25-20261021`  
Parent：受保护 M1 `TBSI_E_MHSA_FULL_4090D_seed42`，epoch 15

## 假设与唯一改变量

假设是：融合后的 search feature 中存在可学习的目标/硬背景差异；用模板条件的空间 token score 直接调制 CENTER 输入，并用 GT 区域相对最高 hard-background token 的 pairwise hinge ranking loss 约束，能够把目标排序传递到最终闭环决策。

唯一改变量是新增 `TargetRankGate`。父干路、TBSI、`tbsi_fuse_search`、CENTER、数据划分、Loss（除新增 ranking 项）、optimizer 和 tracker 状态更新保持不变。候选只训练 `target_rank_gate.*`，关闭 gate 的 matched control 使用同一候选 checkpoint 的其余权重，并自动剥离 gate 参数。

## 合同与训练

- 输入 fused search `[B,768,16,16]`，RGB/TIR template token `[B,64,768]`；输出保持 CENTER 输入 shape。
- 模块为零初始化残差，`alpha=0`；随机输入 identity、shape、save/load、反向和有限梯度检查通过。
- 参数量：24,674。
- LasHeR proxy：512 samples/epoch，3 epochs，batch 2，grad accumulation 1，FP32，worker 4；从公开 `TBSITrack_SOT_Pretrained.pth.tar` 初始化。
- ranking loss 从约 `0.02384` 降至约 `0.0043`，说明该分支可训练且梯度合同成立。

## 25-event paired closed-loop 结果

证据：`output/experiments/M1_TARGET_RANK_GATE_CLOSED_LOOP_25_20261021_RETRY2/{summary.json,paired_per_sequence.csv}`。

| 指标 | 结果 |
|---|---:|
| event IoU 平均 delta | `+0.0162481` |
| event IoU 中位数 delta | `-0.00000065` |
| prefix mean IoU 平均 delta | `+0.0031540` |
| failure 组（20）平均 delta | `+0.0184959` |
| control 组（5）平均 delta | `+0.0072568` |
| 明显胜出（delta > 0.01） | 2 |
| 逐事件正/负/近零（按原始浮点） | 8 / 13 / 4 |

平均收益几乎由 `girl`sblkbag`（`+0.3760`）和 `girlafterglassdoor`（`+0.0375`）贡献。其余事件的变化基本在 `1e-4` 量级；`leftunderbasket` 反而下降 `-0.00621`。failure 组虽然均值为正，但中位数为零附近，且有 9 个下降事件，不能称为稳定改善。control 组也有正均值，说明当前 gate 可能改变一般定位排序，缺少足够的目标条件性证据。

## 归因

已支持：

1. `TargetRankGate` 的输入输出和梯度合同成立。
2. 只训练一个小分支时，GT/hard-background ranking loss 可以被优化。

尚未支持：

1. ranking score 已经稳定转化为 CENTER 的目标峰排序。
2. 该分支在闭环中稳定改善目标/背景区分。
3. 该分支不会改变高 AUC/control 序列的原有定位合同。

本结果不能说明 MHSA 或 `tbsi_fuse_search` 已被定位为首因。更直接的解释是：训练监督作用在一个新 score 上，但该 score 对 CENTER 峰选择和后续 state 更新的影响仍然很弱或不稳定；单条大幅提升不足以排除偶然的 crop/state 反馈效应。

## 结论

**INCONCLUSIVE / NOT-PROMOTED**。停止当前 gate 的长训练、增大 residual 强度和继续扩大模块搜索。下一步应做固定 state、固定 crop 的单帧诊断：记录 gate 激活的 GT/BG 空间差、CENTER 输入前后 peak margin 与 GT rank，并将 score 直接替换/屏蔽作可逆对照。只有在固定状态下确认 score 改善 GT rank，才值得重新设计更严格的 target/distractor supervision 或 candidate-conditioned template update；否则应回到峰值解码和状态反馈链路。

本实验是 3 epoch proxy、25 个定向事件，不能作为 full-244 或多 seed 性能结论。
