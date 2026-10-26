# M1 TargetRankGate 固定状态单帧诊断

实验 ID：`M1-TARGET-RANK-GATE-FIXED-STATE-DIAGNOSTIC-25-20261021`  
Parent：`M1-TARGET-RANK-GATE-QUICK-CLOSED-LOOP-25-20261021`

## 目的与设计

闭环快速实验中 ranking loss 明显下降，但 event IoU 中位数没有变化。本实验固定每条事件的初始化 template、事件 frame、manifest 保存的 `tracker_state_before` 和由该 state 产生的 crop，只替换 gate 开关。候选使用 quick checkpoint；control 使用剥离 `target_rank_gate.*` 后的 matched parent checkpoint。

每个事件记录：

- gate score 在 GT cell 与 hard-background cell 的最大值及 margin；
- gate score 的 GT rank；
- CENTER 原始 score 的 target/background margin 和 GT rank；
- 候选与 control 的 CENTER 差异。

## 结果

证据：`output/experiments/M1_TARGET_RANK_GATE_FIXED_STATE_DIAGNOSTIC_25_20261021/{summary.json,fixed_state_paired.csv}`。

| 指标 | 结果 |
|---|---:|
| gate GT-background score margin 平均 | `+0.045458` |
| gate GT-background score margin 中位数 | `+0.014168` |
| gate score margin 为正的事件 | `13/25` |
| gate score GT rank 平均 | `4.28` |
| gate score GT rank 为 1 的事件 | `13/25` |
| gate score spatial std 平均 | `0.033292` |
| CENTER 原始 margin delta 平均 | `-1.135e-05` |
| CENTER 原始 margin delta 中位数 | `-7.138e-06` |
| CENTER GT rank delta 平均 | `0` |
| CENTER peak-GT margin delta 平均 | `-7.313e-07` |

## 归因

该分支已经产生了有空间结构且具有一定 GT 偏好的 score：13/25 事件的 gate score 最高点落在 GT cell，平均 GT-background score margin 为正。但 CENTER 的 peak 排序几乎完全不变，GT rank 在 25 个事件中没有改变；CENTER margin 的变化只有 `1e-5` 量级。

因此当前瓶颈不是“新 score 没有激活”，而是 **score 到 CENTER 输入的残差通道太弱，且监督的 score 与 CENTER 的响应函数没有形成有效的排序耦合**。这解释了前一轮 ranking loss 可下降而闭环多数序列无变化：gate 学会了一个独立的空间排序信号，却没有改变决定峰值的特征/响应。

本实验支持停止当前 gate 的强度搜索和长训；不支持把它直接推广为融合修正。下一步若继续，只允许一个可区分的实验：保持同一 score，做 fixed-state 的直接 score-to-CENTER 可逆替换（将 score 作为 CENTER response 的显式加性项）并记录 GT rank 是否变化。若直接替换也不能改变 rank，应停止该监督路线，回到 CENTER 峰值解码/状态反馈；若能改变 rank，再设计受约束的 response coupling，而不是继续堆叠融合模块。

本结果仍是 25 个定向事件的机制诊断，不是 full-244 或多 seed 性能结论。
