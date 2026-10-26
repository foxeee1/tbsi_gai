# M1 / S1 / AMCM-E2 / 表示监督组横向比较（2026-10-18）

## 对照身份与总体指标

四组均使用 LasHeR full test 244/244、FP32 和官方评测脚本。M1 为受保护 epoch15；S1 为 run1031 的冻结 predictor/action/state 路径；E2 为 `M1_AMCM_E2_10EP_FULL244_20261017`；表示监督组为 `M1_AMCM_DR_REPR_CONTRASTIVE_PROXY_20261018`。

| 方法 | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| M1 | 56.21 | 68.33 | 47.13 | 70.21 | 66.20 |
| S1 | 56.46 | 68.77 | 47.15 | 70.54 | 66.53 |
| AMCM-E2，10 epoch | 56.68 | 69.03 | 47.60 | 70.83 | 66.88 |
| 表示对比监督，3 epoch | 56.41 | 68.64 | 47.50 | 70.44 | 66.60 |

相对 M1 的 AUC 增量分别为 S1 `+0.25`、E2 `+0.47`、表示监督 `+0.20`。E2 是当前四组中最高，但仍是单 seed，且 E2 只训练 AMCM 分支，不是从零训练全模型。

逐序列 AO 重算文件：`output/experiments/compare_m1_s1_e2_repr_full244_20261018.csv`。

## 逐序列变化

相对 M1 的胜负统计（按逐序列平均 overlap，严格比较浮点值）：

| 方法 | 上升 | 下降 | 持平 | 平均 AO 增量 |
|---|---:|---:|---:|---:|
| S1 | 113 | 130 | 1 | +0.285 |
| E2 | 126 | 117 | 1 | +0.510 |
| 表示监督 | 116 | 128 | 0 | +0.239 |

共同增益序列包括 `ab_bikeoccluded`、`leftfarboycomingpicktheball`、`girlrightthewautress` 和 `runningcameragirl`。这些序列可能共享明显的 crop 内模态/外观变化，AMCM 或 S1 都有机会改变定位轨迹。

E2 的代表性增益：`ab_bikeoccluded` `+66.02`、`3rdfatboy` `+55.10`、`leftfarboycomingpicktheball` `+47.79`、`girlrightthewautress` `+25.55`、`runningcameragirl` `+24.08` AO points。

E2 的代表性退化：`rightblkfatboyleftwhite` `-25.29`、`bluegirlbiketurn` `-15.86`、`redmidboy` `-15.14`、`carcomingfromlight` `-15.09`、`ab_whiteboywithbluebag` `-14.80`、`boyss` `-8.41`。

表示监督组的代表性增益：`ab_bikeoccluded` `+67.31`、`3bike1` `+57.85`、`leftfarboycomingpicktheball` `+43.08`、`girlrightthewautress` `+29.87`、`whiteboyrightcoccergoal` `+19.93`。

表示监督组的代表性退化：`whiteridingbike` `-26.65`、`rightblkfatboyleftwhite` `-25.95`、`shinycarcoming` `-25.92`、`leftexcersicebookyellow` `-17.75`、`carcomingfromlight` `-14.87`、`boyss` `-8.63`。

## 核心实现差异

| 维度 | M1 | S1 | E2 | 表示监督组 |
|---|---|---|---|---|
| 改动位置 | 无 | CENTER 输出/状态动作策略 | AMCM post-fusion | AMCM pre-fusion directed residual |
| 学习对象 | 主模型已训练权重 | 冻结 predictor 的动作和状态提交 | 只训练 AMCM | 只训练 AMCM |
| 训练进度 | 15 epoch 主模型 | 无新增训练 | 10 epoch，512 samples/epoch | 3 epoch，256 samples/epoch |
| 监督 | 原始 tracking loss | 固定 S1 utility/policy | reliability supervision | reliability + target/background representation margin + Hann failure pairwise |
| 是否改 state | 是原 M1 state 闭环 | 是，S1 action/state 闭环 | 原 state 闭环 | 原 state 闭环 |

因此 E2 和表示监督不是只差一个 loss：它们同时差了注入位置、训练时长、采样量和辅助损失。E2 的 +0.47 不能单独归因于“AMCM 更复杂”；它至少混合了 post-fusion 位置和 10 epoch 优化充分度。表示组的 3 epoch 结果也不能证明表示监督无效，因为它的 feature margin 仍未稳定改变。

## 为什么出现“拆东墙补西墙”

1. **局部修改改变了闭环轨迹。** 目标峰、背景峰和后续 template/state 更新相互耦合。某序列在早期获得正确框，可能在后期因 template 更新偏移而退化；反之亦然。
2. **E2 的 post-fusion 调整与表示组的 pre-fusion residual 作用对象不同。** 前者改变已经融合的响应，后者改变进入 TBSI 的两路 token；相同可靠性信号在两个位置不具有相同语义。
3. **监督目标不一致。** S1 优化动作效用和 state commit，E2 优化模态可靠性，表示组额外优化 GT/background feature margin；这些目标没有共同的逐帧闭环标签。
4. **收益序列重叠有限。** S1/E2 相对 M1 同时上涨 76 条、同时下降 81 条、方向相反 85 条；E2/表示组同时上涨 69 条、同时下降 71 条、方向相反 103 条。直接相加很可能把互补收益和互相冲突的状态改变一起引入。

## 能否联合涨点

目前不能把三组的增益直接相加。正确的检验必须是单独的 2x2 因子实验：

- 因子 A：AMCM post-fusion E2；
- 因子 B：target-conditioned pre-fusion representation loss；
- 固定同一训练数据、epoch、seed、冻结范围和 checkpoint 初始化。

至少需要 `A0B0`、`A1B0`、`A0B1`、`A1B1` 四个 full-244 或先用同一 proxy 的配对实验，计算交互项 `I = metric(A1B1) - metric(A1B0) - metric(A0B1) + metric(A0B0)`。若 `I < 0`，就是冲突；若 `I` 接近零，收益基本独立；只有 `I > 0` 且逐序列方向稳定，才支持联合模块。

S1 不应直接放进这个因子组合，因为它改变 state/action 闭环，属于另一条路线。应先固定 M1 state，再比较 AMCM 因子，避免把 state feedback 混入融合交互项。

## AMCM 复杂度与从零训练判断

当前 `AMCMLiteReliability` 已包含 depthwise 3/5/7 局部上下文、1x1 context mix、可靠性 gate，以及 directed residual 模式下两套 message MLP；它不是纯标量乘法，但仍是轻量 spatial adapter。关键瓶颈不是层数不足，而是：template prototype 使用全 token 均值、训练只更新 AMCM、可靠性监督没有直接约束闭环峰值。

直接增加 hidden_dim、卷积层数或 attention heads 的优先级较低，可能只增加容量而不改变监督信息。更有价值的复杂化顺序是：

1. target-conditioned prototype：用 template 的目标区域 token，而不是全局均值；
2. hard-background mining：使用 GT 外且与 template 高相似的局部 token，并记录 feature margin；
3. cross-modal target interaction：让 RGB/TIR message 由目标 prototype 条件化；
4. 仍保持 bounded residual 和相同 pre/post 注入位置，单独测试位置效应。

从零训练全模型理论上可能提高上限，但当前没有证据证明 AMCM 已受容量限制：3 epoch 表示组已有非零梯度，E2 10 epoch 只有小幅增益，且不同位置方向冲突。若直接 full-from-scratch，会同时改变 backbone、TBSI、head 和 AMCM，无法判断收益来源，计算成本也接近重新训练主模型。

建议顺序：先做同一初始化、同一 10 epoch、同一 512 samples/epoch 的 A0B0/A1B0/A0B1/A1B1 因子验证；只有联合项显示正交互且 feature margin 真正扩大，再做“全模型从零训练 + AMCM”的确认组。否则应停止堆复杂度，回到 target prototype 和监督合同。
