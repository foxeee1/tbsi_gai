# M1 目标-硬背景 Margin 不确定性可学习性诊断（2026-10-21）

## 结论

结论分类：**NOT-LEARNABLE-ENOUGH-FOR-GATE / 暂不接入 Loss 或 MHSA**。

在前一轮预先固定的 50 条 testingset 序列中，26 条有合法固定状态事件。使用这 26 个事件，冻结 M1 的非 GT 输入特征预测 `off` arm 的 GT-hard-background margin，并采用 leave-one-sequence-out ridge，检查“目标背景不确定性是否能被当前输入信号学习”。

输入只包含推理时可获得的特征：loc3/6/9 的 query match、reliability、participation 均值；没有使用 GT mask、GT peak、hard-background 坐标或任何 `off_`/delta 标签特征。

结果：

- Pearson correlation：`-0.1375`；
- rank correlation：`-0.1672`；
- margin 正负符号准确率：`61.5%`；
- 观察到的正 margin 比例：`26.9%`；
- 保守下界为正比例：`3.8%`；
- RMSE：`0.1946`。

当前 query/reliability/participation 特征不仅预测能力弱，方向还与目标 margin 轻微相反。因此直接增加 margin uncertainty loss，极可能只让一个没有目标条件的 estimator 拟合噪声，不能解决前一轮 gate 的背景偏置。

## 实验合同

- Parent：冻结 M1 `TBSI_E_MHSA_FULL_4090D_seed42` epoch15。
- 数据：同 `M1_MSHA_UNCERTAINTY_GATE_50SEQ_20261021`，预固定 50 条序列，26 个合法事件。
- 方法：每次留出一条序列，训练 ridge 预测其 margin；不改模型、不改 Loss、不训练 M1 主干。
- 特征：`loc3/6/9 × {query_match_mean, reliability_mean, participation_mean}`。
- 输出：`output/experiments/M1_MSHA_MARGIN_UNCERTAINTY_LEARNABILITY_50SEQ_20261021/{summary.json,predictions.csv}`。

## 解释

这项结果不能证明“目标-背景监督没有价值”。它只证明当前 uncertainty 输入没有足够的目标条件信息。原因可能包括：

1. 使用的是全局均值，空间目标位置被平均掉；
2. query match/reliability 更像模态一致性，而不是目标身份；
3. hard-background 是 CENTER 输出后的动态错误峰，不能由模板全局统计稳定预测；
4. 目标在遮挡、弱热响应时本身可能更不确定，不能简单把 uncertainty 解释为背景概率。

## 决策

停止以下尝试：

- 直接给当前 reliability 增加 margin loss；
- 调整 uncertainty temperature、bias 或 gate 强度；
- 用全局 uncertainty 直接调节所有 MHSA 跨模态消息。

如果继续目标背景监督，必须改变输入和标签粒度：

- 使用每个搜索 token 的 target/background signed margin，而不是全局均值；
- 使用 template-conditioned token pair 或局部 hard-negative 集合；
- 预测 margin 的下界 `mu-kappa*sigma`，而不是预测“目标概率”；
- 先做 token-level 固定状态可学习性审计，再考虑加入辅助 Loss。

下一步优先级转为：

1. 50 序列 token-level target-conditioned signed-margin 审计；
2. CENTER size/offset 解码误差分解；
3. 状态更新与恢复机制诊断。

当前不进入主 Loss，不开始长训。
