# M1-S1 单次 readout 与 state commit 归因（2026-10-15）

## 结论

本诊断把同一事件帧上的 readout 修正价值与修正状态写回后的未来效应分开。当前冻结的 S1 事件动作几乎没有可重复的即时 IoU 收益：21 个 offset 事件的平均 `IoU(S1)-IoU(M1)` 为 `+0.00066`，中位数为 `0`，序列聚类 bootstrap 95% CI 为 `[-0.00554, +0.00727]`；正、负、零事件各 7 个。当前证据不支持把 S1 当帧 offset 修正称作有效增益，也不支持继续只调 admission gate 来期待明显涨点。

单次提交修正 state 的未来效应方向混杂。H10 平均 `+0.00512`（中位数 `+0.00096`，95% CI `[-0.00120, +0.01034]`），H30 平均 `+0.00995`（中位数 `+0.00119`，95% CI `[-0.00633, +0.03102]`），两个区间均跨零。此结果**尚未区分**小样本真实异质性与抽样波动，不能据此拟合风险门控。

## 设计与有效性

- Parent：受保护 M1 epoch15；冻结 S1 predictor 与 policy；没有训练权重。
- 数据：LasHeR train-side 中 12 条固定整序列，排除上一轮分支关联诊断；复用 retry2 冻结的同一批 21 个事件，覆盖 11 条序列。所选事件全为 offset。
- 每个事件都重放同一前缀。事件帧两支报告同一个 S1 框，提交前 tracker state 相同；`commit` 支把 S1 框写为下一帧 state，`fallback` 支把同帧未经修正的 M1 框保留为 state，之后都恢复 coupled S1。
- 即时 readout 标签为同事件帧 `IoU(S1 reported box)-IoU(M1 raw box)`；未来标签为两支之后 10/30 帧平均 IoU 差。
- 21/21 事件完整；事件输出最大差 `0`，提交前状态最大差 `0`，状态干预均非零。retry3 未来效应与 retry2 的 21 个事件结果逐项一致。单进程 FP32，RTX 4090 D，200.4 秒。
- 这是 train-side 机制诊断，不是 official test 结果；事件有意从 S1 非 noop 动作中选择，所以即时结果不能代表所有帧或全序列平均。

## 逐序列结果

数值为该序列所选事件的均值；`immediate` 是 S1 相对 M1 的事件帧 IoU 差，`commit H10/H30` 是提交 S1 state 相对保留 M1 state 的后续 IoU 差。

| 序列 | 事件数 | Immediate | Commit H10 | Commit H30 |
|---|---:|---:|---:|---:|
| 2runfive | 2 | -0.002978 | +0.016713 | +0.003865 |
| bikefromnorth | 2 | +0.000147 | +0.006197 | -0.036683 |
| bikeumbrellacome | 2 | +0.011597 | +0.009028 | +0.003179 |
| boybesidescarwithouthat | 2 | -0.003780 | +0.026155 | +0.033326 |
| boyinsnowfield4 | 2 | -0.012801 | -0.001034 | -0.001188 |
| boyplayingphone | 1 | +0.000000 | -0.012323 | +0.001703 |
| carleaveturnleft | 2 | -0.004303 | +0.000020 | -0.000728 |
| fatmancome | 2 | -0.017159 | -0.000881 | +0.000232 |
| girlgoleft | 2 | +0.011956 | -0.000277 | +0.100839 |
| manatwhiteright | 2 | +0.023328 | +0.005016 | +0.001243 |
| whitehatgirl`sheadleftwhiteumbrella | 2 | +0.000946 | -0.001040 | -0.000421 |

H30 大均值受少数序列/事件影响（例如 `girlgoleft`），中位数接近零；跨事件平均容易被少数长时域轨迹支配。序列聚类区间比把事件当成独立样本更合适。

## 对研究问题的归因

1. **降低 remaining162 退化：**先前官方 full-244 的 output-only-state 对照显示，保留 M1 内部 state 能让 remaining162 相对 S1 的 AUC 提高 `+0.9605`（95% CI `[+0.1932, +2.0016]`），支持 state feedback 是这组退化的重要来源；但全局保留 M1 state 会抹掉 low-AUC62 上 S1 的收益。之后 selective state commit 的官方结果又低于 M1，说明现有同状态风险代理没有解决该 trade-off。此次单事件诊断只在 11 条 train-side 序列上得到跨零、混杂的 state 效应，不能提供可用 gate 标签或证明延迟提交有效。
2. **提高修正收益：**事件帧即时 S1 offset 相对 M1 的效应接近零且区间跨零；配合既有 action-direction audit 中预测 offset 向量方向弱的结果，当前主要瓶颈更像是**修正动作向量/目标定位信号不足**，而不是单纯放宽或收紧 gate。此次只测 offset，不能推断 size 或联合动作。
3. **因果边界：**这次固定了当前事件帧报告框与前缀，仅对单次 state commit 做干预，因此可把事件 readout 差与之后的 state-mediated 差分开；它没有隔离“单次 readout 对状态无反馈”的长程效果，也没有验证全序列策略的组合效应。

## 决策与下一步

结论标签：**暂不支持**当前 offset 动作带来稳定即时收益；**暂不支持**以本批分歧/风险信号训练 state gate；**支持下一步单变量检验**修正向量是否有可学习方向。

下一步不重复拟合已测试过的 peak-offset/top-2 位移特征。先做固定 crop、固定 M1 state 的**候选可用性审计**，不跑 full tracking：对完全相同的固定状态解码 fused、RGB-only、TIR-only 候选框，逐状态报告候选 IoU、候选间分歧、各模态独立胜出的比例，以及 GT 从任一候选可达的 oracle 上限。oracle 只表示候选覆盖上限，不能当作可部署收益。若单模态候选稳定包含 fused 遗漏的目标，再用 leave-one-sequence-out 训练轻量候选选择器并冻结 gate/state；若候选集合本身没有改善 oracle 上限，则停止候选路由，转查融合表征/解码定位本身。之后才在 train-side 闭环验证 state commit 策略，不能拿固定状态候选收益替代闭环验证。

## 产物与 provenance

- retry2：`output/experiments/M1_S1_SINGLE_COMMIT_EFFECT_20261015_retry2/`
- retry3（补齐即时读出）：`output/experiments/M1_S1_SINGLE_COMMIT_EFFECT_20261015_retry3/`
- Runner SHA256：`3355c848e08b97efafe8542c708f0f0f35b8269fb793c5cbd91ee26ef14358c2`
- M1 YAML SHA256：`fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`
- M1 checkpoint SHA256：`1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`
- S1 predictor SHA256：`b97bf2fe4d576a447f6cabe62a7523a2276f6837ffcc8761714b181119d79c6c`
- S1 policy SHA256：`ff4cbef73341eae20c726911f916b25cd80b6a61458bba0a1e5196004840c098`
- retry3 metrics SHA256：`14b43f70f64e494134df8852ea8d1418d273efbd7343b35a0ec7f4d135e4c3b1`
