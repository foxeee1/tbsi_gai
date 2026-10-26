# DCVR 与 Early-conditioned MSHA gate full-244 逐序列比较（2026-10-26）

## 判断

状态：`DIAGNOSTIC-COMPLETE / DESCRIPTIVE-ONLY / PROVENANCE-UNRESOLVED / OPERATOR-REFINEMENT-DESIGN`。

DCVR 的现存 full-244 序列结果比 Early-conditioned gate 更重尾：不少序列有正向变化，但少数大幅退化拉低了总体均值。两者的逐序列变化只有弱相关，且方向相反的序列很多。这个形态支持继续研究“如何限制错误 token 路由造成的尾部损失”，不支持“有条件调制必然优于无条件调制”，也不能证明 DCVR 或 Early gate 的条件信号造成了现有变化。

这次逐序列表格使用同一份现存 M1 结果目录作为算术 parent；244 条 parent 指标逐项相同。但该目录复算 AUC 为 `56.7659`，不符合受保护 M1 的固定 `56.21`。Early gate 的预测目录另有已登记的文件来源漂移。因而这里可用来定位候选序列和形成假设，不能当作受保护 M1 的正式配对结论。

## 并排结果

| 统计量 | Early-conditioned gate | DCVR | 含义 |
|---|---:|---:|---|
| 当前预测目录复算宏 AUC | `56.841` | `55.956` | DCVR 比 Early gate 低 `0.886 pp`；目录计算值 |
| full-244 序列 AUC 平均差 | `+0.075 pp` | `-0.810 pp` | 相对同一现存 M1 预测目录；非有效锚点比较 |
| AUC 序列差中位数 | `-0.038 pp` | `+0.036 pp` | DCVR 中位数微正但平均值被大幅负尾部压低 |
| AUC 上升 / 下降 / 持平 | `117 / 124 / 3` | `124 / 117 / 3` | 上升条数不能替代幅度和来源可信度 |
| 平均 IoU 差 | `+0.000741` | `-0.008313` | 序列宏平均，IoU 范围为 0–1 |
| 平均 IoU 上升 / 下降 | `122 / 122` | `125 / 119` | 两种候选均有明显异质性 |
| 两候选 AUC delta Pearson 相关 | — | `0.201` | 两候选变化仅弱相关 |

逐序列方向交叉：AUC 同时上升 `68` 条、同时下降 `68` 条、Early gate 上升而 DCVR 下降 `48` 条、Early gate 下降而 DCVR 上升 `54` 条，另有 `6` 条在持平边界。平均 IoU 的方向分组为 `72 / 69 / 50 / 53`，没有边界持平序列。相同 parent 说明这些方向差异不是 parent CSV 不同造成的；训练和推理合同仍未完全匹配，不能据此归因于算子本身。

注意：Early gate 历史资产登记的官方 AUC 是 `56.56`，但当前 prediction-directory 复算得到 `56.841`；DCVR 当前复算 `55.956` 与自身官方 `55.96` 一致。两者并排展示的 `56.841` 与 `55.956` 是当前分析输入文件的值，不替换原有登记指标；当前 M1 parent 仍不符合受保护锚点。

代表序列 AUC 差（百分点评价）：

| 序列 | Early gate | DCVR | 观察 |
|---|---:|---:|---|
| `3rdfatboy` | `+4.03` | `-54.18` | DCVR 严重负尾；quick7 中也出现过，必须核对预测文件和帧级分叉 |
| `leftfarboycomingpicktheball` | `+4.75` | `-44.07` | 与上一条类似，不能靠增加平均置信度信号解释 |
| `whiteridingbike` | `-25.98` | `-26.44` | 两种候选共同失败，优先怀疑共有的系统/数据/闭环因素 |
| `ab_whiteboywithbluebag` | `-14.28` | `-14.19` | 两种候选变化几乎相同，新增门控没有拉回该序列 |
| `shinycarcoming` | `+7.16` | `+16.97` | 两者同向改善，DCVR 幅度更大 |
| `girlrightthewautress` | `+14.97` | `+16.64` | 两者同向改善，具体原因仍需状态级证据 |
| `whiteboyrightcoccergoal` | `+31.91` | `+14.89` | 两者同向，但增益幅度不同 |
| `womanback2car` | `-22.91` | `-0.75` | DCVR 相对少退化，但两种候选在目录中都低于 parent |
| `right5thflag` | `-0.31` | `+19.07` | DCVR 独有的大幅正变化之一 |

DCVR 当前目录的五个最大 AUC 负变化为 `3rdfatboy (-54.18)`、`leftfarboycomingpicktheball (-44.07)`、`bluegirlbiketurn (-35.18)`、`swan_0109 (-28.25)`、`whiteridingbike (-26.44)` pp。此前逐序列 bootstrap 虽给出区间，但由于 parent 未通过受保护锚点检查，该区间不作为正式推断。

## 算子对照与归因

Early gate 在早期 block 对每个模板 token 取 RGB/TIR 各自对全搜索位置的最大余弦相似度，再加模板 token 的跨模态一致性，输入 `3→32→2` MLP。经 `tanh` 和指数映射后，它只调制两个方向的跨模态 template Value 消息；自注意力消息和 QK attention logits 不变。它保留了方向轴，但 `max` 丢掉了匹配位置。已有代表序列统计中方向 gate 几乎不随序列变化（RGB→TIR 约 `1.0004–1.0006`，TIR→RGB 约 `0.9894–0.9900`），表现更像弱的全局方向偏好，不像可靠的序列自适应选择器。该代表序列集合按既有结果挑选，不能外推。

DCVR 为每个模态、每个 template token 从模板到 search 的相似分布计算熵、top1-top2 竞争度，并结合前景 token 权重和 RGB/TIR agreement，归一化为 token source scale。它在联合 attention 前缩放 template Value，因此同一个源 token 会影响读取它的同模态和跨模态 query；它没有保留“只调制跨模态消息”的边界。理论上归一化 source 权重可集中到单 token；64 个 token 时 `1 + alpha * (64 - 1)` 可达 `16.75`（alpha=`0.25`），所以 alpha 并不是 ±25% 的硬限制。现有诊断没有保存 token 级 min/max，不能确认实际是否出现这类极端值。

DCVR 还有一个已确认的 train/test 函数差异：训练 actor 未传 `template_fg_masks`，fallback uniform 的比例为 `100%`；推理用真实 mask，fallback 为 `0%`。因此 DCVR 学到的表征是在一种路由证据下训练、另一种证据下推理。这个输入合同问题应先修复，不能用调温度或增加条件信号绕过。

### 对“有条件调制一定更好”的判断

条件化增加了表示能力，常量门控可以看成条件门控的一个特例；但在有限数据、错误或错配的条件信号、有限优化步数和分布偏移下，条件门控仍可能学到错误排序，或者对少数序列过度反应。它是否更好取决于条件能否预测“这条消息是否改善目标定位”，而不只取决于条件是否非恒定。当前证据证明 DCVR 路由活跃，未证明其熵、竞争或 agreement 与定位效用校准；Early gate 也没有稳定的 CENTER rank 修复证据。

所以两组的差异应解释为：Early gate 的方向调制幅度较弱、空间信息被 max 池化；DCVR 加入了 token 级变化，但同时扩大了作用路径并暴露于不一致的前景 mask 输入，序列负尾变重。后一句是需要用同状态重放验证的假设，不是已经证实的算子因果。

## 下一项算子设计

建议把下一项候选限定为 **Early gate 上的有界 token reliability residual**，不改《新方案》中的问题定义或总机制：

1. 以 Early gate 的方向性跨模态 Value gate 为基线，保留两个方向的原有控制；self-message、QK logits、CENTER head、crop 状态和在线模板更新保持原样。
2. 取 DCVR 的逐 token 置信信息，但在 mask parity 修复之后计算。令 `z_i=log(r_i+eps)-mean_i(log(r_i+eps))`，再令 `q_i=tanh(z_i/tau)-mean_i(tanh(z_i/tau))`；这样 `q` 的 token 均值严格为 0。
3. residual 只乘到对应的跨模态 source Value 消息，不进入同模态 self-message；例如 `g_i=1+0.05*q_i`，于是增量因子严格落在 `[0.9, 1.1]` 且均值为 1。它保持 Early gate 的方向控制，同时检验 token 级条件能否弥补其空间压缩，也避免 DCVR 的 token budget 归一化形成大倍率。

假设：已有同向改善序列（例如 `shinycarcoming`）可保留一部分 token 级收益，而 DCVR 的极端负尾序列（尤其 `3rdfatboy`、`leftfarboycomingpicktheball`）不会因过大的源 Value 改写而进一步放大。反证信号：route residual 虽非零，但固定相同 crop/state 下没有改善 GT-vs-hard-background CENTER rank/decoded box；或候选仍出现相同的大幅负尾。此时停止当前 source reliability 族，不继续堆新的 entropy/margin/agreement 特征。

进入训练前的顺序：

1. 先修复 DCVR train/test 前景 mask 定义一致，并在固定 batch 记录两端 token mask、source scale 的 min/max/分位数和梯度；这属于合同修复，不作为性能贡献。
2. 在同一 SOT 初始化、seed、quick3 schedule、相同 7 序列开发清单上，对 Early gate-only 与 Early gate + bounded residual 做配对筛查。每次记录实际 optimizer steps。7 序列中的 `ab_bikeoccluded` 曾主导旧 quick3 均值，因此不能用它单独决定保留算子。
3. 只有当固定 state 上的 token residual 对目标 GT/hard-background rank 有方向性改善、并且 quick7 不再由单条序列主导，才使用更大的预先固定序列集检验。正式收益判断仍需新鲜输出目录中重跑匹配 parent/candidate，并先通过 M1 锚点一致性检查。

这次只做离线 CSV 分析；没有训练、跟踪推理或修改《新方案》。

## 证据

- Early gate 输入：`output/experiments/M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023/analysis/recompute_retry1_20261024/per_sequence.csv`、同目录 `provenance_audit.md`。
- DCVR 输入：`output/experiments/M1_DCVR_FULL15_SOT_20261025/attribution/per_sequence_paired.csv`、`attribution/summary.json`。
- 合并逐序列表：`output/experiments/M1_DCVR_VS_EARLY_GATE_SEQUENCE_DIAG_20261026/per_sequence_comparison.csv`；复算摘要：同目录 `summary.json`。
- 可复算脚本：`tools/analyze_m1_dcvr_vs_early_gate_20261026.py`。
- 算子来源：`lib/models/layers/tbsi_layer.py`；Early gate 代表序列分析：`experiments/reports/m1_early_conditioned_gate_operator_sequence_analysis_20261023.md`。
- quick7 固定清单：`experiments/manifests/m1_dcvr_quick7_legacy_20261025.txt`，SHA256 `44247b6dbc9259c3e17c61a2d513ca1dff9588ec977946934c3ec4e87bc8d087`。此清单是从现存 quick 预测文件回建，不是原始 run manifest。
