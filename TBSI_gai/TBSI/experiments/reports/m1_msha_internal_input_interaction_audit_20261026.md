# M1 MSHA 中间输入建模与交互机制审计（2026-10-26）

## 研究目标校准

当前贡献目标不是继续扩展 Early-conditioned MSHA gate。Early gate 是已取得小幅正点估计的先导组和比较对象；论文方法应进一步改变 MSHA 内部的中间特征建模及 RGB/TIR token 交互，使“输入如何被组织—两模态如何交换信息—定位读出如何响应—闭环跟踪是否获益”成为一条可检验链路。

因此，单纯增加 gate 条件、改变 gate MLP 或继续调 Value 缩放不符合目标。DCVR-v2 也只是既有草案，不是已选终版；DCVR-v1 与 v2 的结果只能作为证据，不能自动定义当前创新。

## 代码中的原生信息流

在 `lib/models/layers/tbsi_layer.py` 的 `V2TemplateFusion.forward` 中，RGB/TIR 模板 token 先拼接，再共享 LayerNorm 和 QKV 投影：

```text
X = cat(T_rgb, T_tir)
Xn = LayerNorm(X)
(Q,K,V) = shared_QKV(Xn)
L = Q K^T / sqrt(d)
A = softmax(L)
Y = W_o(A V) + residual
```

注意力矩阵按 RGB/TIR query 与来源切成四块：RGB←RGB、RGB←TIR、TIR←RGB、TIR←TIR。原生 MSHA 因而在一个联合 token 图上同时形成同模态和跨模态交互。后续层接收融合后的模板表示。Early gate 只在 Value 上缩放跨模态消息；它没有改变 `Xn` 的输入表征、Q/K 关系或注意力支持集。

## 既有候选与证据边界

| 候选 | 改动所在 | 可确认的证据 | 当前解释 |
|---|---|---|---|
| Early-conditioned MSHA gate | 对早期证据生成方向/token scale，缩放跨模态 Value | full15 相对登记 M1 为小幅正点估计，但无 matched gate-off full15；固定状态 CENTER top-1 变化 `0/28`，offset map 变化更明显 | 说明该支路可调制消息；没有证明它修复目标排序，也不是中间输入建模创新的充分形式 |
| DCVR-v1 | 在联合注意力前改写 Value token scale，影响 self/cross query 所读取的源 Value | full15 训练 mask fallback 100%、测试 0%；M1 指标 provenance 不匹配；固定状态定位差异很小 | 当前结果受输入合同和比较来源限制，不能判定算子因果；其“只改跨模态消息”的边界也不成立 |
| DCVR-v2 | 草案提出分布证据、方向预算与 token 分配 | 尚无训练或评测；和 gate/DCVR-v1 同属 Value 分配家族 | 不是已验证方法，且不满足用户要求的中间输入与交互改造 |
| Early Template Value Reconstruction | 对跨模态模板 Value 做低秩 token 变换 | fixed-state strength replay 未显示定位机制支持 | 排除当前形式的固定状态效用主张，不排除 MSHA 输入特征变换的其他设计 |
| Query-conditioned Cross-Value | 用 query/key/evidence 生成 pairwise Value residual | fixed-state 中 feature 有变化而 target rank `0/28` 改变；quick3 AUC `-0.135 pp`，12 序列 mixed | 当前 Value-only 交互没有得到定位支持；候选具备可学习性但未获性能支持 |
| Role-Pair Bias | 在 QK logits 上增加前景 token pair bias | mask plumbing 已修复且参数有梯度；但训练未在固定小样本上证明可拟合，配置还偏离预登记 manifest | 不能称为算子已证明无效；也不能据此晋级。先前置拟合/读出合同未通过 |
| Foreground Query Role | 限制/重组参与交互的 query token | fixed-state 有结构变化但未提升定位；24-sequence holdout proxy IoU 略低于 M1 | 当前 query restriction 未获支持，不代表所有 token 交互重建无效 |
| Spatial Token Bridge | 在 search 特征侧加入稀疏局部跨模态 correspondence residual | 46 fixed states 和 10-epoch learnability 诊断中 CENTER/query rank 均未变；target 对 shifted 无稳定优势 | 停止当前 search-side residual bridge；这不是模板 MSHA 中间输入改造的直接检验 |
| Local Cross-Modal Correspondence Exchange | 在 backbone block 2 对 search token 做局部 3×3 双向交换 | RPP60 AUC `-1.113 pp`，区间跨零，FPS 下降约 12.1% | 停止该具体位置/形式；不能外推为 MSHA joint template interaction 整类无效 |
| Differential attention audit | 只读统计 self/cross attention 与消息重叠 | 28-event 的 matched-good 与 severe-onset 统计近似相同；未发现可支持统一相减的事件判别信号 | 已排除“仅凭统一 self/cross overlap 即做差分”的依据；不能据此排除关系条件化交互 |

上述证据共同说明：已经试过若干 Value 幅度、pairwise Value、空间桥接和注意力偏置，但它们并不等价，也不能笼统总结为“复杂 MSHA 交互都无效”。具体失败结论分别受定位读出不变、short-fit 合同不完整、训练/测试 mask 不一致、代理集合及续训合同限制。可靠的下一步应先识别中间输入还是交互本身是限制点。

## 独立判断

现有材料不支持马上选定一个更复杂的新算子进入训练。它支持的是：Early gate 不足以代表最终贡献；现有 Value-only 变体不能回答“MSHA 的联合输入和跨模态关系建模是否丢失了有用的目标/互补结构”；search-side bridge 的负结果也不能直接否定模板 MSHA 内部建模。

当前最需要补的不是另一个模块，而是把候选放在原生 MSHA 的准确边界上，使用同一帧、同一 crop、同一 tracker state 做反事实输入/交互分析，测出哪一种改变能把目标相关信息传到 CENTER 与框。该诊断如果没有分辨力，就不应把复杂度当作创新性的替代证据。

## 下一步：固定状态 MSHA 输入—交互反事实分析

### 问题

当目标仍在搜索 crop 内而原生定位发生失配时，MSHA 前的 RGB/TIR 模板 token 是否已经含有可利用的目标信息？若存在，损失发生在 joint QK 对关系的组织、Value 消息内容，还是融合输出到 CENTER/回归头的传递？

### 设计

使用受保护 M1 checkpoint，只做固定状态 replay，不训练、不写 checkpoint、不更新 tracker state。事件应从已有 train-side calibration manifest 按预先规定的 matched-good/severe-onset 类别取样，不能按候选结果挑选。固定模板、search crop、随机态和后续网络状态。

在 MSHA 实际输入 `T_rgb/T_tir`、QK logits、四个 attention 子矩阵、四路消息、输出投影、CENTER map/size-offset 与解码框上设置配对反事实臂：

1. `native`: 原生 M1 完整路径，必须逐状态复现保存的 M1 输出。
2. `aligned-pair`: 保留原生 self-attention；在被审计的 cross-attention 子矩阵中，允许每个目标模板 query 读取其同网格位置的另一模态 token，并按行归一化。它是“空间对齐交互是否有用”的强干预，不代表原生 MSHA 当前已有此约束。
3. `shifted-pair`: 用预先固定的循环位移映射替换上述同网格对应，保持每行可读来源数、注意力温度和 token 范数相同，作为错配控制。
4. `foreground-context-swap`: 对等量前景/背景来源 token 交换 cross-attention 可读角色，保持每行参与数和范数分布，判断消息效用是否依赖目标角色而非 token 能量。

这些是固定权重下的交互支持集干预，只用于测量原生特征是否包含可利用的目标条件关系，不是候选训练方法，也不声称同网格配对一定正确。反事实只改指定 MSHA 层的 cross-attention 支持集，不改 backbone 参数、注意力温度、Value 内容、定位头或闭环状态提交。先分别审计 `[3,6,9]` 层，每次只干预一层；确认最早有效边界后才测联合层。若干预会显著改变每行 attention entropy，应另报并做等熵控制，避免把信息量变化误归为空间配对效应。

### 记录与可证伪判据

逐事件记录：输入 token cosine/范数及 foreground/background 分布；QK logits 与 attention mass；四路消息的范数、前景来源占比及 query-token 相关性；融合后特征目标/干扰子集间距；CENTER top-1、GT rank、GT-hard-background margin、peak 邻域质量；size/offset、同 crop 解码 IoU。按序列汇总，不只报告跨帧均值。

- 若 `aligned-pair` 相对 `shifted-pair` 不能稳定改变目标来源消息或定位读出，说明该条件下空间配对假设不成立；不要继续写 pairwise bridge。
- 若 attention 支持不变但 Value 消息内容与读出对目标/背景角色交换敏感，优先研究模板输入表征重构，不先改 QK。
- 若 QK/attention 支持发生目标对齐变化，且 target-vs-shifted 的 CENTER/框读出同向改善，支持后续做关系条件化的 joint QK/Value 交互。
- 若 MSHA 内部消息发生明显差异，但 CENTER/框读出不动，瓶颈位于融合后投影/后续 head 的可读性；不应继续扩大 MSHA 交互复杂度。
- 若反事实能改善固定状态读出但未改变闭环轨迹，需要后续独立验证状态反馈是否抵消收益；不能宣称已解决 M1 失跟链。

这一步是机制定位，不是候选方法结果。其价值在于决定下一版应改 `X→Q/K/V` 的输入建模，还是 joint attention 的关系核，或停止在 MSHA 内继续加复杂度。

## 后续方法与训练闭环（条件触发）

只有上述分析显示目标/背景 token 角色或跨模态空间配对确实能因果改变 localization-facing readout，才登记一个独立候选。优先考虑的研究家族是“目标条件的互补子空间输入 + 关系条件化的跨模态消息”，而不是把两个现有模块拼接：分别建模 RGB/TIR 的共享目标证据与模态私有补充证据，由配对关系决定 MSHA 如何形成跨模态 Q/K/V。是否采用该形式取决于反事实结果；目前没有证据支持直接实现或训练它。

候选进入训练前需有：M1 disabled parity；输入 mask/特征轴一致；单 batch 梯度和固定小样本可拟合；冻结主干状态不漂移；同初始化、数据序列、增强轨迹、优化器和更新步数的 candidate/control。训练后分三层闭环：

1. 训练侧机制：梯度、子空间能量、QK/attention 变化、各路消息变化、前景/背景角色差异。
2. 固定状态定位：同一 crop 下 CENTER rank/margin、offset/size、解码框，配 target-pair 与 shifted-pair 对照。
3. 闭环 tracking：先用预注册、较大且配对的序列集合做筛查，再按固定完整 LasHeR 流程对照 Early gate 和 M1；记录逐序列 AUC/OP、失败起点、crop 内错误持续时间、出界及恢复。模板更新状态必须固定并写清。

代理训练只用于机制和排序，不可当作论文主结果。若短训 control 的重跑波动尚未解释，应先完成增强后输入 tensor hash/RNG 审计，再解释候选与 control 差异。M1 继续保持受保护 parent，所有新配置、输出和 checkpoint 使用独立身份。

## 当前决定

固定状态 cross-support 分析已执行，结果见 [m1_msha_cross_support_fixed28_20261026.md](m1_msha_cross_support_fixed28_20261026.md)。本轮同位置 3×3 窗口没有胜过等大小错位窗口，且当前源码下的 M1 no-op replay 未复现历史 28-event 输出。状态更新为 `DIAGNOSTIC-COMPLETE / ALIGNED-OVER-SHIFTED-LOCALIZATION-NOT-SUPPORTED / PROTECTED-M1-REPLAY-PARITY-FAILED / NO-TRAINING`。

本轮没有训练或写入 checkpoint。下一步应先定位当前与历史 M1 replay 的首个分叉；本地窗口候选不进入训练，待 replay parity 恢复后再依证据决定是否研究更柔性的关系化输入/交互。

### Replay parity 只读复核（2026-10-26）

- 受保护 M1 YAML 与 epoch15 checkpoint 当前 SHA256 仍分别为登记值 `fab8e818...` 与 `1302de3b...`，因此这次差异不是权重或配置文件被覆盖。
- 旧 `M1_MSHA_QUERY_ROLE_FIXED_STATE_20261023_FULL28` 与本次 cross-support 输出仅保留 28 个事件帧；没有逐帧保存事件前的状态轨迹或输入/中间特征 hash。无法从现有证据恢复 27 个事件此前的首个逐帧分叉。
- 最早可直接核对的事件是 `blackbag` frame 1：旧/新 `state_before` 与 crop 完全相同，但解码框有约 `0.006 px` 的最大坐标差（旧 `[429.6161, 281.8836, 28.9526, 27.6342]`，新 `[429.6152, 281.8837, 28.9539, 27.6387]`）。这说明至少该事件上的首个可见差异发生在网络读出/数值执行层，不能由此前 tracker 状态漂移解释；其余序列的事件输入状态与 crop 已经不同，差异可能包含闭环放大。
- 对照 Oct 23 保存的 `prechange_sources.tar.gz`，当前 `V2TemplateFusion.forward` 的差异以新增可选分支和诊断代码为主；cross-support 关闭时没有看到该窗口干预改变原生 Q/K/V 公式的证据。当前 tracker 的模板更新默认关闭，普通状态提交仍走原始 `reported_state` 分支。以上只能说明代码路径表面上是 opt-in，不能证明历史运行时环境相同。
- 本轮 run manifest 记录 FP32 和 torch threads=1，但没有记录完整 Python/PyTorch/CUDA 版本、cuDNN 设定及所有可覆盖计算路径的 `TBSI_*` 环境变量。当前 shell 中没有这些变量，不代表 Oct 23 运行时也没有。runner 仅显式设置了与本诊断相关的少数开关，未对所有可选运行时开关做白名单隔离。

结论：`PROTECTED-M1-REPLAY-PARITY-ROOT-CAUSE-UNRESOLVED`。目前定位到“blackbag 第 1 帧同输入状态下出现微小输出差异”，尚未定位到首次不同的内部 tensor，也不能归因于 MSHA 算子或 tracker 状态反馈。下一步应先做单事件、同 crop 的 no-op 可重复性诊断，记录从模板输入、每个 MSHA 融合输出、CENTER score/size/offset maps 到解码框的 hash/最大绝对差；运行时采用显式白名单环境并记录软件栈、cuDNN 与完整命令。若同进程重复输出不一致，先隔离数值非确定性；若重复稳定但仍与旧框不同，再比较旧代码快照和当前代码路径。该诊断前不继续扩展或训练 MSHA 候选。

## 证据索引

- [m1_msha_feature_operator_literature_scout_20261023.md](m1_msha_feature_operator_literature_scout_20261023.md)
- [m1_early_gate_optimization_analysis_20261026.md](m1_early_gate_optimization_analysis_20261026.md)
- [m1_early_template_value_reconstruction_fixed46_20261023.md](m1_early_template_value_reconstruction_fixed46_20261023.md)
- [m1_query_conditioned_cross_value_fixed28_20261024.md](m1_query_conditioned_cross_value_fixed28_20261024.md)
- [m1_query_conditioned_cross_value_quick3_20261024.md](m1_query_conditioned_cross_value_quick3_20261024.md)
- [m1_msha_role_pair_bias_retry1_20261023.md](m1_msha_role_pair_bias_retry1_20261023.md)
- [m1_msha_query_role_holdout_quick_20261023.md](m1_msha_query_role_holdout_quick_20261023.md)
- [m1_spatial_token_bridge_fixed46_20261023.md](m1_spatial_token_bridge_fixed46_20261023.md)
- [m1_spatial_token_bridge_learnability_20261024.md](m1_spatial_token_bridge_learnability_20261024.md)
- [m1_local_cross_modal_correspondence_quick3_20261024.md](m1_local_cross_modal_correspondence_quick3_20261024.md)
- [m1_msha_differential_attention_diagnostic_smoke_20261024.md](m1_msha_differential_attention_diagnostic_smoke_20261024.md)
