# M1 DCVR full15 归因与算子设计分析（2026-10-26）

## 判断摘要

本轮结论为：`FULL244-COMPLETE / IMPLEMENTATION-ACTIVE / TRAIN-TEST-MASK-CONTRACT-MISMATCH / M1-COMPARABILITY-UNRESOLVED / PERFORMANCE-GAIN-NOT-SUPPORTED`。

DCVR 确实非退化地重分配了 template MSHA 的 Value token，但现有证据没有证明这些权重对应“对当前目标有用的消息”，也没有证明它改善 CENTER 选择。更关键的是，完整训练时 DCVR 没拿到前景 token mask，完整测试时却拿到了；这不是纯粹的算子效果差，而是训练与推理实际执行的函数不同。与此同时，当前磁盘中的 M1 预测结果按现行评测器复算不等于受保护 M1 日志指标，所以逐序列配对结果不能用作受保护 M1 的正式因果归因。

当前版本不应作为有效改进晋级。先完成评测来源核验；若保留 DCVR，第一项实验应只修复前景 mask 的训练/推理合同，公式、alpha、网络位置和训练计划全部固定。

## 实验与比较口径

候选从 `TBSITrack_SOT_Pretrained.pth.tar` 初始化，seed 42，fresh AdamW/StepLR，15 epochs；每 epoch 59,904 samples，batch 16、gradient accumulation 8，有效 batch 128。DCVR full-244 测试使用 epoch15、FP32、template update off；前 99 条序列串行完成，剩余 145 条使用 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、tracker `threads=2` 断点续测。

候选记录的 FPS 为 9.73，但 M1 没有可比 FPS。测试期间 DCVR 还逐次向 JSONL 写入路由诊断（累计 659,630 条），且续测改用了 `threads=2`；因此该值不能解释为算子单独的推理开销，也不应作为对 M1 的吞吐比较。

| 指标 | DCVR 官方日志 | 受保护 M1 日志 | 原始点差 | 当前磁盘 M1 预测复算 | DCVR 减当前 M1 复算 |
|---|---:|---:|---:|---:|---:|
| AUC / Success | 55.96 | 56.21 | -0.25 pp | 56.766 | -0.810 pp |
| OP50 | 68.06 | 68.33 | -0.27 pp | 69.071 | -1.010 pp |
| OP75 | 47.09 | 47.13 | -0.04 pp | 47.692 | -0.601 pp |
| Precision | 69.72 | 70.21 | -0.49 pp | 70.931 | -1.207 pp |
| Norm Precision | 66.04 | 66.20 | -0.16 pp | 66.912 | -0.868 pp |

DCVR 当前复算与它自己的官方 full-244 日志吻合（AUC 55.9556 -> 55.96）。但同一分析脚本把现存 M1 结果目录复算为 AUC 56.7659，而受保护 M1 的 full-pipeline 日志明确为 56.21。M1 受保护记录的软件是 PyTorch 1.9.0+cu111/CUDA 11.1；DCVR 本次记录是 PyTorch 2.0.0+cu118/CUDA 11.8。现有证据没有解释 M1 目录与原日志的差异，故不得把 `per_sequence_paired.csv` 的差值、bootstrap 区间或序列排名当成与受保护 M1 的严格配对结论。表中的 `-0.25` 等只保留为两份已登记汇总点估计的算术差，不构成因果估计。

逐序列统计仍可描述这两个当前目录间的差异，但只属探索性：AO 有 125 条上升、119 条下降，均值差 -0.00831；AUC 有 124 条上升、117 条下降、3 条持平，均值差 -0.810 pp、中位数 +0.036 pp。少数大幅下降抵消了较多小幅上升：最大 AUC 下降包括 `3rdfatboy` (-54.18 pp)、`leftfarboycomingpicktheball` (-44.07 pp)、`bluegirlbiketurn` (-35.18 pp)；最大上升包括 `right5thflag` (+19.07 pp)、`shinycarcoming` (+16.97 pp)、`girlrightthewautress` (+16.64 pp)。对当前目录配对做的 sequence bootstrap 区间中，AUC 为 [-1.666, -0.062] pp，Norm Precision 为 [-1.954, +0.078] pp；因 parent 复算不匹配且只有单 seed，这些区间不能修复比较合同问题，也不覆盖 seed/训练不确定性。

## 算子的信息流

设早期 RGB/TIR template token 为 `T_m`、search token 为 `S_m`。代码在共享主干 block index 2（第三个 block）捕获早期 token，并把同一份早期上下文送进 TBSI fusion 层 index `[3, 6, 9]`。对每种模态、每个 template token `i`，计算：

```text
C_m[i,j] = normalize(T_m[i]) dot normalize(S_m[j])
p_m[i,:] = softmax(C_m[i,:] / 0.07)
H_m[i] = entropy(p_m[i,:]) / log(N_search)
M_m[i] = sigmoid((top1(C_m[i,:]) - top2(C_m[i,:])) / 0.07)
r_m[i] = (1 - H_m[i]) * M_m[i] * (0.75 + 0.25 * fg_m[i])
```

每个模态的 `r` 先按 token 均值归一化。随后用对应 RGB/TIR template token 的余弦相似度 `a[i]` 乘上共同系数 `0.5 + 0.5 * sigmoid(a[i] / 0.10)`，再按均值归一化，得到 `r'_m`。最后：

```text
scale_m[i] = 1 + 0.25 * (r'_m[i] - 1)
V_m[h,i,c] <- scale_m[i] * V_m[h,i,c]
```

也就是说，该系数对一个源 token 的所有 attention heads 和所有 value channels 完全共享。随后沿原始 `QK^T` logits 计算 attention，再以 `attention @ V` 聚合、过输出投影并进入原有 residual。由于缩放发生在联合 attention 之前，该源 token 会被所有 query 读取，所以它会同时改变同模态 self-message 和跨模态 message；它不是只给跨模态边加权。DCVR 不改 Q/K、attention logits、CENTER head 或在线模板缓存/更新规则；但融合后的模板表示会经 residual 继续进入后续网络，最终可能影响 search 表示和 CENTER 框。Value 本身没有被单独旋转，多个 token 的加权聚合方向则可能变化。

## 已有证据能说明什么

**实现确实起效。** 训练诊断 170,355 条、测试诊断 659,630 条；alpha 始终为 0.25。归一化后的 source scale 均值约为 1，训练 RGB/TIR token-scale 标准差约 0.105/0.092，测试约 0.112/0.110。说明路由不是 identity/no-op。统计没有记录 per-token 极值，所以不能由均值和标准差排除少数 token 的大幅放大。

**前景 mask 合同不一致是已确认的实现问题。** `TBSITrackActor.forward_pass` 只在若干配置开关（如 DER、frequency、template expert、local correspondence 等）下构造 `template_fg_masks`，没有把运行时 `TBSI_DCVR_MODE=on` 纳入条件。因此这次 DCVR 训练阶段传入 `None`，算子退化到全 1 mask，诊断 fallback 比例为 100%；测试 tracker 则总是依据初始模板框构造并传入 foreground mask，fallback 为 0%。训练实际学到的是“不含框内先验”的 Value 路由，测试却加上了这一项。此差异可从训练 actor 与测试 tracker 代码及诊断直接核实，优先级高于继续调 entropy 温度或 alpha。

**置信分数不等同于目标效用。** `1 - entropy` 和 top1-top2 margin 只表示搜索响应是否集中、是否有明显第一名；两者都可能对稳定背景纹理或错误 distractor 给高分。算子没有检查 top-1 是否落在可信目标区域、RGB/TIR 是否定位到相容位置、也没有用 CENTER/框质量校准“这个源 token 的消息是否帮助定位”。因此现有统计能证明“模型在重分配”，不能证明“路由器识别了有用 token”。

**agreement 项在观测均值附近很可能饱和。** 测试平均 cosine agreement 为 0.6005。将均值代入 `sigmoid(a / 0.10)` 得约 0.9975，对应共同乘数约 0.9988，接近 1；训练均值 0.5883 也类似。这只是均值处的量级说明，不代表 token 级精确分布，但提示 agreement 分支在常见正相似区域可能贡献很小。更根本的问题是两种模态乘同一个 agreement 系数：当两模态因退化而分歧时，它倾向于一起削弱对应 token，而不是保留更可靠模态、允许互补。

**训练有有效监督路径，但 DCVR 本身没有可学习参数。** 早期特征未 detach，熵/相似度/余弦路径可把梯度传回共享主干；但没有专门损失监督 token scale 与目标定位效用的对应关系，规则本身的温度、组合和 alpha 均固定。采样 loss 从 epoch1 的约 1.067 降至 epoch10 的约 0.873，后续有波动，epoch15 约 0.910，没有明显全面训练崩溃的证据。梯度 CSV 的 270 次采样中有一次 NaN 记录（epoch1/step600），且每条采样均有 2/228 个参数无梯度；日志没有记录这两个参数名。`total_norm=0.1` 是在 grad clipping 之后采集，不能用它推断原始梯度总量。这个 NaN 需保留为未解释诊断，但不足以单独解释最终掉点。

**序列归因无法连回路由信号。** 测试诊断没有 sequence/frame key；而且 JSONL 有 134 条损坏/交错源行、恢复出 1,350 个拼接对象。恢复后的总体统计可作实现活动性证据，不能按序列关联 gate/scale 与 CENTER 排名、框误差或场景退化类型。日志写入问题影响诊断可追溯性，不等同于 tracking prediction 文件损坏。

## 与 M1 问题链的对应

M1 已观察的系统级失败链是 crop 内中心选择失配后，反馈造成漂移、目标出界并恢复不足。DCVR只改模板融合层中的 Value 源贡献；它可能间接改变后续 search 表征，但没有直接改 CENTER 排序、解码、crop/状态反馈或恢复状态。此前 M1 固定状态诊断中，虽然 source scale 非零，IoU 平均差约 -0.00062，CENTER 排名仅 7 个事件中 1 个改变；退化 smoke 在 blur/contrast/lowlight 下方向不同。这些证据支持“消息权重被改了”，不支持“框内中心失配链已被修复”。此外这次全量测试明确 template update off，不能把它解释成对在线模板更新问题的验证。

因此现阶段的因果结论应写为：

- **已排除：** DCVR 纯粹没有执行或数值恒等。
- **暂不支持：** 当前 DCVR 改善 M1 full-244；现存汇总点估计全低，固定状态 CENTER 排序也没有稳定改善。
- **尚未区分：** full15 负向信号中，多少来自“尖锐度不等于目标效用”、agreement 项冗余/对称抑制、训练/测试 mask 不一致，以及代码/数据/软件结果来源差异。
- **支持下一步单变量检验：** 先核实 M1 结果来源和比较口径；若继续 DCVR，只先修复 train/test foreground-mask parity，再复测同一个算子。

## 算子设计评价与下一步

优点是边界清楚、插入点贴近 M1 的 template MSHA、计算量相对主干较小，Q/K 与 CENTER 结构不动，且零均值归一让它主要做 token 间预算再分配。作为工程诊断算子，这些选择可解释。

设计上的主要缺口不是“信号数量不够”，而是**信号与要优化的量没有对齐**：全搜索 entropy/margin 测的是匹配分布形状，不是目标正确性；RGB/TIR cosine agreement 测的是特征相似，不是互补贡献；同一个 block-2 路由图又被复用于三个不同深度的 Value 表示。此外，token scale 同时作用于该 token 参与的 self/cross 两类消息，没有区分“抑制不可靠模态发出的消息”和“保留该模态自身稳定表征”。这样会增加算子项数，却仍然缺少 token 对目标定位的效用监督。并且 alpha=0.25 不是严格的 ±25% 上下限：归一化权重可在 64 个 token 间高度集中，理论上 scale 最大可到 16.75；当前日志只给标准差，没有 per-token min/max 或分层统计，不能排除偶发尖峰。

建议按顺序做：

1. 先核对受保护 M1 日志、当前 M1 预测目录、原始数据路径/annotation 版本及当时评测环境；在解释 AUC `56.21` 与重算 `56.77` 差异前，不把逐序列配对分析用于论文归因。
2. 若继续 DCVR，在训练 actor 的 foreground-mask 生成条件里显式纳入 DCVR runtime flag，使 train/test 使用同一 `template_box_to_token_weights`；保持其余算子、alpha 和训练日程不变。该项能直接检验已发现的合同错误。
3. 做下一轮训练前先固定并审计 source scale 的 token 级 min/max、分 TBSI 层统计以及每序列/frame 键；检查 scale 是否极端、三个层是否只在复用早期权重。必要时使用有硬界的平滑映射，但不要与 mask-parity 修复同时改。
4. 只有在合同修复后仍有收益/可解释信号，才单独替换“sharpness/agreement”信号为更接近目标定位效用的量。方向上应保留模态非对称性，并测量对应 search 支持是否落在可信目标区域；不要同时改 Value、QK、CENTER 和在线模板状态。

离线证据包括 `tools/analyze_m1_dcvr_full15_20261025.py`、`output/experiments/M1_DCVR_FULL15_SOT_20261025/attribution/summary.json`、`attribution/per_sequence_paired.csv`、full-pipeline/test/analysis 日志，以及两份 DCVR JSONL 诊断。归因脚本现已增加受保护 M1 锚点一致性检查；现有摘要明确标记配对推断无效。

本轮只做既有日志、代码、预测文件和离线统计审计；没有启动训练、跟踪推理或删除任何实验资产。
