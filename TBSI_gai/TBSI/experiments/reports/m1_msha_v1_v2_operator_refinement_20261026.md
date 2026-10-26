# M1 Early-Gate v1/v2 算子收敛设计（2026-10-26）

> **路线状态修订：** 本报告其余部分是历史 FCCR² 设计，供追溯，不再是当前公式或实验建议。FCCR² 将早期关系残差写入 QKV 前输入；按当前论文的 Value-only 贡献边界及用户提供的 2025–2026 相关工作对照，它已降为备选。当前待证伪候选为 FCSV-R simplex：保留原生 Q/K/attention，只按完整早期候选共支持，在前景先验上重分配跨模态来源模板 Value。当前公式与验证闸门见 [m1_dcvr_v1_v2_value_source_reallocation_20261026.md](m1_dcvr_v1_v2_value_source_reallocation_20261026.md)。此修订无代码或运行变更。

## 历史候选设计（已降级）

**本阶段历史推荐候选：前景条件双模态候选共支持重构（Foreground-conditioned Cross-modal Co-support Reconstruction, FCCR），采用二次共支持归一化。该方案已由后续 FCSV-R simplex 公式取代。**

FCCR 保留 v1 的早期模板—搜索信号和 v2 的目标前景条件，但将它们用于重建 MSHA 输入 token 之间的跨模态关系。它不再增加一组 Value 缩放器，也不扩展成搜索帧新分支。原生 MSHA 的共享 LayerNorm/QKV、注意力和定位头继续工作；唯一结构改变位于首个 TBSI-E 的共享 LayerNorm/QKV 之前。信息流锚点是 `self.blocks[2]` 产生早期 token、随后由实际 `tbsi_loc` 调用的 TBSI-E。实现时必须以运行时模块路径和张量 hook 确认，不能只写含混的“block 3”或依赖 `tbsi_layers` 列表序号。

这是“v1/v2 的简化与改进候选”，不是已验证方法。逐序列归因支持保留完整 search-support 轴，不支持直接进入训练。相较先前的 FSCIR 提案，FCCR 的关系由早期两模态模板 token 对**同一 search candidate** 的空间支持构造；本次再将线性平均支持修正为二次加权，避免稀疏但高质量的对应被全部前景 token 数量稀释。

## 证据如何限定设计

1. gate-only full-244 的序列 AUC 差均值 `-0.213 pp`，95% sequence bootstrap CI `[-1.30,+0.88]`；122/115/7 条序列涨/跌/平。少数序列的强涨跌相互抵消。
2. `10runone` 从 M1 AUC `79.48` 降 `65.37` 点；`3rdfatboy` 从 `18.49` 升 `54.29` 点。两者都带 SA/TC/BC 标签。标签不能作为可靠的场景决策开关。
3. 同状态 Early-gate A/B 的有效 GT-in-crop 样本中 CENTER top-1 / GT rank 未改变；已有轨迹分析中大退化会延后出现。故不能把后续算子目标写成“直接修复 CENTER head”；要检验的是跨模态输入关系是否在相同 crop 下改善目标相对困难候选的表征与排序。
4. M1 归档持续失配事件中，365/377 段起点的 GT 仍完整处于近似 search crop。这个时间顺序支持研究 crop 内的目标条件交互，但没有证实 MSHA 是首因。
5. 已试的局部 margin/reliability 仍是在 Value 路径上改分数或消息权重；FCCR 将变化位置移到 QKV 前。它不同于既有失败候选的位置和机制，但效用仍须实测。

以上数字来自 gate-only full-244 与 M1 归档，不使用 full15 joint 组 provenance-invalidated 的逐序列差值。场景标签彼此重叠，DEF 的正向切片仅为探索性观察，不作为 FCCR 的选择依据。

## v1/v2 做减法

| 版本/部件 | 处理 | 理由 |
|---|---|---|
| v1：早期模板—搜索相似度 | 保留，但保留完整 search token 轴，不再对每个模板 token 只取 `max` | 保留 v1 的早期条件，同时让“支持哪个候选位置”可进入后续关系构造。 |
| v1：`3→32→2` MLP 与跨模态 Value 乘法门 | 从新候选中移除；v1 原样作为实验对照 | 逐序列方向门控近乎恒定而结果方向相反；不继续增加输入量/门控容量。 |
| v2：首帧软前景权重 | 保留为来源与接收 token 的空间条件 | 使额外交互集中于目标模板区域，且是 v2 与论文目标条件叙述中已有的核心。训练/推理 mask 生成必须完全一致。 |
| v2：匹配熵、局部 margin、多模态质量 MLP | 移除 | 既有相近 reliability/margin 线路未形成可重复定位收益；属性分组也不支持用这些描述子作场景结论。 |
| v2：方向预算、前景 token 预算归一化、Value 分配网络 | 移除 | 它们仍调节已形成的 Value 消息，未触及用户要求的 MSHA 中间输入与关系形成。 |
| 新增核心 | 仅增加一个双向、目标条件的软共支持关系和一条残差重构 | 让早期空间证据改变 MSHA 实际读入的 token，进而改变 Q/K 关系和 Q/V 内容。 |

因此新版本不是“v1 gate + v2 router + FCCR”的堆叠。v1 是 parent/ablation；v2 的方向/token Value router 不与 FCCR 同时开启。

## 算子定义

在 `self.blocks[2]` 输出处取 RGB/TIR 模板与搜索 token。典型 LasHeR 配置为：

```text
E_t^m ∈ R[B,M,D], E_s^m ∈ R[B,N,D], m ∈ {R,T}
M = 64 template tokens; N = 256 search tokens
```

保留每个模板 token 对完整 search 网格的软匹配分布：

```text
p^m[i,j] = softmax_j(cos(E_t^m[i], E_s^m[j]) / τ)
```

首轮固定 `τ=0.07`，沿用仓库已有 early search relation 的默认温度，不做温度搜索；`p` 直接复用 v1 已计算的两张 cosine similarity map，只将 `max` 读数扩展为归一化候选分布。

两模态模板 token 若支持相同 search candidate，其共支持分数较高。矩阵写法为 `A = P_R P_T^T`，每个元素在 `[0,1]`：

```text
A[i,k] = Σ_j p^R[i,j] p^T[k,j]
```

使用首帧框投影到模板网格的软前景权重 `w_R,w_T ∈ [0,1]^M`。对 RGB 接收 token `i`，先在 TIR 前景来源内归一化共支持关系，再以共支持本身保留关系可信度：

```text
r_R[i] = Σ_k A[i,k]^2 w_T[k] / (Σ_k A[i,k] w_T[k] + ε)
C_R<-T[i,k] = A[i,k] w_T[k] / (Σ_l A[i,l] w_T[l] + ε)
m_R[i] = r_R[i] Σ_k C_R<-T[i,k] x_T[k]
       = Σ_k A[i,k]^2 w_T[k] x_T[k] / (Σ_k A[i,k] w_T[k] + ε)
```

反方向用同一个 `A` 的转置关系和 RGB 前景权重计算 `r_T,C_T<-R,m_T`。在首个 TBSI-E 的原生共享 LayerNorm/QKV 之前只做一次有界残差写入：

```text
x'_R[i] = x_R[i] + α w_R[i] m_R[i]
x'_T[k] = x_T[k] + α w_T[k] m_T[k]
```

`α = 0.1 * tanh(γ)` 是两个方向共享的单一可学习幅度，`γ=0` 初始化，保证 `α=0` 时与 parent 精确一致。固定状态机制筛查使用预先固定的 `α=0.1`，训练版从零幅度学习；筛查需记录 `||ΔX_FG||/||X_FG||`，不能把系数 0.1 误称为 10% 特征范数上限。首版不使用额外投影、MLP、熵、margin、阈值或可学习 mask。送入原生共享 LayerNorm/QKV 后，MSHA 自己形成新的 Q/K/V；CENTER、size/offset、损失和 tracker state 不改。

二次权重是本次的算子优化：线性版本等价于按全部前景来源数平均 `Σ_k A[i,k]w_T[k]x_T[k]/(Σ_k w_T[k]+ε)`；当只有少数 token 对应可靠时，它会被大量不对应前景 token 稀释。FCCR 用 `A²` 聚焦高共支持来源，分母 `ΣAw` 保留支持规模，得到“关系条件内容 × 关系可信度”，且没有新增参数或阈值。该选择仍有待机制比较；若高值来自共同背景干扰，它也会增强错误关系。

## 问题—机制—论文表述

**观察到的问题：** M1 在目标仍处 crop 内时存在严重定位失配；Early gate 可影响闭环结果，却在相同场景标签内部产生大幅正负差异。现有 v1/v2 的标量/Value 权重不能表达“当前哪一组前景 token 在哪个搜索候选上获得跨模态共同支持”。

**待检验机制：** 在目标前景模板区域内，依据两模态对同一 search candidate 的早期软支持，重构另一模态的模板输入。二次加权应保留稀疏但可信的互补对应、抑制弥散低共识；只有正确支持胜过错位支持，并改善目标相对 hard-negative 的读出，才说明该交互有定位效用。它是共同支持机制：单模态完全失效时共支持趋近于零，因此不声称能仅凭健康模态救回失效模态。

**论文贡献边界：** 这仍属于“退化感知的跨模态协同决策”研究，决策位置从已算出的 Value 消息预算前移到 MSHA 输入 token 的形成。它不检测或命名具体退化类别，也不声称某个属性标签必然触发模块。若该算子成为最终方法，现有草案标题中的“Value 路由”需改为“MSHA 中的目标条件跨模态输入交互/协同建模”；研究背景与失配问题保持不变。FCCR 的新颖性须另与最接近 RGB-T tracking / token correspondence 工作比较，当前不宣称首创。

## 风险与反证

- **共同干扰峰会被当成共同支持。** 两模态若同时响应同一背景干扰物，`A` 仍可能较高。必须在同 crop 下检查 GT search 区域相对最强 GT 外 hard-negative 的支持质量，并使用错位支持作反事实。
- **跨模态空间轴可能不一致。** 同一 search candidate 的假设要求 RGB/TIR 共享 crop 与有效配准。若 train/test 的数据增强对两模态不同步，`A` 不可解释；mask 与增强合同先验审计。
- **单模态严重退化时，共支持会下降。** FCCR 的残差会趋近零，避免新增不可靠交换，但它也可能放弃健康模态向退化模态传递有用内容。当前这是保守设计的明确边界，不声称能在单模态完全失效时完成恢复。
- **场景过度拟合。** 不按 full-244 的序列名、SA/TC/DEF 标签挑样或调超参；使用独立 train-side cohort，test set 只在后续封闭评测时使用。
- **增益只来自注意力数值变化。** 若 QK logits 改变但 GT rank、hard-negative margin、decoded box 不改善，停止此算子，不加 MLP/层数补救。

## 复杂度与删减量

- 新增可学习参数仅一个双向共享幅度 `γ`；移除 v1 的 `3→32→2` gate MLP 和 v2 的方向/前景 token Value router，不增加特征投影、损失或推理分支。
- `P_R/P_T` 的主要 cosine 计算已经存在于 v1 的早期模板—搜索相似度路径中。相对该路径，主要额外计算为 `A=P_RP_T^T` 的 `O(BM²N)`，以及两向 token 重构的 `O(2BM²D)`；另有 softmax 与逐元素平方。
- 以 `M=64,N=256,D=768,B=1` 估算，新增矩阵乘约 `7.34M MAC`（`1.05M + 6.29M`），不含 softmax/逐元素操作。该数值是静态估算，不代表实测 latency/FPS；实现后须实测峰值显存与 throughput。
- 风险主要来自**共同干扰物**：两模态对同一背景位置都给高支持时，二次权重会强化错误共识。另一个明确边界是单模态严重退化会使共支持变弱，不能自动转为健康模态单向救援。第一轮不再叠加背景抑制或单模态恢复分支；只有固定状态读数暴露这类具体失败后，才据证据决定是否另立单变量候选。

## 最小实验链

### 1. 输入与 mask 合同

确认 `self.blocks[2]` 的 early template/search 轴为 64/256、首个 TBSI-E 的模板输入轴为 64/64；以运行时 module path 确认早期特征确实流入目标 TBSI-E。RGB/TIR 必须共享 crop 和几何增强；训练 actor 与 tracker 均使用同一 `template_fg_masks` 生成定义。明确记录每个样本 mask 来源、增强后 template/search tensor 与 manifest。历史 DCVR 训练 uniform fallback、测试真实 mask 的差异不得带入新候选。

### 2. 固定状态机制判别，不训练

先完成单事件同 crop 的 no-op 重复回放，解决或隔离当前 replay parity 差异；然后在预先冻结的 LasHeR train-side matched-good/severe-onset 事件上，用同一模板、真实前一状态、同 crop、同一当前源码配对比较：

1. `native`: `α=0`。
2. `linear-correct`: 上一版按全部前景来源数平均的线性共支持，作为公式对照。
3. `FCCR²-correct`: 使用本版 `A²/ΣAw` 的二次共支持输入重构。
4. `FCCR²-shifted`: 对一模态的 16×16 search 支持图施加预先固定的循环位移，再按相同公式构造 `A`；保持 token 数、概率总量与前景权重合同一致。

固定同一 TBSI-E 实例、同一 `α` 和同一 crop/state。逐序列记录 `p_m` 在 GT 区域与最强 GT 外 hard-negative 的质量、`A/C/r` 分布、输入残差范数、QK logits/四块 attention、GT CENTER rank/margin、size/offset 和同 crop decoded IoU。对 linear 与 FCCR² 另做残差范数匹配的只读分析，区分“写入更强”与“来源选择更准”。报告序列数、逐序列涨跌与 bootstrap 区间，不只报帧宏平均。

**通过签名：** FCCR²-correct 胜过 FCCR²-shifted，并在残差范数匹配后仍比 linear-correct 更好地区分目标与 hard-negative；CENTER rank/margin 或 decoded IoU 至少一个主定位读数在多个序列上同向改善，stable-good 事件没有系统性伤害。**停止签名：** correct 与 shifted 无差别、两者同样提高干扰峰、优势在残差范数匹配后消失、只有 QK 变化没有定位读出变化，或改善只来自单个序列。

当前 M1 fixed-state replay parity 根因仍未解决，故该步骤需以同一运行中的 `native` 作为本地对照，不得宣称直接重现受保护 M1。FCCR 的零幅度 parity 和同进程重复性必须先通过。

### 3. 训练验证准入

只有固定状态通过后才做短训。若目标是声称“在 v1 基础上改善”，必须在相同 SOT 预训练初始化、seed、manifest、优化器、更新步数、增强后输入轨迹和 checkpoint epoch 下，比较重训的 Early gate v1、FCCR 替代版，并保留 native control。仅拿历史 full15 v1 单次点估计对比新候选，不是严格增量归因。FCCR 单独启用，不与 v1 gate 或 v2 Value router 叠加。

短训前仍需先解决 quick3 control repeat 的 `11.15 pp` 差异和 post-augmentation tensor hash；否则短训正负排序只能算探索性，不进入算子晋级依据。任何 full-data/full-15 结论均另需匹配协议与完整测试。

## 旧候选阶段结论（已被后续修订替代）

- **当时推荐的设计：** FCCR²，替代 v1/v2 的消息 gate/预算网络，插入首个 TBSI-E 原生共享 QKV 之前。该建议已被 FCSV-R simplex 的 Value-only 论文边界取代。
- **不支持的说法：** FCCR 已解决 M1 首因；任何场景属性可直接触发可靠性权重；全量性能会提升；算子已具备论文新颖性。
- **当时的下一步：** 先完成单事件 replay parity 和输入合同核对，再做固定状态机制筛查；这不是当前的执行方案。
- 本报告仅做算子设计，不改模型代码、不训练、不推理、不写 checkpoint。

## 依据

- `experiments/reports/m1_early_gate_sequence_scene_attribution_20261026.md`
- `experiments/reports/m1_msha_v1_v2_operator_attribution_20261026.md`
- `experiments/reports/m1_msha_internal_input_interaction_audit_20261026.md`
- `experiments/reports/m1_early_conditioned_msha_gate_proxy_20261021.md`
- `experiments/reports/dcvr_manuscript_draft_20261026.tex`（DCVR-v2 是提案；mask 训练/测试合同差异须避免）
