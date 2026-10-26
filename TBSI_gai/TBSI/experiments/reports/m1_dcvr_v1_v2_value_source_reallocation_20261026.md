# M1 Early-Gate / DCVR-v2 Value 源重分配设计（2026-10-26）

> **历史方案，非当前执行计划（2026-10-26 清理标记）：** 本文原始数据、引用和推导保留；其中候选优先级、准入条件和下一步命令不再作为当前指令。当前范围见项目 `README_当前研究.md`；本标记不表示所有假设已被证伪，也不批准新训练。清理前原文见 Git `c0f1aff`。


> **后续审计更正（同日）：** 上轮“来源机制已被否定/停止整条 Value 内容路线”的结论撤回。现有运行保留 Early gate，且 alpha 非零时硬删除零前景来源，alpha0 则特判恒等；尚缺 mask-only 对照。query/boundary CSV 测量接收输出，未测量来源 Value 贡献，因此不能由 r≈0.025 推出来源重分配无效。下方原始数字保留，旧强归因以本更正为准。当前状态为 `SOURCE-MECHANISM-INCONCLUSIVE / CONTRACT-REPAIR-FIRST / NO-TRAINING`。详见 [算子建议与审计更正](m1_early_gate_value_interaction_recommendation_20261026.md)。

> **公式修订：** 当前 FCSV-R 用“前景来源分布重分配”替代上一版的 `κ·d` token 缩放。它在前景来源先验上守恒系数质量，避免把 gate 均值误写成 Value 消息守恒；实际消息范数仍须实测。以下公式与实验判据为当前版本。

## 当前判断

**FCCR² 不再作为当前主算子。推荐的唯一待检验方向是“前景条件共支持 Value 源重分配”（FCSV-R）。** 它保留 Early gate v1/v2 所在的 TBSI 模板 MSHA 跨模态 Value 路径，保留首帧前景条件；将 v1 的 `max` 搜索匹配改为完整候选分布，并将两模态对同一搜索位置的共支持转成来源模板 token 的 Value 权重。

本算子不写入 QKV 前输入，不直接改 Q/K 投影或 attention 公式，不改注入层的同模态 Value、不改 CENTER 或模板更新。由于改后的 Value 输出会进入后续层，后续层的 Q/K 数值可能间接受影响；因此论文只能说“算子不直接重写 Q/K/attention”，不能声称全网各层 attention logits 都不变。相较 FCCR²，这个位置收敛回当前 DCVR 论文的 Value-only 边界；相较 v1/v2，它删除了多路可靠性 MLP、方向预算和额外 token router，只留一个由前景条件共支持驱动的有界重分配。

这仍是待证伪设计，不是已验证的机制，也不意味着已有文献没有相似思想。用户提供的近年相关工作审阅指出，TUMFNet/UATrack 已涉及模板—搜索证据与模态/token 级可靠性，QSTNet/MRTTrack 已涉及搜索 token 选择，CGATrack 与本研究的双向质量引导最接近，CAFormer 涉及相关/注意力图修正。FCSV-R 可检验的窄区别是：**固定原生 MSHA 的 Q/K 和 attention，只在既有模板 MSHA 中重分配跨模态来源模板 Value，分数来自双模态早期搜索分布的共同候选支持，并由接收模板前景条件化。** 尚未完成逐算子代码重合审查，不能宣称首创或优于这些工作。

## 问题与证据边界

| 观察 | 设计含义 | 证据边界 |
|---|---|---|
| Early gate v1 每个模板 token 的 RGB/TIR 搜索相似度取 `max`，再由 `3→32→2` MLP 得到跨 Value 方向缩放；逐序列门控均值近乎固定，full-244 的涨跌高度异质 | 保留搜索位置分布，不再用一个 max 标量代表所有候选；按 token 分配跨模态来源 | 逐序列结果来自 gate-only 历史输出，有来源限制；不能据此称 v1 的机制已经成立 |
| Early gate full15 的 aggregate AUC 点估计比登记 M1 高 `+0.35 pp`，但序列 bootstrap 区间跨零，旧逐序列输出又有 provenance 问题 | 用 Early gate full15 checkpoint 做固定状态算子父模型；并排保留 gate-bypass native 作为同 checkpoint 操作对照 | 这是选父模型的理由，不是性能增益或机制成立证据 |
| M1 多数严重失配事件起始时 GT 仍在近似 search crop；同状态 gate A/B 的 CENTER top-1 / GT rank 未稳定改变 | 把本轮目标限定为检查跨模态 Value 的来源内容是否目标相关，不触碰 CENTER 或状态回路 | 不能证明 MSHA Value 是 M1 首因 |
| 既有 fixed28 只在一层 cross-QK logits 上施加同位置 3×3 局部 key mask；aligned 相对 shifted 的 sequence-macro IoU 为 `-0.000580`，CENTER rank `0/27/1` | 不复用硬局部 attention mask；本轮保留完整全局搜索分布，只改变 cross-Value 的前景来源分配 | 它不检验完整分布共支持或 Value source reallocation；但说明“同位置/空间对应”本身不足以作为收益依据。其旧/新 M1 replay parity 未通过，不能跨运行比较 |
| query-conditioned Value、局部支持、foreground-query 限制、token reliability 和早期 Value 重构已有负面或不确定结果 | 不复用随机 query-key 低秩残差、固定局部窗口、熵/margin gate 或单纯前景 query 截断；新候选必须用真实 Value 消息贡献和 target/background 读数证伪 | 这些结果各自的合同、训练进度和来源不完全一致；仅排除对应公式，不排除所有 Value 机制 |

因此当前可检验的问题更窄：**在目标仍位于 crop 内的状态中，Early gate 的 `max` 证据压缩是否丢失了两模态对候选位置的共识，使跨模态 Value 来源 token 无法按目标前景相关性重分配？** 这是机制假设；若共支持排序不优于错位共支持，或只改变整体消息幅度而不改善目标/困难负样本区分，就停止本候选。

## 算子定义

令 `m∈{R,T}` 表示 RGB/TIR。取 `self.blocks[2]` 的早期模板与搜索 token：

```text
E_t^m ∈ R[B,M,D], E_s^m ∈ R[B,N,D]
M=64, N=256 in the current LasHeR setup
```

沿用 v1 已有的 cosine similarity，保留完整 search 轴：

```text
P_m[i,j] = softmax_j(cos(E_t^m[i], E_s^m[j]) / τ)
τ = 0.07 (fixed for the first mechanism test)
C[i,k] = Σ_j P_R[i,j] P_T[k,j]
```

`C∈[0,1]^{B×M×M}` 表示 RGB 模板 token `i` 与 TIR 模板 token `k` 对相同 search candidate 的共同支持。这里用 `C` 避免与原生 MSHA attention 矩阵混淆。`P_R/P_T` 对应 v1 已计算的相似度图；额外关系只在早期模板—搜索之间形成，不成为新的 search feature branch。

对方向 `R←T`，用 RGB 接收模板前景分布汇总每个 TIR 来源 token 的共同支持。先将软前景 mask 归一化为来源先验；若某个样本的前景权重和为零，该方向直接 no-op：

```text
π_m[k] = w_m[k] / Σ_l w_m[l]
s_T[k] = Σ_i π_R[i] C[i,k]
z_T[k] = tanh(log(s_T[k] + ε) - Σ_l π_T[l] log(s_T[l] + ε))
q_T[k] = softmax_{k:π_T[k]>0}(log π_T[k] + α z_T[k])
g_T[k] = q_T[k] / π_T[k], for π_T[k] > 0
g_T[k] = 1, otherwise
```

反方向交换 RGB/TIR 并使用 `C^T`：

```text
π_R[i] = w_R[i] / Σ_l w_R[l]
s_R[i] = Σ_k π_T[k] C[i,k]
z_R[i] = tanh(log(s_R[i] + ε) - Σ_l π_R[l] log(s_R[l] + ε))
q_R[i] = softmax_{i:π_R[i]>0}(log π_R[i] + α z_R[i])
g_R[i] = q_R[i] / π_R[i], for π_R[i] > 0
g_R[i] = 1, otherwise
```

`q_m` 是新的前景来源分布，`π_m` 是首帧 mask 给出的原始前景来源分布。`α=0` 时 `q=π`、`g=1`，严格退化为原始跨 Value 消息；训练时共享幅度 `α=0.1 tanh(γ)`，`γ=0` 初始化。首轮固定状态机制筛查预先固定 `α=+0.1`，只检验“更高共支持应获得更多来源份额”的单向假设。训练和推理的 mask 定义必须一致。

由于 `z∈[-1,1]` 且 `|α|≤0.1`，忽略数值稳定项时 `g` 的理论范围约为 `[exp(-0.2),exp(0.2)] = [0.819,1.221]`。前景来源系数质量严格守恒：`Σ_k π_m[k]g_m[k]=Σ_k q_m[k]=1`。这是**mask 先验下的来源系数预算守恒**，不是 `A_native(gV)` 的 attention 加权消息范数或方向守恒；后者必须记录真实范数、余弦与前景/背景分解。`ε=1e-6`。背景来源 token 的 `g=1`，本机制不直接抑制背景 Value，也不进行整模态可靠性判断。

只在对方模态的 cross-Value 消息中应用该来源权重：

```text
M_R = A_native_RR V_R + A_native_RT (g_T ⊙ V_T)
M_T = A_native_TT V_T + A_native_TR (g_R ⊙ V_R)
```

`A_native` 是当前 TBSI 层原生 joint Q/K attention；在被干预的该层，`g_T/g_R` 不进入 Q、K 或 attention logits。`V_T` 对 RGB self-message 仍使用原值；`V_R` 对 TIR self-message 同理。新权重在 v1 原有 TBSI-E locations (3/6/9) 使用同一个早期 `P/C` 与共享 `α`，不新增按层调参。后续层的 logits 允许因前层 Value 输出变化而间接变化。训练与推理必须使用同一前景 mask 定义。

信息流可以压成一条：`early template/search features → P_R、P_T 全候选分布 → C 的同位置共支持 → 接收前景聚合得到来源分数 s → 在前景先验 π 上形成 q → g=q/π → 只重分配 cross-modal template Value → 原生 MSHA attention 输出 → 后续 CENTER/框解码`。这里不预测退化类别、不改变单模态分支，也不增加 search feature branch。

### 为什么不是另一组质量门

- v1 的 MLP 将三个标量映射为两个方向增益；FCSV-R 不学习“哪种模态整体更可靠”，而是用完整候选分布的共支持重排**当前 cross-Value 的前景来源模板 token 预算**。
- v2 的熵、margin、模态质量 MLP、方向预算与 token budget 一并删除；不靠堆叠更多可靠性特征来解释涨跌。
- 早期全搜索位置用于形成 `C`，但最终只改变 cross-Value 来源，不改变 Q/K 关系或 search token 表示。它仍是 Value 路由方法，保留当前稿件的论文范围。
- 新算子仍可从代数形式看作有界 token 权重；它的区别不是“没有门控”，而是一个受首帧前景先验约束、由双模态候选共支持驱动的来源分布运输。论文表述应具体写“前景来源分布重分配”，不写泛化的质量门或概率不确定性。
- 高共支持仍可能来自两个模态共同响应的背景干扰；前景 mask 不是目标正确性的证明。正因如此，正确共支持必须与等质量错位支持比较，并跟踪 GT/hard-negative 实际消息及定位读出。

## 复杂度与风险

- 新增一个共享幅度参数 `γ`；不保留 v1 `3→32→2` gate MLP，不加 v2 方向 budget、熵/margin predictor、Value 投影或额外 loss。
- 两模态早期相似度 `P_m` 在 v1 已计算。新增主矩阵计算为 `C=P_RP_T^T`，复杂度 `O(BM²N)`；每方向 token 权重归约 `O(BM²)`，每个 TBSI-E cross Value 分支增加一次 token 维乘法。可缓存 `C`，不重复计算。
- 以 `M=64,N=256,B=1` 估算，新增共支持矩阵约 `1.05M MAC`；这是理论值，不是实测 FPS/延迟。实际额外开销需实施后测量。
- **共同背景假共识：** 两模态共同响应干扰物时，`C` 也高。背景干扰控制若使 hard-negative Value 贡献升高，候选即被证伪。
- **单模态失效边界：** 当共支持趋于均匀或无区分时，`z` 趋近零、权重趋近恒等值；它不会主动放大健康模态、压低另一整路消息。因此本候选只主张“有共同候选证据时的来源 token 重分配”，不主张单模态完全失效恢复或通用可靠性估计。
- **前景 mask 偏差：** 首帧框投影误差、形变、裁剪或不同步增强可能把背景当目标。mask 与 RGB/TIR 几何增强必须一致，新增候选不允许训练 uniform fallback、测试真实 mask。
- **预算与范数混淆：** `Σπg=1` 只守恒首帧前景先验下的来源系数质量；attention 对不同来源的权重、Value 向量方向和相消效应仍会改变消息。所有结论必须报告每方向真实 cross-message norm/cosine、前景/背景来源分解、CENTER 和框回放，不能用 simplex 守恒替代有效性证据。

## 与近年工作的边界

以下区分基于用户提供的文献对照，仍需在实现/论文级逐算子复核：

| 工作簇 | 已覆盖的部分 | FCSV-R 的窄边界 | 不能声称 |
|---|---|---|---|
| TUMFNet / UATrack | 模板—搜索证据、不确定性/模态与 token 级权重，另涉及模板状态或更新 | 不做 subjective-logic/Gaussian uncertainty，不做在线更新；只改既有 MSHA 的 cross-modal template Value source | 首次用模板—搜索关系判断 RGB-T 可靠性 |
| QSTNet / MRTTrack | 按搜索 token 或目标相关 token 选择/加权 | 不重加权 search token，不改 search representation；目标对象是 MSHA cross-Value 中的 template source token | 首次做 token 级选择 |
| CGATrack | 双向质量引导、source ability 与接收侧需求、注意力/特征传递 | 不预测通用 source reliability/receiver need，不改 attention/QK；用早期共同候选支持构造来源 token 分配 | 首次做双向质量路由或 source/receiver 协同 |
| CAFormer | 相关图/attention correction | 原生 attention matrix 固定，变化只发生在跨模态 Value source | 首次纠正跨模态匹配或 attention |
| MoETrack | 模态失效时的专家选择 | 不建 experts，不做何时融合的全局决策 | 解决双模态同时失效或全局模态切换 |

若 CGATrack 源码显示其已在相同 TBSI 模板 MSHA 路径，用双模态候选共支持选择 cross-Value template sources，则当前差异不足以支撑方法贡献，应停止而不是换名称。

这是一项刻意收窄的增量机制，不是新 attention 家族；即使 fixed-state 机制通过，也只说明“前景来源份额按全局候选共支持重排”在本设置下有依据。若 source-level 代码核查发现 TUMFNet/UATrack/QSTNet/CGATrack 已实现实质相同的分布构造与 Value 作用点，或 correct/shifted 不能显示目标特异性，就不应靠改名或堆训练量继续包装。

## 实验闭环

### 0. 合同先决条件

先完成单事件同 crop 的同进程 no-op 重复性与层级 tensor hash，并确认新分支 `α=0` 与本次 parent 的逐元素 parity。受保护 M1 的跨脚本 replay parity 根因仍未解，quick3 control 又有 `11.15 pp` 重复差；这两项阻止训练和跨运行性能归因，但不阻止从同一进程捕获的固定 tensor 做配对机制反事实。

实施前定义新独立 feature flag/config/output。`off` 与 `α=0` 必须逐元素一致；在相同输入张量上确认被干预层的 Q/K、attention logits 和同模态消息不变，只有 cross-Value 来源系数变化；再记录后续层自然产生的差异。校验 `P/C/g` 轴和有限值、首个 Early gate location mapping、三处 cross-Value source 方向、state dict、mask 训练/推理一致性。不得改受保护 M1 配置和结果。

### 1. 固定 crop/state 机制筛查，不训练

以 Early gate full15 checkpoint（登记 SHA256 `e38d18fe50c2895e75e4b9392bf22d9323bb8e6ad31ccbe39ad8c038e3fc1811`）作为算子父模型，使用 train-side fixed states；不使用 provenance 有争议的 full244 逐序列输出来挑事件。若现有 fixed28 缺该 checkpoint 对应的 tensor/mask 记录，就从 train-side 建立并哈希新清单。每个事件固定同一初始化模板、真实前一状态与 crop；从同一捕获的早期张量分叉对照，避免重复跑 tracker 后状态漂移。配对分支为：

1. `v1`: 原样运行 Early gate checkpoint，作为实际父方法的描述参考。
2. `native`: 同一 checkpoint 下旁路 v1 gate、关闭 FCSV-R；这是 FCSV-R 的同 checkpoint no-op parent。
3. `FCSV-R correct`: 同一旁路设置，仅开启原始两模态共支持，固定 `α=+0.1`。
4. `FCSV-R shifted`: 与 correct 完全相同，但只将一模态 `P_m` 的 16×16 search 轴按预注册位移循环平移；保留该模态的概率分布、峰度和 mask 质量。位移值须在读定位结果前写入 manifest。

`FCSV-R correct - native` 识别算子在该 checkpoint 上的局部机制效应；`FCSV-R correct - shifted` 检验效果是否依赖正确的跨模态候选对应；`FCSV-R correct - v1` 只作替代方案参考，不作单变量因果结论。候选分支在三处 TBSI-E 使用相同早期 `P/C` 与一个共享 `α`，并逐层记录增量；不额外搜索层位置。

对相同 frame/crop 做成对输入干预：clean、一个预先固定强度的 RGB 低照度/亮度衰减、一个预先固定强度的 TIR 噪声/对比度损伤。第一轮不注入合成背景 patch；使用原 crop 中最强 GT 外 hard-negative 检查双模态共同背景假共识。每次只改变一种因素，跟踪器 state 不提交更新。

必须记录：

- 每方向 `π/s/z/q/g` 的分布、前景 token 排序与 `Σπg` 数值检查，以及正确/shifted 对照；
- 原始与重分配后的 `A_native(V)` 和 `A_native(gV)` 的 L2 norm、cosine、差分范数；按接收前景/上下文 queries 与来源前景/背景 values 分解；
- GT 与最强 GT 外 hard-negative 的 CENTER rank/margin、size/offset、同 crop decoded IoU；
- 各事件配对差、序列均值/中位数、胜负数和 bootstrap 区间，报告单序列退化，不以帧宏平均替代序列归因。

**通过签名：** 正确共支持相对 shifted control，稳定提高目标前景来源的实际 Value 贡献相对 hard-negative/background 的选择性；差异不能仅由 `||M_cross||` 的整体升降解释。该差异在至少一种预注册单模态退化下按来源侧出现，并在 CENTER rank/margin 或 decoded IoU 至少一个读数上传到多个序列；clean/stable-good 不出现系统性伤害。前景来源系数质量守恒不等于整路方向消息守恒，因此不以 `g` 均值或总范数作为通过条件。

**停止签名：** correct 与 shifted 无差异；权重变化只放大共同背景 Value；实际 cross-message 只有范数变化而来源构成/定位不改善；只改变 feature/attention summary 而 CENTER/box 无改善；或收益仅来自单个事件。停止后不加方向 budget、MLP、熵、margin 或第二个交互分支。

### 2. 匹配短训准入

固定状态机制通过、mask parity 和重复性合同通过后，才比较同一 SOT pretrained 初始化、seed、manifest、优化器、总 optimizer steps、schedule、增强后 tensor 轨迹和 epoch 的两组：重训 Early gate v1 control 与 FCSV-R 替代候选。候选只替换跨模态 Value 权重算子，不与 v1 gate 或其他 v2 分支叠加。现有 Early gate full15 点估计与来源不一致逐序列表不能替代 matched control。

快速训练结果仍只作筛查；除固定状态机制准入外，还必须先解释 quick3 control 的 `11.15 pp` 重复差并保存增强后输入 tensor hash，才能用训练比较候选。只有匹配增益和有效消息/定位链路在更大预注册 cohort 上复核后，才进入全量 LasHeR 测试。该路线目前不对 RGBT234 作结论，也不触碰在线模板更新。

## 当前状态

- **当前候选：** FCSV-R simplex form，Value-only，作为 Early gate v1/v2 的单算子替代设计；相对前版删除 `κ·d` 标量调制，改用前景来源分布上的 softmax 预算重分配。固定状态机制筛查以 Early gate full15 checkpoint 为父模型，并用同 checkpoint 的 gate-bypass/native 分支识别算子效应。
- **降级候选：** FCCR²，保留为 QKV-input 备选；若论文目标继续严格限于 DCVR-v2 Value 路由，不按它写方法主张。
- **归因结论：** `SUPPORTED-FOR-DESIGN-TEST / VALUE-SOURCE-MECHANISM-NOT-ESTABLISHED / IMPLEMENTATION-PREFLIGHT-BLOCKED / NO-TRAINING`。
- **下一步：** 先做单事件同进程 no-op/hash 合同；若通过，只在已有或新建的 train-side 固定状态上比较 native、正确共支持和 shifted 共支持。当前不启动训练、跟踪推理或全量测试。

## 证据索引

- `m1_early_gate_sequence_scene_attribution_20261026.md`
- `m1_msha_v1_v2_operator_attribution_20261026.md`
- `m1_msha_v1_v2_operator_refinement_20261026.md`（FCCR² 的历史设计）
- `m1_query_conditioned_cross_value_fixed28_20261024.md`
- `m1_early_template_value_reconstruction_fixed46_20261023.md`
- `m1_msha_cross_support_fixed28_20261026.md`（只排除已测的单层 hard 3×3 cross-QK key mask，不等价于本候选的全局 soft support Value reallocation）
- `m1_early_gate_token_reliability_quick3_20261026.md`
- `m1_msha_query_role_fixed_state_preflight_20261022.md`
- `m1_msha_query_role_holdout_quick_20261023.md`
- `m1_msha_uncertainty_gate_50seq_20261021.md`
- `dcvr_manuscript_draft_20261026.tex`（DCVR-v2 当前仍为待验证提案）

本报告只做已有证据综合与算子设计，不修改模型代码、不训练、不推理、不写 checkpoint。GPUHours `0`。

## 2026-10-26 固定状态机制筛查结果

实现合同先经过两次 smoke 修正：首次 smoke 暴露诊断脚本将 FCSV-R 的 `NATIVE` 错误写成 `M1`；随后修正基线引用。第二次 alpha0 smoke 暴露零前景 token 的恒等路径问题，改为 `alpha=0` 显式使用全 1 source gate。修正后，`FCSV_R_CORRECT` 与 `FCSV_R_SHIFTED` 相对 `NATIVE` 的逐框预测坐标和 fixed-state IoU 最大绝对差均为 `0`，alpha0 parity 通过。

在 Early gate full15 checkpoint（SHA256 `e38d18fe...1811`）和 train-side fixed28 同状态下，运行 `NATIVE / FCSV_R_CORRECT / FCSV_R_SHIFTED` 三臂，固定 `alpha=0.1`、temperature `0.07`，不训练。结果目录：
`output/experiments/M1_MSHA_FCSV_R_FIXED28_20261026/`。

| arm | 28-event mean fixed-state IoU | matched-good mean | severe-onset mean |
|---|---:|---:|---:|
| NATIVE | 0.543864 | 0.681016 | 0.406712 |
| FCSV-R correct | 0.548393 | 0.693473 | 0.403313 |
| FCSV-R shifted | 0.548425 | 0.693545 | 0.403305 |

相对 NATIVE，correct 的事件 delta 为 `+0.004530`（14 胜、13 负、1 平），shifted 为 `+0.004562`（14 胜、13 负、1 平）。correct 与 shifted 几乎相同，且 severe-onset 平均下降约 `0.0034`；最大正 delta `whiteshoesleftbottle-gai/matched_good +0.1746` 是单事件，不能作为稳定涨点。`co` 和 gate 统计显示两个方向的前景预算约为 `1.0`，但这只验证 simplex 系数守恒；不能解释为真实消息范数守恒。

**机制判断：** 实现合同已通过，FCSV-R 确实只作用 cross-modal Value 消息并保留 native Q/K/self path；但机制签名未通过。正确共支持没有胜过 shifted 共支持，说明当前 `C=P_RP_T^T` 的分布重排尚未证明依赖真实跨模态候选对应。该结果只支持“可执行、可归因的候选”，不支持进入训练或声称性能提升。按停止规则，暂不增加新的门控、熵或第二交互分支；下一步若继续，只能先做来源 token/背景贡献分解和正确/错位的配对统计，确认是否为共同背景或 mask 稀疏导致两臂退化为近似相同算子。

## 来源贡献分解（后续归因）

利用 fixed28 已保存的 `query_feature_deltas.csv` 与 `query_boundary_deltas.csv` 做逐层前景/背景分解。correct 在 matched-good / severe-onset 两类事件的 feature delta 比值（前景/ hard-background）分别为：层 0 `1.052/1.047`，层 1 `1.034/1.027`，层 2 `0.989/0.988`。也就是说，轻微的早层前景偏向在后续 MSHA 层消失，最终变化并未集中到目标 token。

在 `fusion_return` 边界，correct 的前景/背景 delta 分别为 `6.143/6.053`（matched-good）和 `6.146/6.073`（severe-onset），与 shifted 的 `6.106/6.013` 和 `6.112/6.038` 非常接近。层 2 前景/背景变化比与事件 IoU delta 的 Pearson 相关仅 `r=0.025`（correct）和 `r=0.017`（shifted）。因此 fixed28 的小幅正均值不是“目标来源选择性增强”导致，而更符合近似无方向的特征扰动；不能通过训练量直接补足机制缺口。

**最终归因：** FCSV-R 的 simplex 预算守恒和 cross-Value 隔离实现均成立，但 `C=P_RP_T^T` 在当前 Early gate 表示上没有提供足够的真实对应判别力。正确/错位不可分、前景贡献未在深层保留、定位变化与来源选择无相关，三项共同否定本候选作为后续论文算子的证据链。该路线停止在机制筛查阶段，不进入快速训练或全量训练。
