# M1 MSHA v1/v2 算子归因与设计准入（2026-10-26）

## 决策

**归因结论：`OPERATOR-DESIGN-READY / IMPLEMENTATION-PREFLIGHT-BLOCKED / NO-TRAINING`。**

已有证据足以提出一个不同于 v1/v2 的 MSHA 输入建模算子；不支持继续增加 Early gate / DCVR 的输入量、幅度或 token 权重。训练筛查暂不准入：quick3 control repeat 的 AUC 相差 11.15 pp，post-augmentation tensor 未核验；当前 fixed-state replay 与历史记录的 parity 根因也未定位。两项合同应在实现前置检查中解决，而不是靠新算子训练来猜原因。

## v1/v2 和原生 MSHA 的边界

原生模板 MSHA 将 RGB/TIR 模板 token 拼接，经过共享 LayerNorm 与 QKV，形成一个联合注意力矩阵，再分成 RGB←RGB、RGB←TIR、TIR←RGB、TIR←TIR 四路消息。两模态怎样成为查询/键、哪些 token 互相匹配，主要由 Q/K 决定。

Early gate v1 在 block 2 的早期特征上，对每个模板 token 计算 RGB、TIR 各自对整个 search 的最大余弦相似度和跨模态一致性，经 `3→32→2` MLP 生成两个方向的缩放量。它作用于 TBSI-E 的跨模态 Value 消息，不改变 Q/K、注意力关系、同模态消息或 tracker 状态。full15 相对登记 M1 的 AUC 点估计 `+0.349 pp`，但 bootstrap CI `[-0.727,+1.481]` 跨零，且没有严格匹配的 gate-off full15；因此它是有方向性的先导组，不是已完成因果归因的性能 parent。

DCVR-v2 草案把 Value 调节拆成方向预算与前景内 token 重分配，并加入匹配分布证据。它仍然回答“已经形成的跨模态消息分多少”，没有改变 Q/K 的关系形成或 MSHA 输入表示。故 v2 可作为消息分配对照与论文备选，不满足用户当前要求的主要创新边界。

## 证据归因

| 证据 | 可以支持的判断 | 不能支持的判断 |
|---|---|---|
| Early gate full15 有小幅正点估计；无 matched gate-off full15；同 checkpoint fixed-state 有效状态中 CENTER top-1 `0/28` 改变、offset map 更敏感 | Value 路径可扰动回归和后续轨迹；v1 值得保留为先导 parent | 增益由 gate 单独造成，或 gate 修复了 CENTER 排序 |
| v1 序列分析中方向门控均值几乎不随序列变化（约 RGB→TIR `1.0004–1.0006`、TIR→RGB `0.9894–0.9900`），序列结果却方向相反 | 当前汇聚式证据没有表达足够的空间/目标条件差异 | 再调全局方向幅度可解释序列异质性 |
| 局部 margin、token reliability 两种 quick3 未晋级；两次相同 control AUC 相差 `11.15 pp` | 继续添加相近置信度/可靠度信号的性价比很低；短训排序目前受运行波动限制 | 这些短训负结果证明所有空间或输入建模都无效 |
| Query-conditioned cross-Value fixed28 有特征变化但 CENTER rank `0/28` 改变；quick3 AUC `-0.135 pp` | 只改变 Value 内容并未证明可改变 CENTER 候选竞争 | 所有 query-conditioned MSHA 关系建模都无效 |
| Early template Value reconstruction 的同 checkpoint strength 0/1 对照，CENTER rank `0/46` 改变；本轮 cross-support 同位置窗口对错位窗口 rank `0/27` 胜 | 固定 Value 重构与硬局部支持均未给出定位效用信号 | 需要更大的 Value MLP，或固定 3×3 支持是正确方向 |
| 搜索侧 local correspondence quick3 AUC `-1.113 pp`，CI 跨零、FPS 下降约 `12.1%` | 停止当前搜索侧局部交换实现 | 模板 MSHA 的 QKV 前输入关系从未被验证，因此整类无效 |

共因判断：失败候选多在“Value 内容/幅度/支持集受控修改，目标候选排序不动或方向不稳定”处汇合。更合理的瓶颈假设是：MSHA 在进入共享 QKV 前没有显式保留目标相关的跨模态对应与模态私有差异；后续只改 Value 分配，未改变关系核本身。此为当前最佳设计假设，尚非已证实首因。M1 的系统级失败链仍是 crop 内定位失配、状态反馈漂移、目标出界和恢复不足，不能把首因单独归给 MSHA。

## 进入算子设计的单一候选

候选名：**前景条件软对应输入重构（Foreground-conditioned Soft Correspondence Input Reconstruction, FSCIR）**。它借用跨模态 token transport / soft assignment 的思想，但将算子放在模板 MSHA 的 QKV 之前；Early gate v1 仅作为 parent 对照，不与新算子叠加调参。该想法与固定 3×3 mask 不同：保留全局软对应；与 DCVR-v2 不同：重构实际输入 token，使后续 Q/K/V 都读取新表示。

对 MSHA 第 `l` 个 TBSI-E block 的输入 token `X_R, X_T ∈ R^{B×64×D}`，使用同一 block 入口处的 early token `E_R,E_T` 建立软运输矩阵：

```text
P_R<-T = softmax_j( normalize(E_R) normalize(E_T)^T / tau )
P_T<-R = softmax_j( normalize(E_T) normalize(E_R)^T / tau )
C_R = P_R<-T (w_T * X_T) / (P_R<-T w_T + eps)
C_T = P_T<-R (w_R * X_R) / (P_T<-R w_R + eps)
X'_R = X_R + alpha * W_R C_R
X'_T = X_T + alpha * W_T C_T
```

`w_R,w_T` 是首帧模板框映射到 8×8 网格的软前景权重；只约束消息来源，避免背景 token 被当作等价模板证据。`W_R,W_T` 先用低秩线性投影，最终投影零初始化，残差范数上限固定为来源 token 范数的 `0.10`；这样 `alpha=0` 或零初始化都能精确复现 parent，且第一步梯度先更新最终投影。将 `[X'_R,X'_T]` 送入原有共享 LayerNorm/QKV 和全局联合注意力，其他路径保持不变。第一版只在最早有证据可读出的 TBSI-E block 3 插入；不加时间记忆、额外损失、CENTER 改写或模板更新。

### 机制链

| 环节 | 明确陈述 |
|---|---|
| 问题 | Early gate 将 16×16 search 响应压成每模板 token 一个 max 标量；后续 Value-only 修改不会重建 Q/K 的 token 对应关系，目标/背景来源也未充分分开。 |
| 设计 | 用前景来源权重对 RGB/TIR 模板 token 做全局软对应传输，再将互补模态上下文以零初始化残差写回 MSHA 输入。 |
| 预期中间效应 | 正确前景对应相对等面积错位前景控制，应改变 cross-modal QK logits/attention 及四路消息；背景对应不应产生同等 target-token separation。 |
| 定位效应 | 固定同 crop/state 时，GT CENTER rank、GT-hard-background margin 或目标邻域响应应有跨序列同向改善，且回归框 IoU 不以明显退化换取 rank。 |
| 闭环效应 | 后续更少出现 crop 内错误持续、目标出界或恢复时间变长；只有匹配训练和 full-244 才能验证性能贡献。 |

预期贡献不是“又一个 gate”，而是把目标条件从 Value 的后验消息预算前移到 token 表征与 Q/K 关系形成之前。与最新 RGB-T tracking 中的 token filtering/bridging 区分时，论文必须聚焦 M1 的 crop 内失配链，并以“正确前景 transport 对比等面积错位 transport”的因果检验区分贡献；目前不能声称新颖性或效果已成立，需检查 BTMTrack 等公开实现的具体重合算子。

## 风险与证伪

- 前景权重只来自首帧框；遮挡、形变或模板背景混入会让 transport 聚合错误来源。首轮保持权重定义固定，报告前景覆盖率与 mask hash，训练和测试必须完全相同。
- RGB/TIR 的物理像素对齐不等于深层 token 对齐。全局 soft assignment 允许语义错位，但也可能把相似背景互相匹配；必须设置等质量 shifted-mask 对照，并测 attention entropy 与有效 token 数。
- 在全部 64×64 token 上做两向相似度会增加算量；首版只做模板 token 运输，估算/实测额外 FLOPs、峰值显存和 FPS，不能把稀疏化作为首轮共同变量。
- 零初始化上投影可能使首步只有上投影获得梯度；合同检查应记录两层参数至少数步后的梯度和更新量，不把首步 `down` 梯度为零误判为断图。
- 若输入重构改变融合特征，却不改变 Q/K/注意力关系或目标/背景分离，假设未通过；若关系改变却不改善 fixed-state CENTER/框读出，停止该位置继续加复杂度；若正确 mask 与 shifted mask 无稳定差异，停止目标条件 transport 设计。

## 准入闸门与最短实验链

1. **实现合同（可开始）：** 新 feature gate 默认关闭；使用 Early gate full15 checkpoint 作算子 parent 时保存独立代码快照与新 run identity；disabled 与 enabled `alpha=0` 逐元素 parity；检查形状、保存加载、有限值、源 mask 对齐、双向梯度和冻结 BN。该步骤不需要训练。
2. **固定状态机制（实现后必须）：** 先在 1 个 smoke 事件验证 instrumentation，再在预注册 train-side calibration cohort 上按同 crop/state 比较 `parent`、`FSCIR`、`FSCIR+shifted source mask`。主要变量是正确/错位来源对应；固定层、秩、温度与幅度。记录 QK logits、四个 attention 块、四路消息、目标/背景 token 距离、CENTER rank/margin、size/offset、框 IoU。旧 M1 replay parity 未解决时，以同一当前源码的 paired parent/candidate 差值为分析对象，不把它冒充受保护 M1 直接复现。
3. **训练前提：** 在启动 quick3 前先解决适用的 execution contract：单 crop no-op replay 同进程重复确定，并记录模板输入至 CENTER map 的分层最大差；另对 quick3 control 的增强后 template/search tensor 做 hash 重复审计。control variance 未被归因前，不用 7 序列 quick7 的单次涨跌做算子筛选。训练候选与 control 必须同初始化、同数据/增强轨迹、同 optimizer steps、同软件栈，仅切换 FSCIR。
4. **快速训练准入判据：** 固定状态正确-mask 相对 shifted-mask 至少在一个预注册主读数（CENTER GT rank/margin 或 decoded IoU）产生跨多个序列同向的 target-specific 差异；无明显 stable-control 伤害；机制差异超过同运行重复噪声；mask train/test 定义逐项一致；零残差 parent parity 通过。通过后才跑匹配 quick3，再用更大固定 cohort 复核；不通过则停止该算子，不扩大 rank、层数或 alpha 搜索。

## 当前判定与资产边界

- **进入算子设计条件：通过。** 研究对象、信息流缺口、一个单变量候选、预期信号与明确反证条件已定义；候选位于模板 MSHA QKV 输入边界，覆盖用户要求的输入建模/交互改造。
- **进入代码实现：可以开始合同实现；不需先写训练脚本。** 所有新增参数与逻辑必须默认关闭，保留保护 M1 受限路径。
- **进入快速训练测试：暂不通过。** M1 fixed28 no-op replay 的来源差异仍未定位；quick3 control repeat 不稳定且 post-augmentation tensor 未 hash。局部窗口负结果和 Value-only 失败不足以替代这两项可重复性前置。
- 不触碰受保护 M1 YAML、checkpoint、结果与日志；不改 DCVR-v2 文档的已选结论，不删除已有资产。本报告只做已有证据综合与理论设计，没有模型前向、推理、训练或权重写入。

## 证据索引

- `m1_early_conditioned_gate_operator_sequence_analysis_20261023.md`
- `m1_early_gate_optimization_analysis_20261026.md`
- `m1_early_gate_control_repeatability_20261026.md`
- `m1_query_conditioned_cross_value_fixed28_20261024.md`
- `m1_query_conditioned_cross_value_quick3_20261024.md`
- `m1_early_template_value_reconstruction_fixed46_20261023.md`
- `m1_msha_cross_support_fixed28_20261026.md`
- `m1_local_cross_modal_correspondence_quick3_20261024.md`
- `m1_msha_feature_operator_literature_scout_20261023.md`
- `dcvr_manuscript_draft_20261026.tex`（DCVR-v2 是提案，不是验证结果）

## 2026-10-26 算子设计状态更新

逐序列与公式复核后，原 FSCIR 只作为历史提案保留；随后提出的 `FCCR²` 解决了软支持稀释问题，但把残差写入 QKV 前输入，不符合当前论文 Value-only 贡献边界。因此 FCCR² 也保留为历史备选，当前主候选改为 FCSV-R，详细定义见 `m1_dcvr_v1_v2_value_source_reallocation_20261026.md`。

当前 FCSV-R 仍是设计假设，机制未建立。固定状态机制筛查前必须先处理 single-event no-op replay parity；通过后再比较 native、FCSV-R 和错位共同支持反事实，并记录真实 Value 消息变化。当前不代表算子实现已准入或机制有效。
