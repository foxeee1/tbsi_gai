# M1 需求感知的显式信号引导后融合校正建议（2026-10-12）

## 1. 决策结论

当前不应继续把 S1/S2/S3 直接映射为 RGB/TIR 的固定 `+-15%` 缩放。低 AUC 62 序列上的闭环均值增益主要可由固定 TIR 增强解释；在按序列留出的 24 条 train-side 序列、44 个同 crop/同 state 固定状态上，no-op 的 decoded IoU 最高，S1/S2/S3 及其联合均为负差。现有标签只回答“两种非零动作哪一个相对更好”，没有回答“任一动作是否优于不干预”。

推荐下一锚点不是新的融合模块，而是先验证一个三动作、带拒绝选项的效用信号：`no-op / RGB 方向 / TIR 方向`。只有该信号在独立序列上能稳定识别 no-op，并且选择动作的真实框效用超过固定动作，才进入后融合残差算子。

## 2. 文献中可迁移的机制

| 工作 | 已核验机制 | 对 M1 可借鉴部分 | 不能直接迁移的部分 |
|---|---|---|---|
| Li et al., TBSI/IGF, TPAMI 2025, DOI `10.1109/TPAMI.2024.3475472` | 当前 RGB search 图像经冻结 GIE 得到全局照度，再以通道和空间注意力调节 search fusion | 光照是一个物理可解释、部署时可观测的模态质量原因；适合作为 gate 输入 | 光照不等于动作效用；IGF 没有 no-op 决策，也不显式建模目标与困难负样本竞争 |
| Zhu et al., FANet, T-IV 2021, DOI `10.1109/TIV.2020.2980735`, arXiv `1811.09855` | 拼接 RGB/T 特征、全局池化、FC，按通道 softmax 得到两模态权重并加权求和 | 输入自适应的模态可靠性权重；说明质量估计可由联合特征学习 | softmax 强制两路重新分配，总会干预；没有保留已训练 M1 融合输出，也没有显式 abstention |
| Wang et al., MFGNet, TMM 2023, DOI `10.1109/TMM.2022.3174341`, arXiv `2107.10433` | 由当前双模态输入生成动态模态卷积核；每路动态特征以 residual 加回原特征 | 输入条件算子与 identity residual 是合适的组合，可避免完全替换原表征 | 动态核容量较大；仍没有估计“改动是否优于 no-op”，且其 MDNet/RoI 分类合同不同于 M1/CENTER |
| Chen et al., SelectFusion, TNNLS, arXiv `1912.13077` | 用 deterministic soft gate 或 stochastic hard gate 选择可靠的传感器潜特征，并在退化、缺失和错位传感器上验证 | 显式选择/拒绝不可靠信息，以及用合成退化核验 gate 是否响应正确原因 | 状态估计任务，不提供跟踪框效用或 M1 后融合接口的直接证据 |
| Han et al., TMC, TPAMI 2023, DOI `10.1109/TPAMI.2022.3171983`, arXiv `2204.11423` | Dirichlet evidence 表示每视图不确定性，再做动态 evidential fusion | 不确定性应参与是否介入，而不只是决定 RGB/TIR 方向 | 分类层的证据融合不能原样用于密集 CENTER 响应；Dempster-Shafer 组合并非本项目必要组件 |

上述文献支持的是四个设计原则：可解释条件、样本自适应可靠性、identity residual、基于不确定性的拒绝。没有一篇被核验工作同时提供“M1 式后融合校正 + 显式 no-op + 跟踪框动作效用”。下面的组合是针对当前证据提出的新假设，不应表述为既有论文结论。

## 3. 核心公式

保留 M1 原融合：

\[
F_0 = C_{M1}([R,T]).
\]

从部署时可观测量构造信号：

\[
q_t=[\ell_t,\;m_t^R,\;m_t^T,\;\delta_t^{RT},\;h_t^0,\;g_t].
\]

其中 `ell` 是 IGF 式照度/可见质量，`m^R,m^T` 是两路目标相对竞争峰的响应 margin，`delta` 是跨模态响应分歧，`h0` 是 M1 初始 CENTER 响应的熵、top1-top2 margin 等风险量，`g` 是 crop/运动/尺度等状态量。照度说明“为什么某模态可能退化”，分歧说明“两路是否冲突”，CENTER 风险说明“当前输出是否可能需要修正”；三者不能互相替代。

对动作集合 `A={0,R,T}` 预测相对 no-op 的效用及不确定性：

\[
(\hat u_a,\hat\sigma_a)=G_a(q_t),\qquad \hat u_0\equiv 0,
\]

\[
a^*=\arg\max_{a\in\{R,T\}}\hat u_a,
\qquad
LCB_t=\hat u_{a^*}-z\hat\sigma_{a^*}-\tau.
\]

以收益下置信界形成严格 dead zone：

\[
\gamma_t=\gamma_{max}\,\operatorname{clip}
\left(\frac{LCB_t}{s_u},0,1\right).
\]

候选后融合校正为：

\[
F'_t=F_{0,t}+\gamma_t\,\alpha\,
\tanh\!\left(\mathcal R_{a^*}(R_t,T_t,q_t)\right),
\qquad \alpha(0)=0.
\]

`alpha=0` 保证初始时与 M1 数值一致；`tanh`、`gamma_max` 和 residual norm clipping 限制扰动。部署时若 `LCB<=0`，严格得到 `F'=F0`。这比 softmax 两路重加权更符合“好序列不动”的要求。

真实监督标签必须相对于 no-op 定义：

\[
u_a^*(x)=IoU\{D[H(F_0+\rho\Delta F_a)] ,b^{gt}\}
-IoU\{D[H(F_0)],b^{gt}\},
\]

其中 `H` 是同一个 CENTER head，`D` 包含原 Hann、peak、offset 和 size 解码。应包含多个小剂量 `rho`，而不是仅用 `+-15%`。旧标签

\[
IoU(R^+,T^-)-IoU(R^-,T^+)
\]

只比较两个非零动作，无法保护 no-op，必须停止使用。

## 4. 信号角色与优先顺序

1. **Need 信号（最高优先）**：以 `max(u_R,u_T)>0` 为目标，识别是否存在可实现的正效用。CENTER 低 margin/高熵只能作为输入，不能直接当标签，因为 tracker 可能高置信地跟错目标。
2. **Direction 信号**：只在 need 为真时判断 RGB 或 TIR 方向。IGF 照度适合解释 RGB 退化；热交叉、TIR 低对比和两路 target-vs-distractor margin 才能覆盖反方向。
3. **Strength 信号**：回归最佳小剂量或收益曲线，在 gate 和方向跨序列通过后再做。当前数据不支持一开始联合学习复杂空间 gate。

S1/S2/S3 的冻结预测器均未获准复用。其原始分量可以重新作为 `q` 的候选输入，但必须在新标签和新 sequence-group split 下重新筛选。联合信号不是求均值或投票；应分别承担 need、direction、risk/uncertainty，避免三个相关误差相互放大。

## 5. 最小判别实验

第一阶段只筛信号，不训练新融合器。固定 M1、同 crop/state/head，建立动作 `no-op / RGB / TIR` 与 `rho={0.05,0.10,0.15}` 的真实 decoded-box 效用表。数据必须覆盖正常、低 AUC、LI、TC、SA、小目标和高运动，但按序列划分 train/calibration/test；低 AUC 只能做分层，不能作为唯一训练分布。

主指标不是平均 IoU，而是：

- no-op specificity：真实 no-op 最优状态中保持不动的比例；
- harmful intervention rate：相对 no-op 下降超过预注册 epsilon 的比例；
- selected-action regret：oracle 动作效用减去所选动作效用；
- risk-coverage 曲线：只干预最高置信的 5%/10%/20% 状态时的净效用；
- 相对固定 RGB、固定 TIR、随机方向和 confidence-only gate 的配对差；
- sequence-macro bootstrap 区间，以及正常/困难场景分别报告。

准入条件应优先约束伤害而非均值：独立 test split 上，正常/真实 no-op 状态的 harmful intervention rate 不高于预注册上限；最高置信 coverage 内净 decoded IoU 为正且超过固定 TIR；校准误差和分层方向不发生 train/held-out 反转。未通过则停止，不实现 residual operator。

第二阶段才固定已通过的 gate，训练一个零初始化后融合 residual；与 `gamma=0`、恒定小 gamma、随机 gamma、illumination-only gate 和 utility gate 做单变量比较。最后才做从同一固定状态开始的短 rollout，检查单步收益是否转成闭环收益。

## 6. 主要局限

- **信号和动作不可完全分离**：效用标签依赖具体 `Delta F_a`；一个信号对固定缩放无效，不代表对所有残差无效，反之亦然。第一阶段只能筛“对定义动作有用的信号”。
- **不确定性不等于正确性**：单模型预测方差可能仍过度自信。至少需要 sequence-held-out calibration；必要时用小型 ensemble，但会增加计算。
- **初始 CENTER 风险有循环性**：用 `F0` 的响应决定是否修正需要一次预读出，修正后再读出，可能增加 head 延迟；应报告实际 FPS。
- **全局 gate 上限有限**：一帧内可能同时存在局部 RGB 过曝与 TIR 热交叉。第一版先做全局 gate 以保证归因；只有出现空间异质性的固定状态证据后再升级逐位置 gate。
- **序列好坏不是静态属性**：正常序列也会出现局部困难帧，困难序列也有无需干预的帧。应做逐帧 gate，并可用双阈值/EMA 防抖，不能先验地把整条序列永久开关。
- **固定状态不等于闭环**：单步正效用可能因状态反馈消失或反转。信号准入后仍需短 rollout 和完整 matched M1 闭环对照。
- **选择偏差和阈值泄漏**：62 个低 AUC 序列、20 个正常序列及现有 44 held-out states 都已被反复观察，不应继续用于选特征、阈值或模块容量。需要新的 train-side 独立序列。
- **M1 首因仍未确定**：现有证据同时指向 CENTER 选格与 offset/size 读出。即使 gate 有效，也只能证明某些状态可被融合校正，不能证明融合是全局唯一瓶颈。

## 7. 建议选择

推荐选择“**先做三动作 utility/no-op 信号筛选，再做 need-aware residual**”。IGF 光照作为 `q` 的一个输入和 illumination-only 消融；FANet/SelectFusion 提供可靠性 gating 参考；MFGNet 只提供 input-conditioned residual 参考；TMC 只提供 uncertainty/abstention 参考。暂不做逐位置复杂 gate、动态卷积核或多信号投票。

备选的最小路线是只做 IGF 式 illumination gate。它归因最干净、成本最低，但预期只覆盖 LI 及相关暗光状态，无法解释 TC、SA、困难负峰和高置信误跟。若研究目标是先获得一个可解释的窄结论，它可以作为第一个消融；若目标是保护好序列并覆盖多类困难，三动作 utility/no-op 是更合适的主线。

## 8. 来源与证据边界

- 本地 IGF 全文核验：`experiments/module_study/dbf_m1_20261009/01_paper_A_evidence.md`。
- M1 固定状态结果：`experiments/reports/m1_fixed_state_action_replay_20261012.md`。
- M1 场景与机制优先级：`experiments/reports/m1_full244_fusion_mechanism_priority_20261010.md`。
- FANet primary preprint: <https://arxiv.org/abs/1811.09855>。
- MFGNet primary preprint: <https://arxiv.org/abs/2107.10433>。
- SelectFusion primary preprint: <https://arxiv.org/abs/1912.13077>。
- TMC primary preprint: <https://arxiv.org/abs/2204.11423>。

本报告是文献与既有实验的设计综合；没有运行训练、推理或新诊断，没有修改受保护 M1 资产，也没有产生性能结论。
