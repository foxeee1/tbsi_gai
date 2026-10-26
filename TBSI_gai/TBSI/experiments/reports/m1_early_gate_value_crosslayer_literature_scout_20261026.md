# Early gate 后继 Value 细化：跨领域文献与算子候选（2026-10-26）

> **历史方案，非当前执行计划（2026-10-26 清理标记）：** 本文原始数据、引用和推导保留；其中候选优先级、准入条件和下一步命令不再作为当前指令。当前范围见项目 `README_当前研究.md`；本标记不表示所有假设已被证伪，也不批准新训练。清理前原文见 Git `c0f1aff`。


## 结论

**建议保留 Early-conditioned MSHA gate 作为研究 parent，并把唯一主候选收敛为“Early-conditioned Low-rank Cross-layer Value Retention”（ECLVR，早期条件低秩跨层 Value 保留）。** 它借鉴语言模型的 Value Residual Learning：将早层的 Value 信息带到深层；再将 Early gate 的三项证据用于选择每个模板 token、每个 attention head 的低秩 Value 子空间。它仍只作用于原 MSHA 跨模态 Value 来源，不改变 Q/K、attention logits、同模态路径或 tracker 更新状态。

这是一条可证伪的设计候选，不是有效涨点结论。Early gate full15 相对 M1 的 AUC 点估计为 `+0.349 pp`，sequence-bootstrap CI `[-0.727,+1.481]` 跨零，且没有同初始化、同训练轨迹的 gate-off full15；gate-only full244 点估计约 `-0.213 pp`。因此可保留它作为先导机制与实验资产，但不能把“门控已证明有效”当作新算子的因果前提。

## 研究边界

当前 Early gate 对来源模板 token `k`、方向 `m→n` 生成一个标量：

```text
e[m,k] = [RGB-to-search max similarity,
          TIR-to-search max similarity,
          RGB/TIR template agreement]
g[m→n,k] = exp(tanh(MLP(e[m,k])))
V_eff[m→n,k] = g[m→n,k] * V_current[m,k]
```

源码方向约定为 `gate[..., 0]` 控制 TIR Value 进入 RGB queries，`gate[..., 1]` 控制 RGB Value 进入 TIR queries。这个算子对同一个 Value 向量的所有通道等比例缩放，所以不能表达“保留早层中的某些目标特征方向、抑制深层新增的另一些方向”。

下一步只沿 Value 的**深度来源轴和低秩通道子空间轴**细化，不把搜索 token 重排、可靠度统计堆叠或新的门控 head 加进来。问题仍写为：当前 MSHA 跨模态 Value 传递对早期目标证据缺少层间选择性，现有 scalar gate 只能调消息总量，不能选择 Value 内容方向。它是对 M1 crop 内定位失配问题的一个机制假设，不是已证实的 MSHA 首因。

## 近期相关工作与代码核验

| 工作 | 时间与来源 | 实际算子 | 对本路线的用途与边界 |
|---|---|---|---|
| **Value Residual Learning / ResFormer** | arXiv:2410.17897v5，2025-06-08 更新；官方代码 MIT，commit `b7d8873683d68f81caa10f3b004b1904d74ff9e5` | 在 Transformer attention 中保留第一层投影 Value；后续层将早层 Value 与当前层 Value 混合。核验到的 Llama 实现对后层使用固定 `0.5*V_first + 0.5*V_current`。 | 直接提供“Value 可以沿层间保留”的算子依据。本文候选改为 RGB-T MSHA 的跨模态来源 Value，并由 Early gate 条件控制低秩子空间；原论文并未验证该迁移，也不能据此声称提升跟踪性能。源码：`https://github.com/Zcchill/Value-Residual-Learning`。 |
| **FreeTransformSR** | arXiv:2609.05912v3，2026-09-12 更新；作者仓库 commit `0290b0daabb14a7ce11627c1513af92945e33371` | 每个通道使用 identity 加低秩残差构成可学习的可分离空间基，在局部窗口变换特征；另有局部卷积分支与 sample-level 强度 gate。 | 是 2026 年非跟踪领域的细粒度基变换参考。它主要变换局部空间位置，不直接提供跨模态 Value 语义；并且公开仓库未发现 LICENSE 文件。只借鉴“受约束低秩基”的设计原则，不复制代码，不把其空间变换整体移到 MSHA。源码：`https://github.com/HJiLi/FreeTransformSR`。 |

FreeTransformSR 的空间 token 变换与已有 spatial token bridge / correspondence 失败资产有相邻之处，因此不选作主算子。Value Residual Learning 年份早于 2026，但与当前问题的 Value 信息流最直接；FreeTransformSR 用作低秩参数化的近期跨领域依据。文献来源支持算子假设，不代表其原任务结果可外推到 LasHeR。

## 主候选算子

令 `X^e_m` 为 Early gate 已读取的早层模板特征，`X^l_m` 为当前 MSHA 层的模板特征。为避免直接相减两个层间坐标系不同的投影结果，先用**当前层同一个** `W^l_V` 得到可比较的 Value：

```text
V^l_m       = W^l_V(LN(X^l_m))
V^{e→l}_m   = W^l_V(LN(X^e_m))
ΔV_m        = V^{e→l}_m - V^l_m
z[m,k,h]    = U_h^T ΔV[m,k,h]                 # rank r, r << head_dim
a[m,k,h]    = tanh(W_e e[m,k] + W_z z[m,k,h]) # r 个低秩系数
ΔV_lr       = U_h diag(a[m,k,h]) z[m,k,h]
V_eff[m→n,k,h] = g[m→n,k] * V^l[m,k,h] + γ_h * bounded(ΔV_lr)
```

`U_h` 是按 head 共享的低秩 Value 基；`a[m,k,h]` 同时读取已有 Early gate 的三项证据 `e` 和 Value 差异在低秩基上的系数 `z`，在每个来源 token、每个 head 上选择要保留的子空间。这样旧 max 相似度不会单独承担所有决策，且没有另造 margin/entropy/reliability 信号。`g` 完整保留 Early gate v1 的方向和 token 标量作用；新增分支只将早期到当前层的内容差异以零初始化 `γ_h` 写回。第一版不增加新场景标签、注意力重排或辅助损失。`bounded(x)` 固定为按每个 token/head 的 `||x||` 平滑限制在 `0.1*||V^l||` 内，不做幅值 sweep。

这个算子比 scalar gate 多出的表达能力是**改变 Value 向量方向**，而不是为整向量乘一个数。它不等同于已失败的同层 Value reconstruction：其残差源是“同一模态、同一模板 token、早层到当前层的表示差”，不是 RGB/TIR counterpart 差分、query-conditioned token 混合或 search-side writeback。它也不等同于 spatial token bridge：不选择、传输或重排 search token。

### 论文因果链

| 节点 | 论文表述 | 可观察的验证信号 |
|---|---|---|
| 问题 | scalar gate 能调整跨模态 Value 总幅度，但不能选择深层 MSHA 中应保留的早期目标特征子空间。 | 目标相关模板 token 的 `ΔV` 与背景 token 的 `ΔV` 在低秩方向或 gate 条件上存在可重复差异。 |
| 设计 | 以 Value Residual Learning 的层间保留为基础，用 Early gate 证据对早层—当前层 Value 差异做 token/head 级低秩选择。 | `ΔV_lr` 非零且对目标前景与背景来源有结构差异；打乱 Early evidence 后该差异减弱。 |
| 中间机制 | 变化发生在 cross-modal Value 消息内容，不只是范数缩放。 | 记录 Value/message cosine、范数、低秩系数分布，以及 cross-message 经原 attention 后的变化；Q/K 与 attention matrix 在被干预层保持相同。 |
| 定位传导 | 有效 Value 内容变化应改善同一 crop/state 的目标竞争或回归框。 | GT-vs-hard-background rank/margin、CENTER top-1、size/offset、固定状态 IoU 成对变化。 |
| 闭环结果 | 若定位改善能延续到状态更新，Early gate 的后继算子才构成有效贡献。 | 在更大预先固定序列集上比较逐序列结果、尾部退化、AUC/PR/NPR/OP50/OP75 与 FPS。 |

任何一环不成立，就收窄或停止贡献主张；不能因为新参数可训练或消息发生变化就写成机制通过。

## 已有失败资产与候选差别

| 已有方向 | 已记录结论 | 本候选如何避开重复 |
|---|---|---|
| margin、token reliability、低置信抑制 | quick3 未晋级，且 control 重复差 `11.15 pp`；不支持靠相似质量信号堆叠。 | 不增加 reliability 特征；沿 Value 的跨层内容来源和低秩子空间变化。 |
| FCSV / 共支持重分配 | 修正后 correct 与 shifted 连续版本没有定位差异；早先变化主要来自背景硬删除。 | 不重分配来源 token 概率，不做 mask 或 support 选择。 |
| Receiver-centered Value residual | fixed28 correct 没有胜过 shifted 和 row-shuffled。 | 不使用 receiver/search 到 template 的对应矩阵；操作对象是同一来源 token 的早层与当前层 Value。 |
| Learnable template Value reconstruction | quick3 AUC `-4.387 pp`，停止当前 source/counterpart 重构实现。 | 不复制该重构公式或直接放大重构支路；若新候选无法展示与其不同的早层保留读数，则不应进入训练。 |
| Spatial token bridge / amplitude sweep | target correspondence 没有稳定改善 rank/IoU；放大后 target 也未胜过 shifted。 | 不动 search token，不添加局部 correspondence 或扩大 residual 强度。 |
| STSER shared/private/null 和 content branch | 有效写回极小，性能未支持；是另一条 search-side T2S 线路。 | 不添加专家路由、不改变 search-side state。 |
| Q/K/logit alignment、模板更新、频域路线 | 与本次 Value-only 边界不同；频域线依顶层规则暂停。 | 本候选固定原生 Q/K、attention、模板更新合同；不使用频域算子。 |

与已审计相关工作的区分边界：TUMFNet/UATrack 侧重模板—搜索证据与模态/不确定性加权；QSTNet 侧重搜索 token 质量；CGATrack 与双向来源/接收可靠性交互的故事最接近。ECLVR 不声称首次做 RGB-T 质量感知或双向调制，而聚焦同一来源模板 token 的跨层 Value 内容保留。**CGATrack 等相近 tracking 论文的原始源码尚未逐算子核验是否有等价跨层 Value 残差，因此新颖性仍未确认，投稿前必须补做该项核对。**

## 最快可判别的实验顺序

1. **现有证据核对。** 固定早层与当前层的同一模板 token，核验同一 `W_V^l` 投影后的 `ΔV` 及其低秩能量，在首帧目标前景和背景 token 间是否可区分；同时确认 Early gate 的三个条件与这些变化是否相关。使用独立 train-side 状态，不按测试涨跌挑事件。若 `ΔV` 没有可重复结构，停止 ECLVR，不换 rank 或放大幅度。
2. **实现合同。** 新 feature flag 默认关闭；关闭时保留受保护 M1 和 Early gate parent 的原执行路径。验证 zero-init parity、同层 `W_V` 坐标、方向索引、shape/save-load、双向梯度、冻结状态和早层特征是否能在 train/test 同时取得。此阶段只做合同，不把随机初始化输出当性能读数。
3. **机制对照。** 同一个 gate checkpoint、同一 frame/crop/state 比较 gate-only、ECLVR、ECLVR 的 Early evidence token shuffle；固定 rank、幅度上限和层位。记录 Value 内容方向与 localization readouts。候选必须胜过打乱条件，且不能靠更大消息范数解释全部差异。
4. **训练准入。** 当前 quick3 control 两次相同配置差 `11.15 pp`，且增强后 tensor 差异未定位；在该可重复性合同解决前，不用 quick3 正负决定 ECLVR 去留。机制信号通过后，做从同一 SOT 预训练权重初始化的 matched gate-only/control 与 gate+ECLVR 两臂，fresh optimizer、同样本顺序、同训练进度；不能把现有 Early gate full15 的单臂点估计当作匹配 control。
5. **论文晋级。** 完整 LasHeR 配对结果只在相同训练、数据和测试合同下报告；逐序列归因需验证定位改善是否沿 `Value change → target/background competition → box → closed-loop` 传导。RGBT234 当前尚无数据，不提前写结果。

## 独立判断

当前最值得继续的不是再扩大 Early gate 的证据输入，而是利用它已给出的“早期来源可信度”来决定**保留哪些早层 Value 特征方向**。这条路线同时保留已有 Early gate 资产，且较同层重构、token routing 和普通 channel gate 有清楚的算子区别。其主要风险是层间特征语义未必可直接相减；因此使用当前层 `W_V` 投影两层特征，并把 `ΔV` 的目标/背景可分性设为第一道证伪条件。

**结论状态：** `SCOUT-COMPLETE / SINGLE-OPERATOR-CANDIDATE / NO-IMPLEMENTATION / NO-TRAINING`。支持进入最小机制分析，不支持宣称有效涨点或直接启动 full training。若早层差异不能区分目标与背景，最合理结论是保留 Early gate 为论文主候选并收窄 claim，而不是叠加新的空间支持、专家或可靠度门控。

## 来源与可追溯性

- Zhou et al., *Value Residual Learning*, arXiv:2410.17897v5（2025-06-08）：<https://arxiv.org/abs/2410.17897>；官方实现：<https://github.com/Zcchill/Value-Residual-Learning>；commit `b7d8873683d68f81caa10f3b004b1904d74ff9e5`；MIT。
- Li and Li, *FreeTransformSR: Efficient Lightweight Image Super-Resolution via Free Low-Rank Learnable Transform*, arXiv:2609.05912v3（2026-09-12）：<https://arxiv.org/abs/2609.05912>；作者实现：<https://github.com/HJiLi/FreeTransformSR>；commit `0290b0daabb14a7ce11627c1513af92945e33371`；仓库未发现 LICENSE 文件。
- 本次检查了仓库的 Early gate、Value reconstruction、receiver-centered Value、spatial token bridge、STSER 归因记录；仅克隆两个公开参考仓库到 `/tmp` 只读检查，没有复制源码到项目。
- 项目 git HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`；worktree 原本有其他修改，本次没有编辑模型代码、配置、checkpoint、测试输出或受保护 M1 资产。
- 无模型训练、推理或 benchmark；AUC/PR/NPR/OP50/OP75/FPS 均为 `NA`，GPUHours `0`。
