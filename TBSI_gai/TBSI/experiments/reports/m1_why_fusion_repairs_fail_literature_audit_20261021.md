# 为什么 M1 的内部和旁路修正持续无效：RGB-T 跟踪文献对照审计

日期：2026-10-21  
范围：M1 `TBSI_E_MHSA_FULL_4090D_seed42`，epoch15 checkpoint，RGB/TIR 单目标跟踪、模板融合与模板更新。

## 结论先行

M1 目前的主要瓶颈不是“融合模块表达能力不够”，而是**没有一个被训练目标直接约束的、空间位置敏感的目标身份决策链**。现有模块大多只能改变消息幅度或局部特征，却没有同时解决以下四件事：

1. 在 search token 上区分目标和 hard background；
2. 把跨模态互补性与模态一致性分开；
3. 防止错误峰进入状态与模板更新；
4. 让最终 CENTER peak、size、offset 与融合特征共享同一个目标选择目标。

因此，内部改动和旁路改动都会出现同一种现象：输入特征有变化，但变化不是集中到目标位置；或者目标证据存在，CENTER 仍选错峰；一次错峰随后又通过 crop/state/template feedback 被放大。

## M1 已有证据如何指向共同瓶颈

- PVL 放在模板 loc6：AUC `48.15`；放在 search Neck：AUC `44.07`，均低于 matched control `52.33`，说明通用旁路增强不能自动产生目标选择性。
- 50 序列 message gate：GT-hard-background margin 平均变化 `-0.000266`，GT rank 改善 `3/26`；loc9 query foreground-background 差 `-0.06303`，participation 差 `-0.05155`，不确定性输入反而偏向背景。
- token-level 审计：模板匹配 margin 在 loc3/6/9 为正但逐层衰减，和 CENTER margin 的相关性只有 `-0.257/-0.106/-0.088`；RGB/TIR agreement margin 为 `-0.0769/-0.0678/-0.0876`，即目标区一致性低于背景区。
- 早期信息流案例显示没有统一的“MHSA 首因”：有的序列在 block02 已丢失 TIR/RGB 目标证据，有的序列后层重新获得目标证据但 CENTER 仍偏移。
- 固定状态下 query-conditioned 的收益很小，闭环中出现大幅涨跌，说明大部分灾难性变化来自状态/crop 反馈，而不是单帧融合算子本身。

这些结果排除了“再加一个普通 attention/adapter 就能解决”的假设，但没有排除目标条件监督、干扰物建模和保守更新这条主线。

## 至少 10 篇同领域论文的解法

以下论文均直接涉及 RGB-T/可见光-红外跟踪或其多模态融合决策。机制描述依据论文摘要、正式出版元数据和论文题目；没有把论文结果直接当作 M1 的性能保证。

| 论文 | 年份 / DOI | 论文实际解决的环节 | 对 M1 的启示 |
|---|---|---|---|
| SiamFT: An RGB-Infrared Fusion Tracking Method via Fully Convolutional Siamese Networks | 2019, [10.1109/access.2019.2936914](https://doi.org/10.1109/access.2019.2936914) | 双流分别提取 RGB/TIR，再分别融合 template/search 特征；强调互补信息进入匹配链，而不是只做全局拼接 | M1 的模态融合应保留 modality-specific evidence，并在 search matching 上验证目标性；不能把 agreement 当可靠性 |
| DSiamMFT: Dynamic Siamese Networks Using Multi-layer Feature Fusion | 2020, [10.1016/j.image.2019.115756](https://doi.org/10.1016/j.image.2019.115756) | 动态 Siamese、多层特征融合，利用不同深度的目标信息 | M1 的单一 16×16 读出容易把小目标和背景混在一起；需要明确的层级目标监督，而不是只改 loc6/Neck |
| SiamCDA: Complementarity- and Distractor-Aware RGB-T Tracking | 2021/2022, [10.1109/tcsvt.2021.3072207](https://doi.org/10.1109/tcsvt.2021.3072207) | FPN 获取跨层信息；CA-MF 建模跨模态互补；额外显式处理 distractor-aware 信息 | 这是与 M1 最直接的反例：别人把“目标/背景区分”作为独立问题建模，而不是只估计模态一致性。M1 缺少 hard-distractor 分支或损失 |
| RGB-T Tracking by Modality Difference Reduction and Feature Re-Selection | 2022, [10.2139/ssrn.4137009](https://doi.org/10.2139/ssrn.4137009) | 先减少两模态的表示差异，再重新选择有用特征 | M1 的 agreement 不是 feature re-selection；需要学习“哪些维度/位置可用于目标匹配”，不能直接把高一致性当高可靠性 |
| SiamCDA 的后续跨模态互补路线（Complementarity-aware fusion） | 2022, [10.1016/j.patcog.2022.108881](https://doi.org/10.1016/j.patcog.2022.108881) | 以互补性而非单纯相似性作为 RGB-T 融合依据 | M1 的 uncertainty estimator 应至少分解 shared/common 与 modality-specific/complementary 两类证据 |
| Robust RGB-T Tracking via Consistency Regulated Scene Perception | 2023, [10.1109/icip49359.2023.10222904](https://doi.org/10.1109/icip49359.2023.10222904) | 多任务 nested global reasoning，建模目标与周围语义区域的关系；用 meta-unsupervised strategy 更新 scene perception | 该方法直接补上 M1 缺少的 scene/distractor context；只看 target-template similarity 无法知道 hard background 是否更像目标 |
| Visible and Infrared Object Tracking via Convolution-Transformer Network With Joint Multimodal Feature Learning | 2023, [10.1109/lgrs.2023.3259583](https://doi.org/10.1109/lgrs.2023.3259583) | CNN/Transformer 双路径同时做表示学习与融合；区分 local modality-independent 和 global modality-shared 特征；显式利用 inter-modality correlation | M1 的 MHSA 主要解决消息交互，却没有把 local modality-independent 与 global shared evidence 分开约束 |
| Anchor-free Siamese Network Tracker With Transformer for RGB-T Tracking | 2023, [10.1038/s41598-023-39978-7](https://doi.org/10.1038/s41598-023-39978-7) | anchor-free 定位，并加入 channel/spatial attention 抑制背景干扰 | 这不是建议 M1 换成 anchor-free head，而是说明背景抑制必须进入定位决策；M1 当前 CENTER ranking/size/offset 仍可能独立失配 |
| Siamese Infrared and Visible Light Fusion Network for RGB-T Tracking | 2023, [10.1007/s13042-023-01833-6](https://doi.org/10.1007/s13042-023-01833-6) | 以双模态 Siamese 融合为主线，保持跨模态互补特征进入匹配 | 对 M1 的启示是模板和 search 必须采用一致的融合语义；只在 template 或只在 search 旁路注入会产生接口语义不对称 |
| Semantic-guided Fusion for Multiple Object Tracking and RGB-T Tracking | 2023, [10.1049/ipr2.12861](https://doi.org/10.1049/ipr2.12861) | 低分辨率语义分支先选 top-K 区域，高分辨率细节分支只在这些区域做注意力，降低背景计算与干扰 | M1 可以借鉴“先候选区域选择，再局部融合”，而不是在 256 个 search token 上无条件传播消息 |
| Robust RGB-T Tracking via Consistency Regulated Scene Perception / semantic scene reasoning | 2023, [10.1109/icip49359.2023.10222904](https://doi.org/10.1109/icip49359.2023.10222904) | 通过目标-场景关系和跨域一致性调节场景感知 | M1 的 hard-background 应被作为结构化场景关系建模，而非事后挑一个错误峰做统计 |
| FADSiamNet: Feature Affinity Drift Siamese Network | 2024, [10.1007/s13042-024-02420-z](https://doi.org/10.1007/s13042-024-02420-z) | 将 feature affinity drift 作为 RGB-T tracking 问题处理 | 这与 M1 的模板更新直接相关：模板不能只按置信度更新，必须监测目标特征关系是否发生漂移 |
| Efficient Transformer-based Single-stage RGB-T Tracking With Larger Search Region and Acceleration | 2024, [10.1145/3663976.3664032](https://doi.org/10.1145/3663976.3664032) | 更大 search region 与高效 Transformer 结合，提高目标出界/大位移容错 | M1 的 crop 漂移和恢复不足不能靠局部融合完全修复；搜索范围与恢复机制是独立瓶颈 |
| Mining Representative Tokens via Transformer-based Multi-modal Interaction for RGB-T Tracking | 2025, [10.1016/j.patcog.2025.112162](https://doi.org/10.1016/j.patcog.2025.112162) | 代表性 token 挖掘和多模态交互，核心是减少无效 token 进入融合 | 与 M1 最匹配的成熟思想是 token selection；它比全局 reliability 更接近目标/背景区分，但必须用目标/干扰物监督验证 |
| Robust RGB-T Object Tracking via Multilevel Fusion and Motion-cue-based Correction | 2025, [10.1007/s11760-025-04391-2](https://doi.org/10.1007/s11760-025-04391-2) | 多层融合 + motion cue correction；用运动线索纠正外观融合错误 | M1 的状态反馈链条缺少独立 motion/recovery 约束；仅修改外观融合无法处理错峰后的连续漂移 |
| Transformer-Based RGB-T Tracking With Channel and Spatial Feature Fusion | 2025, [10.1109/jsen.2025.3579339](https://doi.org/10.1109/jsen.2025.3579339) | channel + spatial feature fusion，同时控制通道可靠性和空间位置 | M1 目前把 channel/message reliability 与 target spatial reliability 混在一起；两者必须分开预测 |
| RGB-T Object Tracking Method Based on Multi-granularity Adaptive Fusion | 2026, [10.1038/s41598-026-67207-4](https://doi.org/10.1038/s41598-026-67207-4) | 多粒度 feature pyramid 和 adaptive fusion，针对尺度变化与 CNN/Transformer 表示差异 | 与 M1 当前 16×16 单尺度读出直接相关；尺度/分辨率不足可能比模块类型更早限制目标背景区分 |

## 别人的解法可以归纳为五条

1. **互补性不是一致性**：使用 common/specific 分解、互补性建模或重选特征，而不是把 RGB/TIR cosine 高当成目标可靠。
2. **显式干扰物和场景建模**：SiamCDA、scene perception、semantic-guided fusion 都把背景/周围区域放进目标函数或候选选择。
3. **代表性 token/区域选择**：先压制无效 search token，再做跨模态交互；避免所有背景 token 获得同等消息传播。
4. **漂移与更新是独立模块**：feature affinity drift、motion correction、大 search region、candidate elimination 处理的是状态和模板反馈，不是 fusion operator 本身。
5. **多层/多粒度定位**：FPN、multi-layer fusion、multi-granularity fusion 让小目标、尺度变化和 offset/size 有可用信息；单个 16×16 CENTER 峰不能承担全部任务。

## 对 M1 的根因判断

### 首因：目标条件监督缺失

M1 的 query/reliability/participation 是无监督的一致性量，实验证明它对 target-hard-background margin 的预测能力不足。它甚至在 loc3 已偏向背景，loc9 更明显。因此后接 loss 只会把错误信号拟合得更稳定。

### 第二个问题：融合边界与定位目标不一致

内部 MHSA/TBSI 改的是 token message；旁路 PVL、NLformer、CBAM 等改的是特征残差，但最终优化和选择仍由 CENTER peak、size、offset 决定。特征变化没有被强制转化为 GT peak margin，导致“表征变化”和“决策改善”脱钩。

### 第三个问题：模板更新和闭环恢复放大单帧错误

固定状态下很多变化很小，闭环中却出现大幅涨跌，说明错峰后的 state/crop/template feedback 是放大器。旁路修正即使在当前帧略有帮助，也可能改变下一帧 crop，使目标出界并污染模板。

### 第四个问题：空间和尺度信息不足

M1 主要在 16×16 search 网格上做融合和 CENTER 读出。对细长、小目标或目标/背景纹理相似的事件，target token 与 hard background token 的表示容易混合；即使目标中心接近，size/offset 仍可能错误。

## 为什么反复加模块仍然无效

你尝试的内部和旁路模块虽然形式不同，但共同满足以下条件：

- 输入仍是相同的目标不明确 token；
- 没有显式 hard-background / distractor 监督；
- 没有保证 residual 能量集中在目标区域；
- 没有阻断错误峰对 state/crop/template 的反馈；
- 没有把融合 margin 与 CENTER ranking 绑定。

所以这些候选不是在解决五个不同问题，而是在同一条未闭合的决策链上反复改变中间表示。不同模块得到相似的失败，是共性接口和监督瓶颈的证据。

## 推荐路线

### P0：停止通用旁路和当前 uncertainty loss

现有证据不支持继续换模块，也不支持直接给 global reliability 加 loss。

### P1：构造“目标身份保持 + 干扰物抑制”的最小训练对象

只允许一个小 head 读取空间 token，输出 target score 与 uncertainty；监督必须来自 GT token 和 hard-background token 的 signed ranking。它的输出必须直接参与 search token gating 或 candidate score，而不是只作为诊断量。

准入条件：

- 独立 train/dev 序列上 margin sign accuracy 明显高于当前 `61.5%`；
- GT rank 在固定状态下稳定改善；
- 非目标 residual/消息能量不高于目标区域；
- 关闭 head 后严格 identity parity。

### P2：把模板更新改成保守的 candidate-conditioned update

只有当目标候选、模态互补性、时序一致性同时满足时才更新模板；否则保持旧模板并触发扩大 search/recovery。这个问题不应由 MHSA 内部 uncertainty 单独承担。

### P3：单独审计 CENTER 的 size/offset 与 recovery

对目标中心排名已经接近但 IoU 仍低的序列，优先检查 size/offset 解码与状态更新；这类问题靠融合模块无法解决。

## 最终判断

你现在不是“模块选错了”，而是 M1 缺少一条完整的目标身份决策链。文献中有效路线都至少包含以下三类机制中的两类：

- 显式目标/干扰物选择；
- 跨模态互补性而非一致性建模；
- 模板漂移、候选淘汰和状态恢复。

因此下一步不应再替换新的普通融合模块。最值得做的是一个小而可审计的 **token-level target/distractor ranking + conservative template update** 实验；如果这个监督头在独立序列上仍不能改善 GT rank，就应停止融合线，转向 CENTER 解码和状态恢复，而不是继续堆模块。

