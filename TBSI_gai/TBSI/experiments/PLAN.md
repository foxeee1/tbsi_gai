下面分析的是**当前已实现的自适应混合实验组** `M1_AMCM_DR_ADAPTIVE_NOSUP_QUICK15_3EP_20261021`，不是未来拟议的新模块。最关键的位置判断是：**它在 M1 的三处 TBSI 模板融合和整个 ViT 主干之后、`tbsi_fuse_search` 之前**。因此它能调整进入搜索融合 Neck 的两路特征，却不能直接引导已经完成的 loc3/6/9 模板 MHSA。【TBSI_gai/TBSI/experiments/tbsi_track/M1_AMCM_DR_ADAPTIVE_NOSUP_QUICK15_3EP_20261021.yaml】【TBSI_gai/TBSI/lib/models/tbsi_track/tbsi_track.py:491】【TBSI_gai/TBSI/lib/models/tbsi_track/vit_tbsi_care.py:265】

## 一、完整信息流

下图中的 shape 采用当前配置：每模态模板 `128×128`、搜索区域 `256×256`、patch stride 16、特征通道 768。`B` 为 batch size。【TBSI_gai/TBSI/experiments/tbsi_track/M1_AMCM_DR_ADAPTIVE_NOSUP_QUICK15_3EP_20261021.yaml】

```text
初始化帧目标框 ──→ 裁剪 RGB/TIR 模板，各 [B,3,128,128]
                    └─→ 目标框覆盖率权重 w_fg：[B,64]
当前 tracker state ─→ 裁剪当前 RGB/TIR search，各 [B,3,256,256]
                              │
                              ▼
Stage 1  共享 ViT patch embedding
         每模态：模板 [B,64,768]；搜索 [B,256,768]
                              │
                              ▼
Stage 2  双流 ViT blocks；在 loc3、loc6、loc9 插入 TBSI-E
         TBSI-E：RGB/TIR 模板 MHSA 融合及模板↔搜索交互
         其余 ViT blocks 继续处理两路 token
                              │
                              ▼
Stage 3  主干最终输出
         RGB：[模板 64 | 搜索 256] × 768
         TIR：[模板 64 | 搜索 256] × 768
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
       模板 T_rgb、T_tir                搜索 X_rgb、X_tir
       各 [B,64,768]                  各 [B,768,16,16]
                └─────────────┬─────────────┘
                              ▼
Stage 4  【新增 AMCM：自适应混合 + 定向残差】
         对每个 search 位置、每个模态：
         ① 全模板匹配 M_global
         ② 前景加权模板匹配 M_fg 与注意力熵 H
         ③ λ = 1 − max(H_rgb, H_tir)
         ④ M = λ·M_fg + (1−λ)·M_global
         ⑤ 局部上下文 + 两模态差异 + 两张 M → 门控与消息
         ⑥ 用另一模态的 M 路由消息，形成 Δ_rgb、Δ_tir
         输出：X'_rgb = X_rgb + Δ_rgb
               X'_tir = X_tir + Δ_tir
         两路 shape 仍为 [B,768,16,16]
                              │
                              ▼
Stage 5  tbsi_fuse_search 搜索融合 Neck
         拼接 [X'_rgb,X'_tir]：[B,1536,16,16]
         3×3 Conv + BN + ReLU → [B,768,16,16]
                              │
                              ▼
Stage 6  CENTER Head
         score：[B,1,16,16]
         size：[B,2,16,16]
         offset：[B,2,16,16]
                              │
                              ▼
Stage 7  推理读出与闭环
         score × Hann window → 最高峰位置
         在该位置取 size、offset → 框 → 映回原图、裁剪
         预测框写入 tracker state → 决定下一帧 search crop
```

Stage 4 的实际调用与 Stage 5 的拼接见模型代码；CENTER 分别预测分数、尺寸和偏移，推理时 tracker 另将 Hann window 乘到分数图，再用选中位置的尺寸、偏移解框。【TBSI_gai/TBSI/lib/models/tbsi_track/tbsi_track.py:522】【TBSI_gai/TBSI/lib/models/layers/head.py:98】【TBSI_gai/TBSI/lib/test/tracker/tbsi_track.py:325】【TBSI_gai/TBSI/lib/test/tracker/tbsi_track.py:498】

### 新模块内部：哪个量真正控制哪个量

设 \(x_i^m\) 为模态 \(m\) 的搜索位置 \(i\)，\(t_j^m\) 为模板 token。代码中的匹配先算归一化特征的余弦相似度，再对模板 token 做 softmax 加权平均。前景版本把初始化目标框得到的 token 覆盖权重加到匹配 logits 中；**它是软权重，不是只保留前景的硬掩码**。【TBSI_gai/TBSI/lib/models/layers/amcm_reliability.py:267】【TBSI_gai/TBSI/lib/models/layers/tbsi_layer.py:15】

\[
M_{i,m}^{g}=\operatorname{Match}(x_i^m,T^m),\qquad
M_{i,m}^{f}=\operatorname{Match}(x_i^m,T^m;w_{\mathrm{fg}})
\]

\[
\lambda_i=1-\max(H_{i,\mathrm{rgb}},H_{i,\mathrm{tir}}),\qquad
M_{i,m}=\lambda_iM_{i,m}^{f}+(1-\lambda_i)M_{i,m}^{g}
\]

这里两模态**共用同一个** \(\lambda_i\)：任一模态的前景注意力熵高，两路都更倾向退回全模板匹配。混合得到的 \(M\) 不是直接加到 CENTER score；它进入后面的特征、门控和空间路由。【TBSI_gai/TBSI/lib/models/layers/amcm_reliability.py:524】

再令局部上下文为 \(c_{\rm rgb},c_{\rm tir}\)，模块构造：

\[
F_i=[c_i^{rgb},c_i^{tir},|c_i^{rgb}-c_i^{tir}|,
                  M_{i,rgb},M_{i,tir}]
\]

\[
(p_{i,rgb},p_{i,tir})=\operatorname{softmax}(\operatorname{Gate}(F_i))
\]

消息由独立投影产生，经过 `tanh × MAX_DELTA`，并减去各自空间均值；**给 RGB 的修正由 TIR 匹配图路由，给 TIR 的修正由 RGB 匹配图路由**：

\[
x_i^{\prime rgb}=x_i^{rgb}
 +p_{i,tir}\,\bar d_i^{rgb}\,
 2\sigma\!\left(\frac{M_{i,tir}-\overline M_{tir}}{0.07}\right)
\]

\[
x_i^{\prime tir}=x_i^{tir}
 +p_{i,rgb}\,\bar d_i^{tir}\,
 2\sigma\!\left(\frac{M_{i,rgb}-\overline M_{rgb}}{0.07}\right)
\]

其中 \(\bar d\) 表示有界且去除空间均值的学习消息。**所以“自适应混合权重有效”与“最终修正有效”之间还隔着消息方向、模态门控、空间路由、Neck 和 CENTER。**【TBSI_gai/TBSI/lib/models/layers/amcm_reliability.py:506】【TBSI_gai/TBSI/lib/models/layers/amcm_reliability.py:592】

## 二、按位置看缺陷

| 位置 | 现有设计的合理性 | 当前可能的缺陷与能力边界 |
|---|---|---|
| **模板前景权重** | 初始化框提供明确的目标位置先验；全模板分支可保留上下文。 | 权重来自**初始化框**，不能表达此后目标外观变化，也不标记 search 中的相似干扰物。前景加权仍可能对错误对象匹配得很集中。 |
| **global/foreground 匹配** | 两种证据互补；25-event 中 `basketball849` 的前景单用退化确实被混合大幅缓解。 | `global` 是**全部模板 token**，不等同于纯背景模型；两张图都是对各自模板的匹配强度，缺少“目标候选比最强背景候选强多少”的显式比较。 |
| **熵决定混合比例** | 注意力分散时回退全模板，是合理的保守启发式。 | 低熵只证明注意力集中，**不证明匹配到目标**。共用的 \(1-\max(H_{rgb},H_{tir})\) 还可能在一模态清晰、另一模态含糊时，同时压低清晰模态的前景证据。 |
| **定向消息与路由** | 允许一模态借助另一模态的匹配位置，且零初始化、有界，便于保护 parent。 | 路由相对的是**整张搜索图的均值**，不是最强 hard background；如果干扰物匹配更高，它同样获得较强路由。消息去空间均值也不保证正修正落在目标、负修正落在背景。 |
| **注入位置：主干后、Neck 前** | 保留 M1 已学到的主干，直接影响两路搜索特征，工程风险较低。 | 信息若在 loc3/6/9 或后续 ViT 中已被混淆，此处只能利用残存证据；它**不会重新训练或重新执行**上游模板 MHSA，也不直接控制 `tbsi_fuse_search` 如何使用新增信息。 |
| **Neck → CENTER → state** | 保持 M1 原读出与闭环，便于隔离 AMCM 的作用。 | AMCM 没有显式保证 Neck 后的目标/背景 margin、Hann 后的峰排名或选中峰的 box quality。一次错误框进入 state 后，会改变下一帧 crop；帧内修正无法自行找回已离开 crop 的目标。 |

这些是**结构上可解释的限制或待证假设**，不是把所有低 AUC 序列的首因都判给 AMCM。尤其 TBSI loc3/6/9 在 AMCM 之前、CENTER 与状态提交在它之后，位置关系决定了它只能覆盖整条失败链中的一段。【TBSI_gai/TBSI/lib/models/tbsi_track/vit_tbsi_care.py:316】【TBSI_gai/TBSI/lib/models/tbsi_track/tbsi_track.py:491】【TBSI_gai/TBSI/lib/test/tracker/tbsi_track.py:239】

## 三、已有实验究竟定位到了哪一段

**已经得到支持的是“全局回退能保护一类前景误匹配”。**相同 25-event 合同下，`basketball849` 的变化从 foreground-only 的 `−0.5033` 到 mix50 的 `−0.0268`、adaptive 的 `−0.0036`。但 `rightcar-chongT` 三组都约 `−0.381`；单改匹配混合不足以覆盖这种失败。另一方面，mix50 与 adaptive 的逐事件变化相关约 `0.9949`，尚不能把差异归功于熵的目标可靠性判断。【TBSI_gai/TBSI/experiments/reports/m1_amcm_all25_global_fg_adaptive_analysis_20261021.md】【TBSI_gai/TBSI/experiments/reports/m1_amcm_global_fg_adaptive_analysis_20261021.md】

**从算子到决策的有效传递尚未出现。**独立 train-side 的 50 条首次更新里，M1 的 GT 均排第一、最小 GT–背景 margin 为 `0.4321`；adaptive 与残差置零组相比，`0/50` 改变 GT rank、`0/50` 翻转最高峰，平均绝对 IoU 差 `9.21×10⁻⁵`。RGB/TIR 残差相对输入范数分别约 `1.78×10⁻⁴/3.71×10⁻⁴`。这说明**在这些高 margin 首帧中**分支近似 no-op；它不能排除低 margin 困难帧存在作用。【TBSI_gai/TBSI/experiments/reports/m1_amcm_adaptive_precision_contract_audit_20261021.md】

另需严格分开一个实现问题：旧 full-244 的 `43.65` AUC 不能充当自适应混合算法的有效负例，因旧轨迹涉及已复现的 FP16 熵 NaN；局部数值修复也不能充当性能正例。目前没有完成匹配 FP32 口径的全量指标。【TBSI_gai/TBSI/experiments/reports/m1_amcm_adaptive_precision_contract_audit_20261021.md】

## 四、对“位置”的结论

如果你的论文贡献是**在 M1 已有表征上，可靠地选择全局或前景证据并纠正搜索融合输入**，当前位置与问题相符。但模块成立需要证明一条连续因果链：

> **目标与硬背景的证据差异 → 混合选择正确 → 目标位置的有效残差 → Neck 后 margin 提升 → CENTER 峰和框改善。**

目前只对其中的**全局回退保护作用**有较明确的小范围证据；熵选择、残差方向和下游决策改善仍未连起来。

如果你希望贡献是**用不确定性引导 loc3/6/9 的 MHSA 跨模态模板融合强度**，那是**另一个注入位置与另一项假设**。当前 AMCM 读取的是最终主干模板与搜索特征，不会回到上游调节 MHSA。两者可以共享“目标条件可靠性”的研究问题，但不能把当前 post-backbone 实验称为 MHSA 内部自适应融合的验证。

因此，眼下最准确的定位是：**自适应混合已显示为一种有保护性的双证据提议机制，尚未显示为可靠的目标/背景判别与决策修正模块。**你要求先分析，我没有启动新实验或改动模型代码。












我的独立判断是：你目前卡住的核心，不是“还没找到足够聪明的融合公式”，而是**把低 AUC 当成一个问题来处理了**。现有资产显示，低 AUC 序列里至少混有早期模态证据弱、融合后排序不稳定、CENTER 选峰或框解码失配，以及错误状态被闭环放大等不同故障。把它们混成一个训练目标或一个融合模块，自然容易出现跷跷板效应。


最值得先锚定的，是这个可观测、也和你融合目标相连的失败场景：

> **目标仍完整落在当前 search crop 内，但 CENTER 的目标区域排序输给了局部 hard background；同时至少一个早期模态或融合前特征仍保留目标证据。**

这比“遮挡场景”“模态不可靠”或“融合权重不准”更适合作为研究问题：可以用几何和响应直接识别，不依赖人工场景标签；也明确了后融合修正的能力边界。若融合前已经没有任何目标证据，融合模块无法凭空恢复；若 CENTER 排名正确但框的尺寸或 offset 错，改跨模态融合也不是对症处理；若目标已经出 crop，帧内模块也无法单独恢复跟踪。

已有资产对这个判断的支持是有限但有方向的：

- [低 AUC 信息流报告](/root/autodl-tmp/TBSI_gai/TBSI/experiments/reports/m1_lowauc_information_flow_cases_20261020.md)显示，案例并不共享同一个首因：有的早层证据持续变弱，有的单模态证据增强但 CENTER 仍偏移，有的峰落在 GT 中心附近而 IoU 仍低。报告也明确指出，MHSA 中最早观察到差异，不等于已经证明它是因果首因。
- [MSHA 低 AUC 干预报告](/root/autodl-tmp/TBSI_gai/TBSI/experiments/reports/m1_lowauc20_msha_foreground_cross_20261022.md)表明，跨模态交互确实会改变 CENTER 输出，但关闭交互和限制前景 query 都是正负混合，不能推出“前景 query 限制就是稳定解法”。
- [固定状态 action replay](/root/autodl-tmp/TBSI_gai/TBSI/experiments/reports/m1_fixed_state_action_replay_20261012.md)中，已有信号在拟合集上看似有益，留出序列上却反向；说明从人为挑选的 margin 或可靠性信号设计调制，再观察少数样本，很容易得到不能迁移的规则。
- Adaptive AMCM 的旧 full-244 结果后来发现精度合同有问题；后续 train-side 固定状态分析中，残差对 CENTER 排名几乎没有影响。它们不能作为“该类模块已经能解决首更新崩溃”的依据。
- Early-gate 的 12 序列闭环回放有局部收益，但大部分均值增益由单序列贡献，且早期 IoU 未变。它支持“融合路由可能影响后续轨迹”，还没有证明改善的是目标/背景判别。

所以问题、原因、模块、验证应当连成下面这条链：

**观测到的失败事件**：目标在 crop 内，CENTER raw top-1 落在 GT 外的 hard background。  
**必要前提**：至少一个早期模态或融合前表示，在同一 crop 上对 GT 区域相对 hard background 有正证据。  
**因果检验**：只改 MSHA 的一项 query/message 交互，固定模板、state、crop 和 frame；同时用空间错位的非目标 query 控制作对照。  
**有效机制签名**：目标 query 干预提高 GT 对最强 hard-background 的 margin/rank，并传到 CENTER；错位控制不产生同样变化。  
**否定结果**：两种干预都不改变排序，或变化没有集中到 GT 区域，就停止改这个交互位置；若 margin 保持但框仍错，转查 CENTER 的位置和尺度解码。

你最近尝试的 train-side query-role smoke 还没有通过这个因果检验：平移对照在采集到的三个 TBSI 输出上是零差异，但 CENTER 框发生变化。因此这批结果不能用于判断目标 query 是否有效。这个采集合同先修好，之后才能谈完整 cohort 的结论。

你问“为什么始终没效果”，我认为主要有四个原因：

1. **样本单位错位。** 用整条低 AUC 序列评价帧内融合，混进了漂移、出界、恢复和框尺度误差；模块可能只影响少数帧，序列指标把不同机制平均在一起。
2. **失败定义太宽。** “目标背景区分差”没有固定成可重复事件规则，模块实际优化的信号可能与 CENTER 最终选峰无关。
3. **信号尚未证明可用。** agreement、可靠性、template query 强度或 residual 幅值只是特征，不等于目标效用。此前多个旁路模块都有明显中间变化，但没有稳定改变 target/hard-background 排序。
4. **缺少能排除伪因果的控制。** 训练短、样本少确实会增加方差，但不是首要解释；在监督路径、注入强度、目标事件定义和中间读数都未对齐前，加数据或 epoch 只会更贵地拟合不确定的问题。

因此我建议把近期研究范围收窄到 **“crop 内目标与局部干扰物的帧内排序失配”**，而不是笼统研究“融合不可靠”。已有的 M1、train-side fixed-state、早层 gate 和 query-role 资产都可以服务于这个问题；暂时没有证据支持再增加新的融合算子，也没有证据支持直接把前景限制模块送去训练。

当目标前景证据不存在时，修复边界是输入表征或更早的信息流；当目标证据存在、但经过 MSHA/TBSI 后 margin 才下降，才是主回路融合模块的明确靶点；当 margin 一直存在而 CENTER 仍选错，则是融合到读出的接口问题；错误选择只有进入后续 state 后才恶化，则属于闭环状态更新。把这些分开，你就能围绕一类明确故障改造现有模块，而不是继续要求一个模块同时挽救所有低 AUC 序列。