# M1/MSHA 的 2026 模块文献收敛（2026-10-26）

> **历史方案，非当前执行计划（2026-10-26 清理标记）：** 本文原始数据、引用和推导保留；其中候选优先级、准入条件和下一步命令不再作为当前指令。当前范围见项目 `README_当前研究.md`；本标记不表示所有假设已被证伪，也不批准新训练。清理前原文见 Git `c0f1aff`。


## 结论

本轮只做论文与公开源码调研，没有修改模型代码、没有训练或测试。建议将后续研究范围收敛到一个**文献启发的待验证问题**：借鉴 SEATrack 的跨模态注意力图互引，检查 M1 的 RGB/TIR 匹配关系是否在 crop 内目标与 hard-negative 之间发生系统性冲突。它目前不是获准实现的算子，也不是已证明的 MSHA 首因。

M1 的系统级失败链可描述为“目标仍在近似搜索 crop 内时出现中心选择失配，随后状态反馈使误差放大并导致目标出界/恢复不足”；现有资产没有证明 MSHA 是这条链的唯一或首要原因。Early gate full-244 的配对平均差为 `-0.213 pp`，序列 bootstrap 95% CI `[-1.30,+0.88]`；Early gate full15 相对 M1 的 `+0.349 pp` 是未完成严格 gate-off 配对的单点估计。两者都不能支持稳健增益结论。

截至本日，Value 幅度调节、连续共支持重分配、接收端 Value 残差和可学习 Value 重构均没有给出稳定的定位排序证据。最新 Value reconstruction quick3 在 retrospective quick7 cohort 上 AUC 为 `-4.387 pp`，因此不再扩展 Value 路由/重构作为近期算子搜索方向。该结论仅排除各自已测合同，不外推为所有内容交互都无效。

## 候选对照

| 工作 | 2026 状态与代码 | 核心信息流 | 对 M1 的借鉴价值 | 当前处理 |
|---|---|---|---|---|
| **SEATrack: Simple, Efficient, and Adaptive Multimodal Tracker** | CVPR 2026 Oral；arXiv:2604.12502；作者官方 MIT 仓库，审计 commit `33905559007284c50f9e8ea268defc45afa0d0b7` | 两个模态流分别由 Q/K 得到未归一化匹配 logits；共享低秩适配 K/V；再以双向可学习系数把每个模态 logits 朝另一模态 logits 移动，softmax 后乘各自 V。官方实现位置：`lib/models/layers/attn_blocks.py::CEBlock_AP.amglora_attn` 与 `forward`。代码中两个方向的 scaling 是逐层可学习标量，初始化为 1；它们不是由当前帧质量网络预测的动态可靠性分数。 | **最贴近当前 MSHA 主线。** 它改变 attention logits/匹配分布，和 Early gate 的 post-attention Value 缩放在作用点上不同。可借鉴“显式处理两模态匹配图冲突”的假设。 | 只保留 AMG 的思想作候选；不照搬整套 two-stream backbone，也不把其结果当作 M1 预期增益。需要先验证 M1 当前注意力图确有与定位失配相关的跨模态冲突。 |
| **ESMTrack: End-to-End Self-Supervised RGB-T Tracking without Modality Misleading** | 2026-09-29 arXiv 预印本；作者公开仓库 `LiShenglana/ESMTrack` | 融合/RGB/TIR 三分支；用 APCE 响应差评估模态贡献，以 grounding 与跨帧 temporal triplet 目标训练，并以 forward-backward consistency 筛样本。 | 提示可靠性可以进入表示学习与训练目标，而非只在推理时缩放 Value。 | 暂不选。它改整套训练目标与分支结构，既不局限 MSHA，也会重引入用户希望避免的“再加一个可靠性决策”叙事。 |
| **Uni-MDTrack: Learning Decoupled Memory and Dynamic States for Parameter-Efficient Visual Tracking in All Modality** | 2026-03-15 arXiv 预印本；本轮未找到可核验的官方代码仓库 | Memory-aware compressed prompt 与逐层 Dynamic State Fusion，将压缩历史信息和连续动态状态送入主干。 | 与“失败后反馈漂移、目标出界、恢复不足”在问题层有联系。 | 暂不选。它把研究对象移到时序状态/记忆，超出当前 MSHA 中间交互边界；公开源码状态本轮未核实。 |
| **Group Orthogonal Low-Rank Adaptation (GOLA)** | AAAI 2026；arXiv:2512.05359；作者公开仓库 `MelanTech/GOLA` | 将低秩适配子空间分组并施加组间正交约束，以减少冗余、促成互补表征。 | 可作为特征子空间互补的跨领域参考。 | 不进入当前候选。它优化的是 PEFT rank 利用率，不提供 crop/目标条件，也没有直接针对 M1 中间 attention 信息流。 |

## SEATrack 信息流与边界

论文 §3.2 的基本步骤是：

```text
RGB tokens -> Q_rgb, K_rgb, V_rgb -> logits_rgb
TIR tokens -> Q_tir, K_tir, V_tir -> logits_tir
logits_rgb <- logits_rgb + w_tir * (logits_tir - logits_rgb)
logits_tir <- logits_tir + w_rgb * (logits_rgb - logits_tir)
softmax -> each stream reads its own V -> projection + residual
```

所以它直接作用在 softmax 前的匹配关系，而不是在注意力已完成后缩放跨模态 Value。SEATrack 的 HMoE 另在 attention/FFN 后对拼接的模板或搜索 token 做 sub-token dispatch、低秩 expert 变换和 token-expert 合并。**本轮不推荐 HMoE**：M1 已有 global/target-weighted template expert 的 RPP-v1 结果（AUC `52.01` 与 `50.96`，control `52.02`），且后续多条 expert/router 分支也未形成可靠整体改善；再移植另一种 MoE 会与既有资产重叠并增加归因变量。

### 与 Early gate / v1-v2 的对照

- Early gate：用早期模板—搜索和模态一致性特征预测 token/方向权重，作用于已有 cross-Value 消息；Q/K 与当前层 attention logits 不被直接修改。
- DCVR-v2 草案：细化 Value 来源预算/分配，仍在已形成的 Value 消息路径上；它是提案，不是独立验证过的完整贡献。
- SEATrack AMG：从另一模态的匹配 logits 得到指导，直接改变匹配分布。借鉴它能够形成“关系形成层”的研究问题，但不代表简单替换缩放系数就构成新颖贡献。

风险与区别必须说清：SEATrack 已经提出跨模态 attention-map alignment；CGATrack、TUMFNet、UATrack 等相关工作也覆盖跨模态质量/交互。因此不能宣称“首次跨模态注意力对齐”或“首次可靠性融合”。可研究的窄边界只能是 M1 特有的 MSHA 模板—搜索匹配冲突：在目标仍处 crop 内的失配事件中，正确模态支持与退化/干扰模态支持是否发生定位相关分歧，以及这种分歧是否能通过 MSHA 的关系形成路径传到 CENTER 读出。该区别尚待证据支持，不是现成的新颖性结论。

另一个重要风险是负迁移：若一模态当前响应更差，双向互引仍可能把错误峰传播到健康模态。SEATrack 论文使用的是分离的双流 Transformer，并联合 LoRA、AMG 与 HMoE；其综合结果不能单独证明 AMG 在 TBSI 的共享联合 MSHA 中有效。论文报告中的消融只能支撑其原模型合同，不能代替 M1 上的配对验证。

## 收敛建议

当前只保留 **SEATrack AMG 的注意力关系互引思想** 作为下一候选来源；不是立刻实现，也不同时加 HMoE、动态专家、时序记忆或额外质量门控。

进入任何训练前，先用既有 M1/早期 gate checkpoint 的同一 crop/state 证据确认三件事：

1. MSHA 中可按 RGB/TIR 查询与模板来源拆出的匹配图，在失配起点确实分歧；
2. 分歧发生在目标位置与 GT 外 hard-negative 竞争上，而不只是 attention 熵或图像外观差异；
3. 更可靠的一侧确实把另一侧引向目标而不是共同干扰峰，且该变化在同状态 CENTER rank/margin 或 decoded box 上可见。

若这些信号不存在，停止 attention-alignment 方向；若存在，再决定是否做一个单变量、MSHA 内部的机制检验。当前 fixed-state replay parity 尚未解释，quick3 control repeat 也有 `11.15 pp` AUC 差异，因此当前训练比较合同尚未准入。这个调研没有运行上述检查。

## 来源与复核范围

- SEATrack 论文：<https://arxiv.org/abs/2604.12502>（2026-04-14；论文 HTML §3.2/§3.3 与消融段落）。
- SEATrack 官方源码：<https://github.com/AutoLab-SAI-SJTU/SEATrack>；MIT；审计 commit `33905559007284c50f9e8ea268defc45afa0d0b7`；重点文件 `lib/models/layers/attn_blocks.py`、`lib/models/layers/attn.py`、`lib/train/base_functions.py`。
- ESMTrack 论文与代码：<https://arxiv.org/abs/2609.37162>、<https://github.com/LiShenglana/ESMTrack>（预印本；README 声明官方实现和数据/权重位置）。
- Uni-MDTrack：<https://arxiv.org/abs/2603.14452>（预印本；本轮未核实官方代码）。
- GOLA：<https://arxiv.org/abs/2512.05359>、<https://github.com/MelanTech/GOLA>（AAAI 2026）。
- 本地实验依据：`m1_msha_v1_v2_operator_attribution_20261026.md`、`m1_msha_v1_v2_operator_refinement_20261026.md`、`m1_receiver_centered_value_fixed28_20261026.md`、`m1_early_gate_value_recon_quick3_20261026.md`、`template_expert_p1_rpp_v1_serial_20260928.md`、`m1_early_gate_control_repeatability_20261026.md`。

## 资产范围

此记录是文献/源码 scouting，不是训练、推理或模型诊断结果。没有加载数据或权重，AUC/PR/NPR/OP50/OP75/FPS 均为 `NA`，GPUHours `0`。本地代码没有修改。工作区 Git HEAD 为 `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9` 且当前 dirty；SEATrack 源码只在 `/tmp` 临时检视，权威引用为上面的官方 URL 和固定 commit。
