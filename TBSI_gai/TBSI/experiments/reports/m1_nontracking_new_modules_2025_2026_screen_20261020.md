# M1 AMCM 新模块候选筛选（2025–2026，非跟踪领域）

## 筛选边界

当前 AMCM 合同是：输入 `search_rgb/search_tir: [B,C,16,16]`、`template_rgb/template_tir: [B,64,C]`，输出两路同形状 search feature；输出在 `tbsi_fuse_search` 之前注入。目标是改善 M1 已观察的 crop 内错误峰竞争、RGB/TIR 冲突和错误状态更新，而不是增加 CENTER head 后处理。

候选必须满足：

1. 论文年份为 2025 或 2026，且原任务不是单目标跟踪；
2. 算子能在 AMCM 内部实现为同形状双流调制或交互残差；
3. 能解释目标/背景分离、跨模态冲突或局部上下文中的至少一个核心问题；
4. 明确记录论文是否依赖额外监督、蒸馏、不确定性标签或任务专用输入；
5. 与此前候选 `CMX/FiLM/MMTM/SeAFusion/CDDFuse/DIDFuse/SwinFusion/GMU/CBAM/SKNet/Dynamic Convolution/TokenFusion/Focal Modulation/CrossViT` 去重。

“未在 tracking 中使用”是截至本次检索对论文标题、摘要、DOI 和公开检索结果的判断，不是对全部后续论文的绝对证明。

## 候选清单

| 优先级 | 2025–2026 候选 | 原领域与核心机制 | AMCM 接口映射 | 额外 loss/输入依赖 | 对 M1 的价值与风险 |
|---:|---|---|---|---|---|
| 1 | **Uncertainty-aware Feature Modulation + Reliability Auditing**（2026） | 生理多模态；噪声扰动敏感度估计模态不确定性，再调制融合 | 用 RGB/TIR template-conditioned sensitivity 生成两路 reliability map，调制 search residual | 原文依赖辅助分类、互蒸馏、噪声扰动和 EMA；直接迁移可先去掉审计与蒸馏 | 最直接对应“模态失真/冲突”；需防止不确定性只变成全局 gate |
| 2 | **Orthogonal Sequential Fusion**（2025） | 通用多模态学习；按顺序融合并约束模态增量正交，减少重复/冲突信息 | `delta_rgb` 与 `delta_tir` 做正交残差分解后回注两路 search | 通常需要正交/去相关辅助项；可先用 tracking loss 做可拟合性检查 | 最符合“避免 RGB/TIR 冲突和重复增强”；风险是正交约束损伤目标共享信息 |
| 3 | **Region-adaptive Morphology-guided Feature Modulation**（2026） | 脑肿瘤分割；形态先验生成区域自适应调制 | 模板匹配图替代形态 mask，生成局部 residual route | 原文依赖分割标签和 morphology prior；迁移时必须用 template-derived soft region | 与目标/背景空间分离最贴近；需避免把错误模板形态当成监督 |
| 4 | **CAFF-Net: Diffusion Enhancement + Cross-modal Feature Modulation**（2026） | 多模态融合；扩散增强后做跨模态特征调制 | 将扩散增强器简化为冻结/轻量上下文，保留 cross-modal modulation | 原文可能依赖扩散训练和融合重建 loss；不宜直接搬完整 diffusion | 适合清理模态噪声；复杂度和训练成本较高，先做无扩散 cross-modal core |
| 5 | **CALA-HSI Content-adaptive Local Attention**（2026） | 高光谱分类；内容自适应局部 attention | 在 16×16 search 上生成内容自适应局部邻域权重，template map 作为条件 | 主要依赖分类 loss；无额外目标标注要求 | 直接针对 crop 局部背景峰；风险是局部 attention 可能增强纹理背景 |
| 6 | **Spectral-spatial Attention Decoupling + Adaptive Gating**（2026） | 高光谱分类；解耦光谱/空间注意并自适应门控 | C 维通道关系对应模态/通道可靠性，空间分支对应目标峰 | 通常只需分类监督；原始光谱维度与 RGB/TIR 不同 | 可同时检查通道冲突和空间定位；避免退化成 CBAM 式显著性 |
| 7 | **N-gram Transformer + Local Adaptive Fusion**（2026） | 高光谱/多光谱融合；局部 n-gram token 交互和局部自适应融合 | 用 3/5/7 邻域 token n-gram 统计替换当前平均模板匹配 | 原文依赖重建/融合 loss；迁移可先只训练 AMCM | 对局部峰结构有针对性；token 组合数和显存需限制 |
| 8 | **Adaptive Spatial-spectral Collaborative Attention**（2026） | 高光谱 pansharpening；空间与谱协同注意 | 双流空间关系与通道关系协同生成 residual | 依赖重建/质量指标 loss；没有模板输入依赖 | 能建模模态互补；需将全局谱质量改成 template-conditioned |
| 9 | **Multimodal Feature Modulation Adapter**（2025） | visual grounding；轻量多模态条件调制 adapter | template tokens 作为 grounding condition，调制 RGB/TIR search tokens | 原文依赖语言/grounding condition；这里改用视觉模板，不需新增 loss 起步 | 接口最简单、参数少；若模板均值不可靠，仍会复现 FiLM 失败 |
| 10 | **Dynamic Multimodal Fusion via Meta-learning**（2025） | 多模态推荐；样本条件动态融合权重 | 用模板/搜索一致性生成 sample-conditioned fusion coefficients | 原文依赖 meta-learning episode；迁移时先取消元学习 | 能检验“每个事件应使用不同融合强度”；风险是只学全局权重 |
| 11 | **TCTFusion: Triple-branch Cross-modal Transformer**（2025） | 红外-可见光图像融合；三分支 cross-modal transformer | 三分支改为 RGB self、TIR self、template-bridged cross 分支，输出双流残差 | 依赖融合重建/结构 loss；可先只用 tracking loss | 与 RGB/TIR 输入最匹配；但同领域已有大量 fusion 工作，创新风险较高 |
| 12 | **Edge-guided Cross-modal Transformer**（2025） | 红外-可见光融合；边缘引导跨模态交互 | 用 search 梯度/局部高频近似 edge prior，并由模板匹配抑制背景边缘 | 原文依赖 edge/reconstruction loss；迁移需避免固定 edge 监督 | 可能改善目标轮廓；也可能把背景边缘变成错误峰 |
| 13 | **CTGAN: Cross-modal Attention-guided GAN**（2025） | 红外-可见光融合；cross-modal attention + hybrid CNN/Transformer | 仅保留 cross-modal attention 调制，不迁移 GAN 判别器 | 原文强依赖 GAN/reconstruction loss，不建议直接复现 | 结构可用但 loss 合同风险大，优先级低 |
| 14 | **Text-guided Cross-attentive Visual Feature Modulation**（2026） | 皮肤病多模态识别；跨模态条件视觉调制 | 用 template token 代替 text condition，做 cross-attentive residual | 原文依赖文本/类别语义；视觉模板迁移需验证 | 可作为模板条件调制对照；语义条件缺失是主要风险 |

## 独立排序与建议

### 第一梯队：优先做合同诊断

1. **Uncertainty-aware Feature Modulation**：最直接回答“哪一路在当前 crop 不可靠”，同时能把不确定性限制在空间位置而非平均 gate。
2. **Orthogonal Sequential Fusion**：最直接回答“RGB/TIR 冲突时是否重复或互相污染”。
3. **Region-adaptive Morphology-guided Modulation**：最直接回答“目标区域和强背景区域是否需要不同的局部修正”。
4. **CALA-HSI**：只使用内容自适应局部关系，适合检验 M1 的局部峰问题，不依赖显式 tracking 先验。

### 第二梯队：第一梯队失败后再做

5. Spectral-spatial decoupling + adaptive gating；
6. N-gram Transformer + Local Adaptive Fusion；
7. Adaptive spatial-spectral collaborative attention；
8. CAFF-Net 的轻量 cross-modal modulation core。

### 第三梯队：接口可行但创新/监督风险较高

9. Multimodal Feature Modulation Adapter；
10. Dynamic Multimodal Fusion via Meta-learning；
11. TCTFusion；
12. Edge-guided Cross-modal Transformer；
13. CTGAN cross-modal attention core；
14. Text-guided cross-attentive modulation。

## 与当前实验的关系

CBAM 在 25 个 M1 事件上取得了局部上涨但出现严重序列退化，说明“通用空间显著性”不是目标条件可靠性建模。Focal Modulation 和 SwinFusion 也没有改善困难组，说明单纯扩大局部/多尺度上下文不能替代目标条件信息。因此下一候选必须至少显式建模以下一项：

- 模态不确定性或跨模态冲突；
- 模态增量的正交/去冗余关系；
- template-conditioned 的区域结构，而非全局均值相似度；
- 内容自适应局部关系，而非固定平均融合。

## 推荐的最小筛选顺序

先对 1–4 做 parameter-free signal proxy 和 identity/gradient/fusion-transfer 合同检查；每轮只实现一个 AMCM mode。通过条件为：GT/background AUC 高于当前 AMCM、residual 在 GT 与强背景上有结构差异、tracking loss 方向下降，并且 25 事件中困难组不出现 CBAM 式少数大涨抵消大退化。未通过合同检查的候选不进入训练，不通过增加 loss 或增大调制强度补救。

## 参考来源

- Trustworthy Multimodal Learning via Uncertainty-aware Feature Modulation and Reliability Auditing, 2026, DOI `10.2139/ssrn.7327625`.
- Region-adaptive multimodal learning with morphology-guided feature modulation for brain tumor segmentation, 2026, DOI `10.1007/s44163-026-02348-z`.
- CAFF-Net: A multimodal fusion network based on diffusion enhancement and cross-modal feature modulation, 2026, DOI `10.1109/cnml68938.2026.11453228`.
- Orthogonal Sequential Fusion in Multimodal Learning, 2025, DOI `10.23919/fusion65864.2025.11124022`.
- A Simple yet Efficient Multimodal Feature Modulation Adapter for Visual Grounding, 2025, DOI `10.1109/ecnct66493.2025.11172398`.
- Hyperspectral image classification via Transformer-based spectral-spatial attention decoupling and adaptive gating, 2026, DOI `10.1080/19479832.2026.2618653`.
- Hyperspectral and multispectral image fusion via N-gram transformer and local adaptive fusion strategy, 2026, DOI `10.1016/j.image.2026.117552`.
- Adaptive spatial-spectral collaborative attention network for hyperspectral pansharpening, 2026, DOI `10.1007/s11760-025-05086-4`.
- TCTFusion: A Triple-Branch Cross-Modal Transformer for Adaptive Infrared and Visible Image Fusion, 2025, DOI `10.3390/electronics14040731`.
- Using Edge-Guided Cross-Modal Transformer for Infrared and Visible Light Image Fusion, 2025, DOI `10.1145/3747912.3747945`.
- CTGAN: Cross-Modal Attention Guided GAN with Hybrid CNN-Transformer for Infrared and Visible Image Fusion, 2025, DOI `10.1109/aibdf67964.2025.11440755`.
- Text guided cross attentive multimodal learning with visual feature modulation for automated skin lesion detection, 2026, DOI `10.1038/s41598-026-47271-6`.
