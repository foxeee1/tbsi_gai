# M1 非跟踪领域模块筛选（2026-10-20）

## 1. 筛选问题

M1 的已观察问题是 crop 内目标峰与背景峰竞争、目标出界/中心失配后的反馈漂移，以及 RGB/TIR 冲突时融合表征缺少稳定的目标选择性。现有 AMCM/LCGM 证据已经显示：

- LCGM 适配版的残差能到达 `tbsi_fuse_search`，但目标/背景 residual 幅值比约 `0.992/0.992`；
- 在原始 GIoU + L1 + CENTER focal loss 下，GT 区域 residual 方向使 loss 上升，RGB `10/10`、TIR `9/10`；
- 当前 template-all-token matching 低于随机，RGB/TIR map AUC 均值为 `0.144/0.106`；
- 因此下一候选必须先解决**输入信号是否含目标条件信息、空间选择性是否存在、训练目标是否与峰竞争一致**，不能只改变残差算子或放大调制幅度。

候选约束：

1. 原论文必须来自非单目标跟踪领域；
2. 能接收 `[B,C,H,W]` 的 RGB/TIR 搜索特征，并返回两个同形状流或一个融合特征；
3. 优先不依赖像素级分割标签，或者能在现有 tracking loss 下工作；
4. 接口可放在 AMCM 原位置：`(search_rgb, search_tir, template_rgb, template_tir) -> (rgb', tir')`；
5. 排除已经讨论或实现过的 IGF、MCCA、LCGM、AMCM residual/gate、频域 CFVI/TAFH/FMR、SATNet 和 tracking 专属模块。

## 2. 现有问题与接口协议

AMCM 当前入口是两路 search feature `[B,768,16,16]` 和两路 template tokens `[B,64,768]`；其输出必须仍是两路 `[B,768,16,16]`，随后拼接后进入已训练的 `tbsi_fuse_search`。模板可以只作为条件，不允许修改模板状态或 tracker state。关闭模块时必须精确 identity。

快速筛选应在固定的 20 个困难序列 + 5 个正常控制上完成，使用 `output/experiments/M1_PRECISION_SCREEN_MANIFEST_20261020/selected_events.csv`。晋级必须同时满足：目标/背景表示距离改善、GT-vs-hard-background margin 改善、GT rank 或 top-1 有改善、tracking loss 的 residual 方向不与 GT 冲突，并且关闭模块保持 parent parity。

## 3. 候选清单

评级含义：`A` 表示可以优先做 AMCM 边界最小验证；`B` 表示接口可行但有明显监督/分布风险；`C` 表示更适合作为对照或暂不做。论文任务不是跟踪，候选是迁移假设而非已验证结论。

| # | 非跟踪论文/模块 | 原始任务 | 可迁移机制 | AMCM 接口适配 | loss/监督依赖 | 对 M1 核心问题的判断 | 级别 |
|---:|---|---|---|---|---|---|:---:|
| 1 | **MMTM: Multimodal Transfer Module**, Joze et al., ECCV 2020 | RGB-D/多模态识别 | 用跨模态全局统计生成通道调制，显式建模模态互补 | 两流 pooled statistics -> 两个 channel scale；保留空间位置，输出同形状 | 通常只依赖任务 loss；不需要像素标签 | 比当前 LCGM 更稳定地估计模态可靠性，但只有全局通道信号，不能单独解决错误空间峰 | A |
| 2 | **CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation**, Zhang et al., 2023 | RGB-X 语义分割 | feature rectification + cross-modal fusion，利用跨模态互补和差异 | 将 rectification 分支拆成两个同形状输出；不引入 decoder | 训练依赖分割 CE/Dice 才能学空间选择性；无额外标签可先用 tracking loss | 最接近“保留有效模态、压制冲突模态”的需求；需要验证 tracking loss 是否足够驱动空间 selectivity | A |
| 3 | **SeAFusion: An Infrared and Visible Image Fusion Network Based on Semantic Awareness**, Tang et al., 2022 | 红外-可见光图像融合 | 用语义/显著性引导融合，抑制背景、保留目标结构 | 只取其 spatial fusion mask，生成 RGB/TIR 两路残差或 convex mixing | 原版依赖 fusion + semantic auxiliary loss；语义分支可去掉，但需保留边缘/目标 proxy | 直接针对 RGB/TIR 互补和背景抑制；风险是原任务像素重建目标与 tracking peak 不一致 | A- |
| 4 | **CDDFuse: Correlation-Driven Dual-Feature Fusion Network**, Zhao et al., 2023 | 红外-可见光图像融合 | 通过 correlation/low-high frequency 分解保留互补信息，减少冗余 | 只保留 correlation-driven spatial/channel fusion，禁止照搬频域路线 | 原版依赖重建/结构损失；可先用 tracking loss 做可拟合性检查 | 有模态相关性建模，但对 hard-negative 峰没有显式条件，复杂度偏高 | B |
| 5 | **DIDFuse: Deep Image Decomposition for Infrared and Visible Image Fusion**, Li et al., 2020 | 红外-可见光融合 | 将 common/detail 分解后按内容互补重组 | common/detail encoder 输出两路 correction；接口需固定通道 | 依赖重建、梯度和 SSIM 类 fusion loss | 可分离共同目标与模态特有背景，但像素级重建可能把背景纹理当有效内容 | B |
| 6 | **SwinFusion: Cross-domain Long-range Learning for General Image Fusion**, Ma et al., 2022 | 多源图像融合 | window/self-attention 与 cross-domain feature interaction | 取一层 cross-domain interaction，输出 fused residual；需裁掉重建 decoder | 依赖 fusion reconstruction loss | 长程互补能力强，但对当前 16x16 crop 峰选择可能过度建模 | B |
| 7 | **FiLM: Visual Reasoning with a General Conditioning Layer**, Perez et al., AAAI 2018 | 条件视觉推理 | 条件向量生成 feature-wise affine modulation，明确“条件控制内容” | template pooled token 生成 `gamma,beta`，对 RGB/TIR search 做 affine；zero-init 保 parity | 通常只依赖下游任务 loss；不需要额外标签 | 可把模板真正接入 AMCM，修复 LCGM 完全忽略 template 的缺陷；但全局条件仍缺空间目标定位 | A- |
| 8 | **GMU: Gated Multimodal Unit**, Arevalo et al., ICLR Workshop 2017 | 多模态情感/场景识别 | 学习每个模态的内容混合，而非平均融合 | 两流 concat 后生成 `z`，输出 `z*x_rgb+(1-z)*x_tir`；可扩展为 per-location gate | 仅依赖任务 loss；原版没有可靠性标签 | 是最小可解释的双流互补基线；空间 gate 需从 scalar 改为 map，否则无法解决背景峰 | A- |
| 9 | **CBAM: Convolutional Block Attention Module**, Woo et al., ECCV 2018 | 通用视觉识别 | channel attention + spatial attention 顺序调制 | 用双流 concat 生成 shared spatial mask，再分别输出；保留 identity 初始化 | 只依赖下游分类 loss | 能提供空间选择性，但没有模板/跨模态 target condition；适合做“空间算子是否足够”的对照 | B |
| 10 | **SKNet: Selective Kernel Networks**, Li et al., CVPR 2019 | 图像识别 | 多尺度分支竞争选择，按输入动态选择 receptive field | 两流共同统计生成 3x3/5x5 branch weights；每模态输出同形状 | 仅依赖任务 loss | 针对局部目标尺度和遮挡有帮助，但不能识别 RGB/TIR 哪一路是背景 | B |
| 11 | **Dynamic Convolution**, Chen et al., CVPR 2020 | 图像识别 | 输入条件下动态组合多个 convolution kernels | 用双流/template pooled condition 产生 kernel mixture；输出两流 | 仅依赖任务 loss；动态 kernel 训练可能不稳定 | 能提高局部建模容量，但更像算子增强，不能证明目标/背景信号来源 | C+ |
| 12 | **TokenFusion: A New Token Fusion Strategy for Vision Transformers**, Wang et al., 2022 | RGB-D/RGB-T 视觉 Transformer | 根据 token importance 交换/补全低质量 token | 在 16x16 search tokens 上生成 keep/replace mask，再还原两流 feature map | 原版通常依赖任务 loss；重要性可能需要蒸馏/重建 | 最贴合“错误 token 破坏融合”的假设，但 token 删除/替换会破坏 M1 的连续 feature 合同，风险最高 | B- |
| 13 | **Focal Modulation Networks**, Yang et al., NeurIPS 2022 | 图像识别 | 用输入相关的 focal context levels 生成调制信号 | 从 RGB/TIR 联合 context 生成 per-location modulation，再分别作用于两流 | 仅依赖任务 loss | 比 LCGM 更有多尺度上下文，但没有显式跨模态可靠性，需防止再次得到均匀残差 | B |
| 14 | **CrossViT: Cross-Attention Multi-Scale Vision Transformer**, Chen et al., ICCV 2021 | 图像分类 | cross-attention 用一类 token 引导另一分支 | 模板 pooled token 作为 query，search tokens 作为 key/value，输出两流 correction | 仅依赖任务 loss；没有 dense target 标签 | 可建立 template-conditioned search modulation，正面回应当前 AMCM 缺少模板使用；计算和接口改造中等 | A- |

## 4. 排除与降级理由

- **不把 MMTM 当最终方案**：全局 channel scale 可能改善模态质量，但现有失败是空间峰竞争；它只能作为低成本可靠性基线。
- **不先做 CBAM/SKNet/Dynamic Conv**：这些模块主要增加空间或尺度算子，不能回答“目标信号从哪里来”。如果在当前 10 帧上没有目标条件输入，算子容量增加仍会重演 LCGM。
- **不直接照搬 SeAFusion/CDDFuse/DIDFuse/SwinFusion**：它们的原始监督通常是像素重建、结构或语义损失，和 CENTER 峰排序不一致。可以迁移内部的 correlation/saliency operator，但必须先做无训练 forward 与 tracking-loss gradient contract。
- **不把 TokenFusion 作为第一候选**：token 删除或替换会改变已训练 TBSI 的 token 分布，除非先证明低质量 token 是共因，否则不能作为 AMCM 原接口的最小变量。
- **不继续 LCGM**：已经完成固定帧 target/background、梯度方向和融合边界审计，证据链不支持继续调强度或堆叠算子。

## 5. 推荐排序

### 首选：CMX-style cross-modal feature rectification

理由是它直接建模双模态差异和互补，输出仍是 feature map，能保持 AMCM 双流接口；它比 LCGM 多了显式 cross-modal rectification，而不要求把目标定位写成额外的工程后处理。最小版本只保留一层 cross-modal rectifier 和双流 residual，zero-init，先在 25 个筛选事件上测试：

1. 关闭时与 M1 identity parity；
2. 不加额外监督，先看普通 tracking loss 是否让 GT residual 方向变为负；
3. 再单独加 hard-background margin loss，不能和结构变化同一实验；
4. 通过后才做 full-data training。

### 第二候选：FiLM-style template-conditioned modulation

这是修复当前 AMCM 最大接口缺陷的最小方案：模板 token 终于真正进入调制。用模板前景 token 均值或 attention pooled token 生成两流 `gamma,beta`，从 `gamma=0,beta=0` 开始。它可能只提供全局条件，因此必须在固定筛选帧上检验空间 margin；若空间选择性不足，停止，不继续加 CBAM。

### 第三候选：MMTM-style cross-modal reliability baseline

它可以用很小的容量检验“模态质量估计”这个假设是否成立。若 MMTM 在困难组改善模态冲突但不改善峰排序，则说明瓶颈不是通道可靠性，而是目标条件空间建模，应转向 CMX/FiLM+cross-attention。

## 6. 统一快速筛选协议

所有候选都只改 AMCM 内部，使用同一个 25-event manifest：

- failure: 20 条低 AUC、持续低 IoU 序列；
- control: 5 条高 AUC、无持续低 IoU 序列；
- 固定 template、`state_before`、crop、M1 checkpoint；
- 先做 disabled parity、shape、单 batch backward 和保存/加载；
- 再做无训练 forward；最后做 3 epoch AMCM-only quick train；
- 记录 GT/background feature distance、residual target/background ratio、tracking-loss directional derivative、fused output change、GT rank、peak margin 和 fixed-state IoU。

建议的晋级门槛是：困难组至少一半事件 GT-region directional loss 不再为正；平均 residual target/background ratio `>1.05`；GT rank 改善事件不少于 `8/20`；control 组不能出现系统性退化。单项涨点不满足晋级。

## 7. 文献与证据边界

以上论文均来自非单目标跟踪任务，但并不等于已经证明适合 M1。CMX、SeAFusion、CDDFuse、DIDFuse 和 SwinFusion 的监督与数据分布和 M1 不同；FiLM、GMU、CBAM、SKNet、Dynamic Conv、Focal Modulation 是通用模块范式，不提供 M1 所需的目标/背景标签。实现前应逐篇核对论文公式、官方代码和许可证；本报告是迁移筛选，不是官方复现结论。

## Decision

当前不建议随机继续替换模块。下一实验锚点应是 **CMX-style rectification 的最小双流接口版本**；FiLM-style template conditioning 和 MMTM-style reliability 作为两个单变量对照。其余候选进入 watchlist，只有在 25-event 筛选显示明确瓶颈证据后才启动。

## 8. 快速筛选通道验证

已运行 `tools/select_m1_precision_screen_sequences_20261020.py --count 20 --controls 5`。结果为 25 个事件、25 条唯一序列，其中困难组 20 条、控制组 5 条。所有事件 GT 有效；GT crop 覆盖率最低 `0.875`、均值 `0.995`；每条 trace 的 SHA256 均与 manifest 一致。困难组平均 M1 AUC 为 `7.57`，控制组为 `89.31`。

该结果验证了筛选清单的完整性和高反差属性，但没有测试上述 14 个模块本身。只有在某个候选完成 AMCM 内部实现后，才能用这个清单做同状态 forward、梯度方向和 3-epoch quick train 对照。
