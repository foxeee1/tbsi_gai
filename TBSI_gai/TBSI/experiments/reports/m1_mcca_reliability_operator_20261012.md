# M1 MCCA 可靠性门控算子实验

## 目的

针对原 MCCA 的主要结构风险进行单变量改造：原实现把跨模态消息无条件注入模板 MHSA 消息路径。本实验增加 query/message 独立 LayerNorm，并加入逐 token 的可靠性门控，令消息残差变为 `delta = MLP(LN(query), LN(message), |LN(query)-LN(message)|) * sigmoid(gate)`。`fc2` 仍零初始化，因此从 M1 加载时严格保持恒等输出。

本轮不启用 ranking loss，避免同时改变算子和监督目标。只训练新 MCCA adapter，M1 主干、原融合器和 CENTER head 冻结。

## 合同与检查

- 配置：`M1_MCCA_FROZEN_RELIABILITY_3EP_seed42`，seed 42，3 epochs，252 optimizer steps，FP32，worker 1。
- 训练数据：既有 RPP-v1 train manifest；评测为预注册 train-side held-out 96 序列。
- 训练参数：仅 6 个方向性 adapter 模块，包含归一化和门控，共 `2,088,582` 个可训练参数。
- 零输出 parity：`max abs diff = 0.0`。
- 6 个 adapter 的 `fc2` 梯度有限且非零，范围约 `0.00243–0.01259`。
- 单步更新后 42 个 adapter 张量发生变化；冻结参数变化 `0`，冻结 buffer 变化 `0`。
- checkpoint 验证：72 个新 adapter 张量、非 adapter 张量逐项保持一致。

## 96 序列配对结果

| 指标 | 冻结 M1 | 可靠性 MCCA | 差值 | 95% bootstrap CI |
|---|---:|---:|---:|---:|
| AUC | 65.6715 | 64.3425 | -1.3291 | [-3.1917, 0.0733] |
| PR | 79.1348 | 77.4870 | -1.6477 | [-3.8255, 0.0196] |
| NPR | 76.7483 | 75.2189 | -1.5294 | [-3.6910, 0.1194] |
| OP50 | 79.4736 | 77.9034 | -1.5703 | [-3.7204, 0.0663] |
| OP75 | 60.1086 | 58.7060 | -1.4025 | [-3.4445, 0.0827] |

AUC 逐序列 49 胜、47 负；候选 FPS `63.76`，冻结 M1 为 `73.25`，下降约 `13.0%`。

## 归因

这次改造排除了“无归一化和无门控导致的简单分布失配”这一种解释，但没有改善任务结果。门控参数和 adapter 参数都实际更新，说明它不是未训练或旁路问题。

更可能的原因是：

1. **门控输入不足以判断跨模态可靠性**。它只看模板 query 和 MHSA cross-message 的局部特征关系，没有显式看到搜索区域质量、RGB/TIR 响应冲突、运动状态或当前 CENTER 峰竞争。
2. **门控作用点过早**。它调节模板消息，最终错误却发生在 search score 的离散选格、offset/size 解码和闭环状态更新阶段。
3. **冻结 M1 下的监督仍是间接的**。标准 tracking loss 会推动 adapter 改变表示，但没有要求门控只在失配帧开启，也没有保护正常帧的 M1 输出。
4. **门控增加了计算和参数，却没有提供可验证的场景信号**。FPS 进一步下降到约 63.8，当前收益不足以覆盖代价。

## 决策

状态记为 `COMPLETE / RELIABILITY-GATE-NOT-SUPPORTED`。当前 MCCA 线的基本算子没有发现计算图错误，但“模板消息 + 特征内部门控”这个设计不足以解决 M1 的搜索定位失败。停止当前可靠性门控版本的继续训练和 full test，不继续调 gate hidden、学习率或门控初始化。

若未来重新开启 MCCA，必须把门控输入改为能观察搜索期证据的量，例如搜索两模态响应差异、模板-搜索相似度、运动状态和当前候选峰质量，并把门控放到搜索融合或 CENTER 前；否则继续扩大模板消息 adapter 的容量不会区分现有已知瓶颈。

证据：

- 配置：[M1_MCCA_FROZEN_RELIABILITY_3EP_seed42.yaml](../tbsi_track/M1_MCCA_FROZEN_RELIABILITY_3EP_seed42.yaml)
- checkpoint：`/autodl-fs/data/M1_MCCA_FROZEN_RELIABILITY_3EP_20261012/output/experiments/M1_MCCA_FROZEN_RELIABILITY_3EP_seed42/checkpoints/TBSITrack_ep0003.pth.tar`
- 配对结果：`output/experiments/M1_MCCA_FROZEN_RELIABILITY_3EP_20261012/metrics/paired.json`
- P0：`output/experiments/M1_MCCA_FROZEN_RELIABILITY_3EP_20261012/metrics/p0.json`
- 冻结状态校验：`output/experiments/M1_MCCA_FROZEN_RELIABILITY_3EP_20261012/metrics/frozen_state_verification.json`
