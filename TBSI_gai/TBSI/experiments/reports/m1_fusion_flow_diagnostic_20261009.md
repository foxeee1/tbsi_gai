# M1 模板 MHSA 与搜索融合信息流诊断（2026-10-09）

## 目的与边界

本诊断观察 M1 的模板联合 MHSA、head 前搜索特征融合及其定位输出之间的信息与梯度变化。它复用既有 LasHeR train-side trace，在冻结 checkpoint 上逐帧重放、计算定位损失敏感度；无 optimizer step、无参数更新、无训练，也未读取 full-244 测试序列。它用于提出机制假设，不构成模块改造或收益结论。

Parent 是受保护实验 `TBSI_E_MHSA_FULL_4090D_seed42`，seed 42、epoch 15。配置 SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`；checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`。两者均与保护记录一致。模型采集点为 TBSI location 3/6/9 的模板联合 MHSA 输入/输出、MHSA 双向跨模态 attention/message、search 输入及 `tbsi_fuse_search` 输出、backbone 输出和定位响应。

## 数据与工程核验

- 输入：预先保存的 3 条 LasHeR train trace：`abreastinnerboy`、`right2ndfarboytakinglight2left`、`biketurndark`；trace manifest SHA256 `72ea329132e02a0a7f6d173c58b4771b9595260eb61d8ff98a652615c072057b`。
- 1,374 个输入帧中，15 帧因 GT 宽/高非正被排除；1,359 帧用于梯度和相关性统计。被排除帧仅在 `right2ndfarboytakinglight2left`，帧号 390–404。
- FP32、eval 模式下重放。与冻结 M1 trace 的最大预测框绝对差为 `0.007744 px`（容差 `0.01 px`），最大 IoU 差 `0.000118`，最终 parity 检查通过。
- 逐帧表包含 217 列；全部数值字段均无缺失/非有限值。MHSA 各位置的 RGB/TIR 输入和输出梯度、head 前融合的 RGB/TIR 输入梯度及融合输出梯度，在 1,359/1,359 帧均为非零。该结果证明此损失路径在本诊断批次可反向传播，不证明模块在训练中有效更新或改善跟踪。
- 损失目标为 Hann 解码定位上的 `2*GIoU + 5*L1`，加 CenterNet focal score loss。梯度是固定权重对该逐帧监督目标的敏感度，不是训练更新方向或跨帧 BPTT 梯度。
- 运行环境：1× NVIDIA GeForce RTX 4090 D 24 GiB，Python 3.8.20、PyTorch 1.9.0+cu111、CUDA 11.1，FP32。成功运行记录没有保存可核验 GPU 时长，故登记为 `NA`。
- 脚本 `tools/m1_fusion_flow_diagnostic_20261009.py` SHA256 `413d3a74f79915a5ff766fec2022625a27cd8c38265a78513f45ca93c1661889`。完整命令：`python tools/m1_fusion_flow_diagnostic_20261009.py --output-dir /root/autodl-tmp/TBSI_gai/TBSI/output/experiments/M1_FUSION_FLOW_DIAG_20261009/full_train_trace_final`。

## 观察结果

| 范围 | 观察 | 与定位结果的关系（序列内 Spearman） |
|---|---|---|
| 序列整体 | 平均 trace IoU `0.5210`，逐帧损失均值 `2.5672`。三序列平均 IoU：`abreastinnerboy` 0.3494（n=474），`right2ndfarboytakinglight2left` 0.5833（n=651），`biketurndark` 0.6956（n=234）。 | 序列基线差别很大；逐帧样本存在强时间相关，合并相关容易混入序列/场景差异。 |
| 模板 MHSA 激活 | location 3/6/9 的模板残差范数相对输入范数约为 RGB/TIR `0.12/0.13`、`0.11/0.12`、`0.12/0.12`。两模态均通过联合 MHSA 更新；残差非零且量级低于输入。 | 仅刻画表征变化幅度，不能由小残差推出融合不足或过融合。 |
| location 9 attention | attention entropy 与 IoU 的序列中心化 Spearman `-0.510`，与中心误差 `+0.527`、损失 `+0.561`。 | 按每条序列去均值后合并的帧相关显示，高 entropy 与较差定位共变；需控制目标尺度、遮挡、状态与时序。 |
| location 9 跨模态消息 | TIR cross-message norm 与 IoU 的序列中心化 Spearman `+0.431`，但未中心化 pooled Spearman `+0.028`；RGB cross-message norm 对 IoU 的序列中心化相关 `-0.043`。 | 信号方向依模态/聚合方式不同；不能据此判断应增强哪一条消息或据此选注入点。 |
| MHSA 参数梯度 | location 9 QKV weight grad norm 与 IoU 的序列中心化 Spearman `-0.413`，与 loss `+0.431`。 | 误差较大的帧通常产生更大的敏感度；不是“梯度被阻断”证据，也不是训练时收敛或更新幅度。 |
| 定位响应 | GT response cell 分数与 IoU `+0.447`、中心误差 `-0.606`、loss `-0.494`；peak response 与 IoU `+0.394`、loss `-0.305`。 | 按序列中心化后，GT 位置响应较直接地跟随定位质量；它是 head 输出关联，并未定位到上游融合因果。 |
| head 前搜索融合梯度 | `tbsi_fuse_search` 两路搜索输入和输出梯度均在所有帧非零；两路输入梯度/梯度乘激活幅度总体相近。搜索输入梯度范数与 IoU 的序列中心化 Spearman：RGB `-0.263`、TIR `-0.273`。 | 两模态都有可导路径，误差帧更敏感；不能据梯度大小判定通道贡献、可靠性或应该替换当前融合算子。 |

`within_sequence_spearman` 的计算方式是先在每条序列内分别对 metric 和 outcome 去均值，再将所有帧合并后对中心化值取秩并计算 Spearman；它不是逐序列相关系数的平均。此处理不能消除连续帧自相关、闭环状态混杂或测量误差；未做多重比较校正。表中 1,359 是帧数而非独立样本数，效应量只作探索性描述。

## 判断

**已确认：** 当前冻结 M1 中，模板 MHSA 与 head 前搜索融合的相关张量均有非零激活和梯度路径；定位响应与逐帧跟踪质量有关。观测到 location 9 entropy、跨模态消息量、QKV 梯度敏感度和定位结果存在序列内关联。

**尚未确认：** 这些相关性不足以说明 M1 的主要性能限制来自模板 MHSA 的融合局限，也不足以说明应在 IGF 原论文位置注入 DBF。当前诊断没有做单模态反事实、融合接口干预或受控跟踪状态对照；三条 train trace 不能代表 full test，也不能支持因果结论。特别是跨模态消息相关在不同序列聚合口径下变化，说明单看 pooled 相关会得出不稳健的解释。

**结论状态：** `DIAGNOSTIC-COMPLETE / CAUSALITY-UNCONFIRMED`。暂不据此启动 DBF 训练或认定某一个接口是主瓶颈。

## 最值得做的下一步

在 train-side 固定 tracker 状态和相同 crop 上，预先选取多条独立序列中的持续低 IoU 事件及匹配正常窗口；分别采集模板 MHSA 前后与 `tbsi_fuse_search` 前后的目标/背景响应、RGB/TIR 单模态可用性和单步定位变化，并加入同状态的单模态屏蔽/轻扰动反事实。只保留聚合标量和少量代表帧。若目标确实位于 crop 且至少一种模态含有效目标信息，而某一接口后目标响应和定位稳定恶化，才针对那个接口提出一个独立 DBF 假设；若恶化在接口前已出现，则转查 crop/状态/定位头。该诊断仍是机制筛选，需后续同数据合同短训验证收益。

证据目录：`output/experiments/M1_FUSION_FLOW_DIAG_20261009/full_train_trace_final/{summary.json,per_frame.csv,metric_correlations.csv,excluded_frames.csv}`。早期 preflight / parity 失败运行留存在同级 `preflight_smoke2`、`preflight_smoke3` 和 `full_train_trace`，最终有效结果单独保留在 `full_train_trace_final`。
