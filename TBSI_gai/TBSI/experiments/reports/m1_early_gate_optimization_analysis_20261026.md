# Early-conditioned MSHA gate 后续优化判断与分析计划（2026-10-26）

> **研究主线更正（2026-10-26）：** 本报告关于继续围绕 Early gate 做同状态归因、再优化其 gate 的建议，已被用户更正为非当前贡献目标。Early gate 只作为先导基线；当前研究目标是沿此前 v1 线路改变 MSHA 中间输入建模与模态交互机制。下方历史数据和测量结论仍是实验资产，但不能据此把下一步限定为增加 gate 输入或改 gate 幅值。新的 MSHA 线路审计与实验建议见 [m1_msha_internal_input_interaction_audit_20261026.md](m1_msha_internal_input_interaction_audit_20261026.md)。

## 结论

**当前证据不支持马上再加一个 gate 输入或 token reliability 算子。支持继续研究 Early-conditioned MSHA gate，但下一步应先做可重复性和同状态路径归因，再决定改算子。**

Early gate full15 的 LasHeR full-244 单次点估计相对受保护 M1 为 AUC `+0.349 pp`，但 sequence bootstrap 95% CI 为 `[-0.727,+1.481]`，逐序列 mean-IoU 为 120 胜、124 负，帧级 IoU 平均只增加 `0.000332`，低 IoU 帧净增加 345。收益集中在若干大涨序列，且中高 parent-AUC 分层有负向均值。该结果是值得保留的方向性正信号，不是稳定涨点证据。

更关键的归因边界是：full15 是从 SOT pretrained 权重出发的整网联合训练，资产中没有同一初始化、同一采样/增强流、同一优化过程的 gate-off full15 配对 control。此前 gate-only proxy full244 为负（AUC `-0.22 pp`），而 joint full15 是正点估计；二者训练预算与 parent 不同，不能据此断言增益来自 gate，也不能断言增益来自 backbone 联合适配。

同一 full15 candidate checkpoint 的 gate on/off 固定状态 A/B 将全部学习到的 checkpoint 张量保持相同，仅切换推理 gate。35 个 outcome-selected crop/state 中，28 个有效 GT-in-crop 状态的 Hann top-1 变化 `0/28`、GT rank 改善/退化 `0/28`，IoU 平均差 `-0.000008`；offset map 的平均绝对变化 `6.64e-4`，高于 CENTER map 的 `1.11e-5`。因此已测样本支持“gate 可微扰回归并经状态反馈放大”，不支持“gate 直接修复 CENTER 排序”。这批状态从既有涨跌序列中选择，不能外推为无偏总体结论。

两种后续 token reliability 公式均未获晋级：平衡 token reliability quick3 AUC `-8.87 pp`，只抑制低置信 token 的版本 `-1.72 pp`；两者都来自 7 序列回建开发清单和每臂 12 次 optimizer update，且 control 的 quick7 AUC 本身曾从 `56.99` 变成 `49.82`。之后更严格的同采样轨迹重复得到 AUC `45.51` 与 `56.65`，相差 `11.15 pp`，最大序列差出现在 `ab_bikeoccluded`（`60.08 pp`）。frame ID/worker/sample trace 一致，但增强后输入 tensor 尚未比较。由此 quick3 上候选的正负方向不能用于可靠排序。

## 问题—机制—证据链

| 链路节点 | 当前论文/算子主张 | 已有证据 | 判断 |
|---|---|---|---|
| 问题 | 原生 MSHA 的双向跨模态 Value 缺少基于当前目标证据的条件调制 | M1 crop 内失配、反馈漂移、出界与恢复不足是系统级观察；没有证据将其首因唯一归给 MSHA | 问题合理，因果首因仍未锁定 |
| Early gate 机制 | block-2 的 RGB/TIR 模板—搜索最大相似度及模板跨模态一致性，经 `3→32→2` MLP 预测 token 级双向 cross-Value 缩放；QK、self-message、CENTER 与状态更新不直接改变 | 585 参数/15 tensors 的 gate-only proxy 能训练；full15 gate 序列均值方向尺度约 RGB→TIR `1.0004–1.0006`、TIR→RGB `0.9894–0.9900`，随序列变化很小；同状态 gate on/off 未改变 CENTER rank | gate 已学习并执行，但其状态适应主要表现为很弱的近全局方向偏置，当前证据未显示稳定 target-selective routing |
| 局部读出 | Value 改变应改善 target/background 响应与定位 | fixed-state 中 CENTER 峰序与有效 GT rank不变；offset map 比 CENTER map更敏感 | 信号主要在框回归分支可见，不能称作 CENTER 修复 |
| 闭环性能 | gate 减少失跟及恢复失败，形成正向全量结果 | full15 小幅正点估计、区间跨零，序列效应高度异质；门控同状态差异可经 crop/state 放大且方向不稳定 | 暂不支持稳定性能贡献；闭环放大是观察，不等于净收益 |
| 后续细化 | 增加局部 margin/token reliability 可解决 gate 缺少空间信息的问题 | target margin quick3 负向；direct margin fixed-state 改动 gate 后 CENTER rank `0` 次变化；balanced/low-confidence token reliability quick3 未晋级 | 已测试信号不足以支撑继续堆叠相近可靠度特征 |

## 算子判断

当前 Early gate 的核心信息瓶颈是 `max_j` 把 16×16 search 响应压为一个相似度标量，丢失峰位置、邻域竞争和候选空间结构；gate 还没有模板前景 mask，无法辨别高相似度来自目标还是背景。但已有 margin / token reliability 变体没有补足“响应是否对应目标”的识别能力，且短训对照自身不稳定。因此，现阶段不应直接把“保留空间维”或“再加可靠度”写成已证明的优化方向，只能作为下一轮机制分析要区分的假设。

另一个事实是，gate 在前置 template MSHA 的 Value 路径，主要可改变融合后的特征与回归输出，并不会直接改 CENTER 头。若论文问题仍写作“修复 CENTER 选峰”，现有链路对不上；更严谨的当前问题表述应为：**早期模板匹配条件能否以序列/目标相关方式调制跨模态 Value，并减少后续框回归和闭环状态偏移？** 这一表述仍需同状态有效性和匹配训练对照支持。

## 下一步实验分析计划

| 顺序 | 分析问题 | 做法与控制 | 关键读数 | 结果如何改变设计判断 |
|---|---|---|---|---|
| A. quick3 输入重复性审计 | 同一采样记录下，两次 control 的模板/搜索输入是否真的逐 tensor 相同？ | 两次相同 control 配置只做 DataLoader replay/训练前向输入记录；在增强、裁剪、归一化完成后，对 RGB/TIR template/search tensor 做稳定 hash，同时保留样本 key、worker、retry、RNG state 来源；不添加随机 API，不改变返回 tensor。先解决当前约 1.1 GB 磁盘空间不足再决定保存方案。 | 输入 hash 一致率；首个差异 key；RGB/TIR/模板/搜索分别差异；worker 与增强 RNG 来源。 | 不一致：先修复 sampler/augmentation RNG 并重做 control repeat。全部一致但 checkpoint 不同：单独检查 cuDNN benchmark/非确定性 CUDA 算子与 BN buffers。未完成前不依据 quick3 候选涨跌挑选算子。 |
| B. Early gate 同状态逐 token 归因 | 当前 gate 输出是否按目标前景/背景区分消息，并且这种差异是否传入 CENTER、size/offset？ | 使用已存 Early-gate full15 checkpoint；固定同一 frame、template、crop、tracker state，对 gate strength 0/1 做配对 replay。使用预先固定的 train-side calibration 序列，不按候选结果选序列；首帧框仅用于离线前景/背景分层，不进入 gate 输入。记录 3 个 TBSI 层、双方向、64 tokens。 | gate scale 的前景/背景分布及序列间方差；cross-Value delta；CENTER/Hann map与 target-hard-background rank；size/offset delta；同 crop IoU。区分 gate 直接效果与后续状态反馈。 | 若前景/背景分配无差异且 CENTER 不变、offset 改变，优化目标应转向抑制不稳定回归扰动，而非继续声称 target ranking。若 token 变化存在但不相关定位效用，应更换证据来源；若相同状态定位有效但闭环衰减，再研究状态反馈传导。 |
| C. full15 涨点来源核对 | Early gate full15 的点估计可能来自 gate 还是不同训练轨迹/联合适配？ | 只做现有配置、初始化、sample trace、模型初始化与 optimizer 日志的 provenance 对照；确认模型新参数初始化是否改变 RNG 消耗、训练样本顺序与 BN 更新。不得将两条不同 run 的 seed 相同称为严格配对。 | 配置/权重初始 hash、训练样本 trace、optimizer step数、BN buffer差异、checkpoint逐张量差异、日志损失与验证曲线。 | 能重建差异来源后，才决定是否需一个同轨迹 gate-off full15 control；无法重建则将 full15 视为独立方向性观察。 |
| D. 算子准入决策 | 是否有足够机制证据提出一个 Early-gate 后继算子？ | A–C 完成后，按一个主假设提出一个改变：①若缺少空间 target correspondence，先离线比较保留空间峰/局部竞争能否预测前景消息效用；②若直接 gate 主要扰动 offset，先比较有界/identity-centered modulation 的固定状态效果。两类不能同轮组合。 | held-out train-side sequence 效用预测、同 crop定位读数、运行稳定性及计算开销。 | 只有在同状态效用方向跨多个序列稳定且超过 control 重复噪声，才进入单变量 quick3；之后使用更大预注册 cohort匹配复核。否则保留 Early gate 为论文基线模块，收窄 claim，不继续扩大算子。 |

## 资产与合同

- 研究 parent：受保护 `TBSI_E_MHSA_FULL_4090D_seed42`。Early-gate full15 checkpoint：`e38d18fe50c2895e75e4b9392bf22d9323bb8e6ad31ccbe39ad8c038e3fc1811`。
- 关键已有记录：`m1_early_conditioned_msha_gate_proxy_20261021.md`、`m1_early_conditioned_gate_operator_sequence_analysis_20261023.md`、`m1_early_gate_candidate_checkpoint_ab_20261024.md`、`m1_early_gate_token_reliability_quick3_20261026.md`、`m1_early_gate_lowconf_suppress_quick3_20261026.md`、`m1_early_gate_control_repeatability_20261026.md`。
- 本记录是对已有资产的只读综合分析，没有新增训练、跟踪推理或模型前向；指标列为 `NA`，不构成新性能实验。
- 前一轮对话曾把 DCVR-v2 误置为当前主锚点；按用户本轮更正，DCVR-v2 不作为最终版本或当前执行主线。后续工作以 Early gate 及本报告的准入分析为准。
- 当前可用磁盘空间约 `1.1 GB`，小于既有 quick3 checkpoint 约 `1.3 GB`。训练/测试启动前需检查外部输出盘或按资产记录安全释放空间；不得删除校准结果、M1 资产或报告所引用的证据。
