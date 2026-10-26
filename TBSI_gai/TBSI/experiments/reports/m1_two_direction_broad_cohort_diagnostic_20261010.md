# M1 两个候选瓶颈的 96 序列覆盖诊断（2026-10-10）

## 结论摘要

用预冻结、结果盲选的 96 条 LasHeR train-side 序列继续诊断后，方向收敛为两个：

1. **首选：CENTER 联合定位读出**，具体是在线 score-cell 选择与同一位置 size/offset 框回归之间的条件不一致。GT-cell 离线上界在 44 个独立 onset 中让 31 个 IoU 上升、中位 IoU `0.371→0.484`；但 GT-cell 下 size L1 中位仍 `0.805→0.856` token，说明不是只改 score 峰就能解决。
2. **次选但未晋级：特定场景的融合证据损害**。失配起点中单路目标 probe 命中但 fused 未命中为 `7/44`，两路均命中而 fused 失手仅 `2/44`；起点时两路和 fused 命中率都明显低于匹配正常帧。它可能存在于局部子群，但不支持“融合普遍抹掉可用目标证据”，暂不够支撑通用可靠性门控或局部 cross-attention。

**“局部建模是 M1 的普遍主瓶颈”目前更像伪命题/过度概括。**较明确的场景风险是高 GT 相对运动：最高运动四分位 onset `15/24`，最低四分位 `7/24`；目标中心选择与回归读出也在多类场景出现。当前证据支持先诊断 CENTER 与运动下的定位一致性，不支持直接把模块改造目标定为局部融合。小目标和低亮度代理有较温和的 onset 富集；TIR 局部对比代理不呈单调关系。以上均是训练集 proxy 分层，不是官方属性因果分析。

## 实验合同与有效性边界

- Parent 为受保护 M1 `TBSI_E_MHSA_FULL_4090D_seed42` epoch 15。运行前/后固定 checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`；YAML SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1`。
- 清单从 300 条未被既有 M1 train-side 诊断/dev/eval/test 名单使用的合格训练序列中，确定性选 96 条；目标面积、GT 运动、RGB 目标亮度、TIR 目标-邻域对比的四分位边际各为 `24/24/24/24`。名单 SHA256 `b27b6a0426cc8e5c05a4cfe1ddf86b392768fe5c7b386f72c74622b03e5bbf3b`。没有用模型结果挑序列。
- 全程闭环 33,624 跟踪帧；冻结 onset 规则为 crop 覆盖≥99.9%、连续 3 帧 IoU<0.5、每序列至多一个事件。得到 44 条 onset、40 条 pre-onset、42 条 matched-good、52 条没有合格 onset 的 good-only 序列；另外每条序列一帧参考帧。stage 共 274 帧，重前向预测框最大差 `0.0000687 px`。
- 固定状态分析对 score raw/Hann/GT-cell 共用同一 forward 的 size/offset map；GT-cell 仅离线反事实。特征 probe 为三折序列留出，分别对 RGB/TIR/concat/fused 拟合 readout。
- Probe 的三折只留出了线性 readout 拟合序列；M1 checkpoint 训练数据包含 LasHeR train，因此这些不是 backbone 意义上的未见序列泛化样本。
- **环境偏差：**实际运行 Python `3.8.20`、PyTorch `2.0.0+cu118`、CUDA `11.8`，不是 M1 登记的 PyTorch `1.9.0+cu111`。当前工作树闭环 replay 也不与受保护归档逐帧等价。因此该结果是固定 checkpoint 的机制诊断，不是严格复现、不能直接替代归档 M1 指标。
- 1× NVIDIA GeForce RTX 4090 D 24 GiB；闭环 trace 约 21.8 分钟（GPU elapsed 约 `0.364 h`）；固定状态重前向 47.37 秒；probe CPU 约 20 秒。无训练、无测试集方法选择、无 checkpoint 写入。FPS/AUC/PR/NPR/OP50/OP75 不适用。
- 操作恢复记录：cohort 生成初次遇到无效 GT 行导致的对齐索引错误，修正为按原始帧索引读取后重建清单；stage 初次启动使用了预建空目录，按脚本合同改用全新输出路径；probe 初次启动时 NPZ 尚在压缩写入，ZIP 完整性检查通过后重跑。未修改保护 checkpoint/config，最终产物均通过校验。

## 场景覆盖与 onset 富集

训练 split 没有在本仓库资产中找到可核实的官方逐序列属性标注，所以按四个可复算代理量做覆盖。每格仅 24 条序列，事件比例是描述统计，不能据此宣称属性显著或因果。

| 代理维度 | Q1 onset | Q2 onset | Q3 onset | Q4 onset | 观察 |
|---|---:|---:|---:|---:|---|
| 相对目标面积（Q1 最小） | 14/24 (58%) | 10/24 (42%) | 11/24 (46%) | 9/24 (38%) | 小目标组较多，但非严格单调 |
| GT 相对运动 P90 | 7/24 (29%) | 9/24 (38%) | 13/24 (54%) | 15/24 (63%) | 最清晰的递增风险富集 |
| RGB 目标亮度（Q1 最暗） | 13/24 (54%) | 13/24 (54%) | 10/24 (42%) | 8/24 (33%) | 暗目标代理有温和趋势，非官方 LI 属性 |
| TIR 目标-邻域对比 | 13/24 (54%) | 12/24 (50%) | 7/24 (29%) | 12/24 (50%) | 不单调，不支持简单“热对比越低越坏” |

这组结果把场景描述收敛到**快速运动下的搜索中心失配/闭环初始化误差风险**，而不是笼统背景复杂或某一模态整体不可靠。目标面积和暗光可能是加重因素，但当前分层样本不足以拆解它们与运动、遮挡、序列难度的交互。

## 方向 1：CENTER 选格与 size/offset 条件读出

44 个 crop 内严重起点的固定状态结果：

| 量 | 在线 Hann peak | 指定 GT cell（oracle） |
|---|---:|---:|
| IoU 中位数 | 0.371 | 0.484 |
| 解码中心距 GT 中位数 | 1.333 token | 0.822 token |
| size L1 中位数 | 0.805 token | 0.856 token |

GT-cell oracle 让 `31/44` 条 IoU 有任意提升、`22/44` 提升超过 0.1，`1/44` 下降超过 0.1。说明当前 argmax 所选格对许多事件确实重要；但 oracle 后 IoU 中位仍低于 0.5，且 size L1 未下降。这是两个读出误差并存的证据：

- **选格/峰竞争：**Hann peak 与 GT-cell 解码中心距离中位差约 `0.51 token`，在 31 条事件上 oracle IoU 更高。
- **条件回归残差：**将格点直接指定为 GT 后，size 误差没有改善；GT 格点也不保证框重叠合格。因此仅做 score-only Strip、peak ranking 或 Hann-aware 监督都没有覆盖完整错误接口。

raw→Hann 在 `12/44` 起点换峰；GT rank 中位 `8→5.5`，peak 距离 `2.11→1.64 token`。这批 broad cohort 仍不显示 Hann 系统性伤害，故不恢复为独立第三方向。

按 proxy quartile 看，GT-cell 中心距离通常降低，但实际收益大小波动。最小目标与最高运动组都有较大的中心残差；GT-cell 的中位尺寸 L1 在多个分层仍不降。每格 onset 仅 7–15 条，分层数字用于描述覆盖，不用来挑方法或设阈值。

## 方向 2：融合是否压低可用目标证据

留序列 probe 的目标区域是否成为最高分区域：

| 状态 | RGB | TIR | Concat（融合前） | Fused |
|---|---:|---:|---:|---:|
| matched-good，42 序列 | 31/42 | 33/42 | 36/42 | 34/42 |
| pre-onset，40 序列 | 28/40 | 26/40 | 28/40 | 29/40 |
| severe onset，44 序列 | 16/44 | 19/44 | 18/44 | 19/44 |

从 matched-good 到 severe onset，RGB/TIR 与 fused 均有可读性下降；fused severe target-top 与 TIR 持平，略高于 RGB，concat 低 1 条。44 个 onset 中，`7` 个至少一路 probe 命中目标而 fused 没命中；只有 `2` 个是两路单独都命中而 fused 同时失手；另 `3` 个 fused 命中、两路都没命中。这个分布不支持跨多数场景的统一融合污染。TIR 低对比 Q1 出现 4/13 个“单路命中/fused miss”，可作为子群观察，但样本小、未校准、没有在线模态可靠性标签。

注意：四种特征分别拟合独立线性 probe，通道维度、拟合边界和分数标度不同。这里比较的是同一事件上的 target-top 二值模式，不比较 margin 大小；即使如此，它也只能作为可解码性证据，不能证明融合的因果增益或损害，更不能等价于 CENTER head 的响应。

## 最终方向排序

1. **优先继续：CENTER 联合定位读出与运动条件下的峰—框一致性。**下一步应在不改模型的情况下，拆分目标邻域 top-1/top-k margin、score 选格正确性、offset residual、size residual；在高运动和非高运动事件上按序列配对报告。若后续进入方法设计，应考虑让“选格决策”与“框回归监督”对齐的单一机制，不要再上 score-only 局部块。
2. **保留为条件性次方向：局部 RGB/TIR 可靠性冲突。**只对新队列里的候选冲突事件做定点人工可见性/配准核查和等容量 probe 复核；当前 7/44 不足以训练通用门控。若可用单路证据在更多独立事件中稳定出现且融合后消失，才设计受模板目标条件约束的局部路由；否则归档此方向。

**当前不做：**不训练新模块、不加 loss、不继续筛第三个 Hann 方向、不用 test-244 反复挑方案。也不把这批结果称为“证明局部建模无效”；它说明尚无证据证明普遍缺局部，而已出现更直接的 CENTER 联合读出与高运动风险信号。

## 可复现证据

- 清单与代理分层：[`manifest.json`](../diagnostics/m1_two_direction_cohort_v1/manifest.json)、[`sequence_list.txt`](../diagnostics/m1_two_direction_cohort_v1/sequence_list.txt)
- 原始闭环与采样：`output/experiments/M1_TWO_DIRECTION_DIAGNOSTIC_20261010/trace_run/`、`stage_capture/`
- Probe：`output/experiments/M1_TWO_DIRECTION_DIAGNOSTIC_20261010/probe_v2/`
- 汇总：`output/experiments/M1_TWO_DIRECTION_DIAGNOSTIC_20261010/analysis/{summary.json,per_event.csv,proxy_strata.csv}`
- 源码快照 SHA256 `e956540509b8d89bd07c129c1407785486f8ebc396de99c3f72a29ca2ae08e4d`
- Cohort builder SHA256 `cc64820bcb18638501472d32cb5016de1badca2cf739045c8d071a2b44808add`；analysis SHA256 `1dd128b1f55be2a814c3261e67acec9639a0cf5c918575829427e3c846883c49`
