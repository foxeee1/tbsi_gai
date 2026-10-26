# M1 新 train-side CENTER 读出定点诊断（2026-10-10）

## 决策摘要

本次使用预冻结、未参与先前 M1 模块/dev 选择的 24 条 LasHeR train-side 序列，按“GT 至少 99.9% 在当前 crop、连续 3 帧 M1 IoU<0.5”的规则抽取事件。共得到 14 个起点，低于 18 序列稳定性门槛，结论限定为**探索性机制证据**。

本轮不支持把问题概括为单一“加 Hann-aware loss”或“局部模态门控”：

- **中心选格有贡献，但新 cohort 中不是充分解释。** GT-cell 离线上界将严重起点中位 IoU 从 `0.413` 提至 `0.495`；10/14 有任何幅度改善，5/14 提升超过 0.1，5/14 达到 IoU 0.5。其余事件不少在预测 peak 已接近 GT 时仍有框重叠误差。
- **raw→Hann 不呈普遍有害。** 14 个起点有 5 个换峰；中位 GT 峰排名从 raw 的 12 降到 Hann 的 8，峰距中位从 `1.78` 降至 `1.45` token。换峰与选格错误确实共存，但总体方向更偏向 Hann 改善排序，而非 Hann 把峰系统性推离目标。
- **局部 RGB/TIR 冲突可能只存在于子集。** 4/14 序列出现至少一个单模态 target-vs-background probe 为正、融合 probe 为负（2 个 RGB-only、2 个 TIR-only 模式）；但在相同 cohort 中，模态 probe 之间不是校准概率/同容量因果比较，且目标区强模糊没有稳定救回。证据不足以训练可靠性门控。

更稳妥的当前结论是：**CENTER 的 score 选格、offset/size 回归、以及失配起点的目标证据不足可能并存；RGB/TIR 融合造成的局部损害是少数事件的候选解释，不是全局主因。**

## 固定实验合同

- Parent：受保护 M1 `TBSI_E_MHSA_FULL_4090D_seed42` epoch 15。YAML SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`；checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`，本轮启动前/结束后均复核匹配。
- 冻结名单 24 条，SHA256 `e1a976257d71969686c1dc0690596465d5e14d24366ecef567c58f8b129e65f6`。依据固定 SHA rank、150–800 帧长度和既有验证/评测名单、历史 M1 train trace/event 排除规则选取；无性能筛选。清单与排除源哈希在 `experiments/diagnostics/m1_center_readout_fresh_v1/manifest.json`。
- 24 条闭环共 7,824 帧（首帧除外）。14/24 条满足预定义 onset；每事件包含更早匹配正常帧、前驱、起点及固定参考帧，共 100 个 feature frames。
- 固定状态重前向最大框差 `0.0000687 px`，通过 `0.01 px` parity。所有 score cell（raw、Hann、GT）使用同一 forward 的 size/offset map 解码；GT cell 和 GT 区域扰动只用于离线诊断。
- 运行：1× NVIDIA GeForce RTX 4090 D 24 GiB，Python 3.8.20、PyTorch 1.9.0+cu111、CUDA 11.1、FP32。trace 253.24 s，stage capture 23.78 s，probe 在 CPU 完成；没有训练/checkpoint 写入。GPU elapsed 约 0.077 h（进程墙钟，不是 GPU profiler 计时），FPS/AUC/PR/NPR/OP50/OP75 不适用。
- 当前工作树重放与受保护归档逐帧不等价；M1 模型曾在这些 train-side 视频上训练，本名单只保证此前未用于本方向的方案/开发选择，不是模型从未见过的外部测试集。

## 方向 1：raw score 与在线 Hann 选择

严重起点的 raw→Hann 换峰为 5/14。raw 与 Hann 的 GT cell rank 中位数分别为 `12`、`8`，peak-to-GT 距离中位数分别为 `1.78`、`1.45` 个 token；rank 前 3 的事件两者都是 4/14。即使训练监督用 raw、推理用 Hann 的合同差异真实存在，这批结果没有显示 Hann 为主要有害来源。当前允许的说法是：**score 峰竞争参与错误，但 raw/Hann 对齐修正的收益未证实；继续改 loss 前应先看困难负峰和 size/offset。**

## 方向 2：选格与 size/offset 联合读出

| 指标（14 个 onset） | 实际 Hann peak | GT-selected cell |
|---|---:|---:|
| IoU 中位数 | 0.413 | 0.495 |
| 选格后 IoU 提升 | - | 任意提升 10/14；提升 >0.1 为 5/14 |
| IoU≥0.5 | - | 5/14 |
| 解码中心距 GT 的 token 中位数 | 1.183 | 0.836 |
| size L1 误差 token 中位数 | 0.795 | 0.854 |

GT-cell 指定后中心距减少，但 size L1 中位数没有同步变小；GT-cell IoU 仍低于 0.5 的 9/14 事件说明正确 cell 也不保证正确框。`whiteof2boys` 是峰选格可明显救回的例子（IoU `0.388→0.899`）；`blackbag` 则 GT-cell IoU 反而由 `0.316→0.249`，表明同一回归图在 GT cell 的读出也可能不合适。建议把下一问题写成**score cell 与 offset/size 的条件化是否一致**，而非单独提升 score 峰或泛化加局部卷积。

## 方向 3：模态证据与融合损害

严重起点的 3-fold 序列留出线性 probe 汇总如下。target/background margin 为各阶段独立拟合的线性 readout 分数；concat 与 fused 的维数不同，数值幅度不可直接作因果比较。

| 阶段 | GT 区域成为最高分比例 | target-background margin 中位 | probe 峰距 GT 中位（token） |
|---|---:|---:|---:|
| RGB | 6/14 | -6.58 | 1.84 |
| TIR | 6/14 | -0.93 | 2.09 |
| 两路 concat（融合前） | 5/14 | -5.63 | 1.80 |
| 融合后 fused | 6/14 | -2.83 | 1.87 |

相对 matched-good 帧，四个阶段都出现明显可读性降低。fused probe 比 concat 的 margin 中位较少负，但维度不同，不构成融合改善证据；也没有总体融合后坍塌。14 个事件中有 4 个“单路 margin 正、fused margin 负”的案例：

| 序列 | 单路正 margin | 该路 margin | 另一路 margin | fused margin | 实际→GT-cell IoU |
|---|---|---:|---:|---:|---:|
| `whiteof2boys` | RGB | +4.59 | TIR -3.37 | -12.74 | 0.388→0.899 |
| `2girlinrain` | TIR | +5.30 | RGB -60.70 | -27.16 | 0.288→0.318 |
| `basketman` | TIR | +1.03 | RGB -7.84 | -16.82 | 0.369→0.614 |
| `blackbag` | RGB | +16.66 | TIR -15.54 | -4.67 | 0.316→0.249 |

这提供了少数“可以进一步审查模态冲突”的序列证据，但每种方向只有两个事件，probe 分数未经跨阶段校准；`blackbag` 的 GT-cell 框还更差。目标区 RGB 强模糊在严重起点 0/14 帧提高 IoU 超过 0.1，3/14 下降超过 0.1；TIR 强模糊也未产生显著稳定救回。轻度 RGB 特征衰减有 1/14 峰翻转、无 >0.1 IoU 获益；TIR 特征衰减有 2/14 翻转、1/14 提升 >0.1。故这些反事实不能直接映射成在线门控策略。

## 方向判断与下一步

这次 fresh 队列没有推翻前三方向，但把优先级重新定细：

1. **优先：CENTER 联合定位读出。**另外 14 个新事件中 oracle 的收益中等且有例外；继续诊断 score rank、offset residual、size error 的事件级对应，并确认它是否与 SA/TC 或小目标场景相关。暂不训练 score-only 模块。
2. **其次：局部模态冲突作为子群假设。**已有 4/14 单模态正而 fused 负的信号，足以在新的训练侧事件上做人工可见性/局部对应审阅或同容量 probe 复核；还不足以做门控短训。若后续仍只有少数、且强扰动没有可预测方向，停止此线。
3. **降低：Hann 不一致修正。**本队列中 Hann 的峰距和排名总体改善，另两批小事件也未显示稳定伤害；除非新数据证明困难负峰被 Hann 系统性放大，否则不做 Hann-aware loss。

没有证据表明“缺局部建模”本身是 M1 的普遍问题。只有先证明 RGB/TIR 两路各自具有可用目标证据、且当前融合在相同容量/相同条件下重复压低它，局部可靠性路由才值得进入训练。此结果也还不足以宣称 MHSA 耦合机制有新颖性；若保留 MHSA 作为来源，下一诊断应比较其模板目标表示对这 4 个冲突模式的增量，而不是只增加一个目标引导门。

## 可复现产物

- 冻结 cohort：`experiments/diagnostics/m1_center_readout_fresh_v1/{manifest.json,sequence_list.txt}`。
- 闭环 trace：`output/experiments/M1_CENTER_READOUT_FRESH_DIAG_20261010/trace/`；状态 `TRACE-COMPLETE`，24/24。
- 固定状态采样：`output/experiments/M1_CENTER_READOUT_FRESH_DIAG_20261010/stage/`；100 帧、14 onset、parity `0.0000687 px`。
- probe 与最终汇总：`output/experiments/M1_CENTER_READOUT_FRESH_DIAG_20261010/probe/`、`.../analysis/summary.json`、`.../analysis/per_event.csv`。
- 源码快照：`output/experiments/M1_CENTER_READOUT_FRESH_DIAG_20261010/provenance/source_snapshot.tar.gz` SHA256 `12af47d565cdfc02fd134397fecaf76af0ebd3e0af6e291d25a4a54a0b793d9d`。
- 新建名单工具 `tools/build_m1_center_readout_fresh_manifest_20261010.py`；新增事件规则聚合工具 `tools/analyze_m1_center_readout_fresh_20261010.py`。trace 与 stage 工具保留旧默认行为，本次通过显式名单和 `onset-3-iou-0.5` 参数运行。
