# M1 前三候选方向的配对诊断（2026-10-10）

## 结论摘要

本次只读复核前三个候选：在线选峰/训练监督错位、CENTER 读出不足、局部 RGB/TIR 证据冲突。结论按证据强弱排序：

1. **CENTER 选格与框回归联合读出失配：支持继续诊断，当前证据最强。** GT 选格能明显改善不少事件，但正确格点上的 size/offset 仍有明显残差。下一步若继续，应把 score 选格与 box 回归当作联合定位问题，不再做 score-only Strip。
2. **训练原始热图与 Hann 在线选峰不一致：差异存在，但尚未证明它是主要错误源。** Hann 换峰在两个不同固定状态队列中总体偏有利；GT-cell 上界显示“选哪个位置”确实重要，但不能把改善归因于消除 Hann。
3. **局部 RGB/TIR 模态冲突：尚未证实，当前是弱证据方向。** 失配起点两路和融合表征本身都难以稳定读出目标；现有单路扰动也没有反复显示“一路有清晰目标证据、融合将其破坏”。暂不应据此做可靠性门控或局部 cross-attention。

因此，前三个方向里最有根据的不是“加局部融合”，而是先明确 **CENTER 的目标峰排序和 size/offset 解码为何同时失准**。这并不排除融合参与，但现有证据没有把首因定位到 RGB/TIR 融合层。

## 数据与口径

- Parent：受保护 M1 `TBSI_E_MHSA_FULL_4090D_seed42` epoch 15；配置 SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`，checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`。
- 使用 LasHeR train-side 当前工作树固定状态既有结果：72 序列、104 帧，其中 16 个匹配正常、16 个失配前驱、16 个严重起点和 56 个额外正常帧；逐帧配对扰动表来自 `M1_SEARCH_STAGE_TRAIN_DIAG_20261009/analysis_final/per_arm.csv`。
- 另用独立 GT-cell 队列：20 个 train-side 序列、每序列一个固定事件；以及同一 20 序列上的 leave-one-sequence-out hard-negative probe。来源为 `M1_STRIP_GT_CELL_COUNTERFACTUAL_20261010/oracle_final_hardneg_v2/`。
- 本次仅 CPU 标准库读取 CSV 并重算配对计数，无新 tracker 推理、训练或 checkpoint 写入。当前工作树 replay 与受保护归档预测不逐帧等价；两批 train-side 事件的挑选规则和队列不同，不合并成一个样本量，也不作 test-244 的方法选择。

## 方向一：raw score 与 Hann 在线选峰

训练用原始 CENTER 热图监督，在线通过 Hann 加窗后的热图做 argmax。差异是代码合同事实，但其是否造成错误要看换峰方向：

| 固定状态队列 | 状态 | raw→Hann 改峰 | 方向/结果 |
|---|---|---:|---|
| 既有 16 序列队列 | 匹配正常 | 0/16 | 不变 |
| 同上 | 失配前驱 | 7/16 | 既有分析中换峰均使中心更近（7/7） |
| 同上 | 严重起点 | 9/16 | 7 个更近、2 个更远 |
| 独立 GT-cell 队列 | 20 个失配事件 | 3/20 | 3 个均同时改善中心误差与 IoU；其余 17 个不换峰 |

GT-cell 队列的 raw / 实际 Hann / GT-cell 解码中心误差中位数为 `26.10 / 22.84 / 9.25 px`，IoU 中位数为 `0.278 / 0.391 / 0.520`。基于原始逐事件 CSV 严格配对，GT-cell 相对 Hann **中心误差改善 16/20、IoU 改善 16/20、两项同时严格改善 15/20**；14/20 的 GT-cell 解码 IoU 达到 0.5，但中心误差小于 5 px 的只有 3/20。

解释：目标位置选择确实能显著影响最终框，但 Hann 在这两批事件中没有表现为稳定的有害因素。原始热图训练与 Hann 决策之间存在分布/排序差异，不能直接推导为应增加 Hann-aware loss；更不能将 GT oracle 的收益视作部署效果。当前判定为 **“选格有贡献；Hann 为主因未证实”**。

## 方向二：CENTER 读出与框回归

GT-cell oracle 使用每臂同一张 size/offset 图，只替换离线选择的 score cell。在 20 个独立事件中，实际 Hann 框到 GT-cell 框的中位 IoU 增加 `0.193`，中心误差中位减少约 `13.59 px`；但正确格点后的中心误差仍为 `9.25 px`，仅 3/20 小于 5 px。这说明失败至少有两项：

- **峰选择误差**：把 cell 指到 GT 会让相当多框的中心和 IoU 改善。
- **格点条件下的回归残差**：即使指定 GT cell，size/offset 仍不能稳定恢复准确框，单独改 score 分支不够。

留序列 hard-negative probe 在 14/20 序列上有正 target-vs-hard-negative margin，中位 margin `+3.26`，但低于预先使用的稳定可读门槛 18/20；其最高响应距 GT 中心中位 3.11 格。该 probe 表示冻结融合特征有时含可解码目标证据，不代表现有 CENTER 非线性 head 能使用它，也不代表证据在全部失败帧稳定。

较大的 72 序列接口 probe 提供了另一层约束：16 个严重起点中，RGB/TIR/concat/fused 的 target-background margin 中位数为 `-15.43/-8.49/-18.22/-6.04`，GT 区域为 probe 最高响应的比例仅 `6.2%/12.5%/6.2%/6.2%`。这更像“表征和读出均受限”，而非仅仅 Hann argmax 写错。特征维度不同的 concat/fused margin 不能解释成融合的因果增益。

判定：**CENTER 联合定位接口是当前最有根据的下一诊断对象；只加 score ranking、只加 score Strip 或把它包装成融合改进，都不充分。**

## 方向三：局部 RGB/TIR 冲突

在 16 个严重起点的同 crop 特征干预中，目标区 RGB 特征衰减 50% 仅 1/16 翻 Hann 峰且仅 1/16 的 IoU 提升超过 0.1；TIR 目标区衰减 50% 为 0/16 翻峰、0/16 提升超过 0.1。轻度衰减基本没有稳定的峰竞争效果。

极强的 GT 目标区 RGB 模糊（90%）在严重起点 3/16 提高 IoU 超过 0.1，但失配前驱中 6/16 下降超过 0.1，56 个额外正常帧中 5/56 下降超过 0.1。它是使用 GT 定位区域的离线强干预，不是可部署质量门，也没有证明 RGB 是污染源。TIR 强模糊没有对应的稳定救回信号。

加上失配起点两路及 fused probe margin 均为负，目前**没有满足可靠性融合的关键证据模式**：“至少一路在多个独立序列上可读，而原融合稳定地把它变差”。更没有量化目标区域 RGB/TIR 的局部配准偏差。因此：

- 空间变化的模态可靠性：**未证实**；
- 同坐标局部交互导致的错位污染：**未测/未证实**；
- 当前不能因为 SA/TC/LI 等全量属性落后，就把原因归为模态冲突。属性是风险富集而非融合因果标签。

## 综合判定与下一步

这不是一个完全的伪问题：全量归档显示大量持续失败以 crop 内中心失配开始，训练侧反事实也显示正确格点可能恢复部分定位。但“训练 raw score 与 Hann 决策不一致”更像一个待拆解因素，不是已经确认的首因；中心格点选择与框回归误差并存，且 search 特征可读性在失配前已下降。相比之下，“局部 RGB/TIR 冲突是主瓶颈”目前更接近未经证实的解释。

建议后续策略：

1. 暂不启动新局部融合模块、可靠性门控或新 loss。
2. 在新的、按序列冻结的 train-side 队列上，做 CENTER 联合定位分解：raw/Hann/top-k/GT cell 各自用同一 size/offset map 解码，并统计目标格、困难负峰和回归残差；同步捕获两路及融合特征上的同维/同容量 probe。
3. 只有出现“单路稳定可读、融合输出重复下降”才继续可靠性融合；若两路/融合都不可读，查模板到 search 的信息形成；若 fused probe 可读而原生 CENTER 选错且 GT-cell 回归准确，再考虑排序/读出适配。
4. 若希望 MHSA 成为耦合来源，先验证模板 MHSA 表征在同一新序列上是否为困难负峰提供超出未 MHSA 模板表示的增量判别信息；不能只把 MHSA 输出接到一个已拥挤的目标引导门控上，并据此宣称方法创新。

## 可复现资产

- 分析入口：`tools/analyze_m1_three_direction_diagnostic_20261010.py`
- 机器结果：`output/experiments/M1_THREE_DIRECTION_DIAGNOSTIC_20261010/summary.json`
- 原始配对输入：`M1_SEARCH_STAGE_TRAIN_DIAG_20261009/analysis_final/per_arm.csv` SHA256 `b5a6eafe0e386d18bf003f719679fb860052dbd5b2d611ed88baaa714d9909c8`
- GT-cell 输入：`M1_STRIP_GT_CELL_COUNTERFACTUAL_20261010/oracle_final_hardneg_v2/frame_results.csv` SHA256 `f49d1b4358baadd624bd53366ff626ee42755674fa3ff5ec473e224d41f98564`
- hard-negative probe 输入：`.../oracle_final_hardneg_v2/probe/heldout_by_sequence.csv` SHA256 `87b3075cbc2884c90ef19e61d789b47fe9e4ff8391b9023c0b107a2515b63923`
- Git parent `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`；新脚本 SHA256 `5cce79c01ccf3965b8f94748b43abff6af3f73a327a11b0021e9c129debb883d`。
- 本次标准库 CPU 汇总，无 GPU 推理/训练，GPUHours `0`，FPS 和模型指标 `NA`。父模型环境仍为 1× NVIDIA GeForce RTX 4090 D 24 GiB；当前 replay 的来源环境与归档不等价，详见各源报告。
