# M1 与 Adaptive AMCM 全量逐序列诊断

> 2026-10-21 provenance 更正：以下逐序列统计描述的是旧保存轨迹，不能解释为 adaptive 算法在匹配 FP32 口径下的因果退化。后续固定首帧精度复算定位到 adaptive AMCM entropy 的 FP16 NaN，旧结果的精度/代码运行合同不统一或无法核验。详见 `m1_amcm_adaptive_precision_contract_audit_20261021.md`；独立 FP32 full-244 复核运行中。

## 目的与对照

本分析按官方 full-244 结果逐序列配对受保护 M1 与 `M1_AMCM_DR_ADAPTIVE_NOSUP_QUICK15_3EP_20261021`。M1 使用受保护 epoch15 checkpoint（runid `1030`）；候选使用从 M1 初始化后只训练 AMCM 3 epoch 的 checkpoint。评测均为 LasHeR testingset 244 条序列，读取官方 evaluator cache，并与预测框及 GT 逐帧核对。完整逐序列表见 `output/experiments/M1_AMCM_ADAPTIVE_FULL244_DIAG_20261021/per_sequence_m1_vs_adaptive.csv`。

这不是匹配训练进度的算子消融：候选系统包含训练后的 AMCM，而 M1 主干为 epoch15；本差异不能仅归因于 entropy 混合公式。配对 bootstrap 以序列为单位，不代表多 seed 置信区间。

## 汇总结果

| 指标 | M1 | Adaptive | 差值 |
|---|---:|---:|---:|
| AUC / Success | 56.208 | 43.645 | -12.562 |
| Precision | 70.207 | 54.136 | -16.071 |
| Normalized Precision | 66.198 | 51.334 | -14.864 |
| OP50 | 68.334 | 52.708 | -15.626 |
| OP75 | 47.135 | 36.846 | -10.289 |
| Average overlap | 57.126 | 44.317 | -12.808 |

244 条序列中 adaptive AUC 胜 95、负 147、平 2；中位差为 -0.118 个百分点。按序列 bootstrap 的均值差 95% 区间为 [-15.926, -9.367] 个百分点。48 条序列下降超过 20 点，36 条下降超过 50 点；仅 8 条提升超过 5 点。

## 逐序列表型

| 序列 | M1 AUC | Adaptive AUC | 差值 | Adaptive 首次更新 IoU |
|---|---:|---:|---:|---:|
| `bikeboyturntimes` | 85.09 | 0.39 | -84.70 | 0.000 |
| `blackboy` | 83.42 | 0.22 | -83.20 | 0.000 |
| `bikeboy` | 82.68 | 0.84 | -81.84 | 0.000 |
| `1blackteacher` | 81.95 | 0.11 | -81.84 | 0.000 |
| `bikeboywithumbrella` | 82.71 | 1.10 | -81.60 | 0.000 |
| `2runseven` | 81.43 | 0.35 | -81.08 | 0.000 |
| `11leftboy` | 81.42 | 0.63 | -80.80 | 0.000 |
| `leftfarboycomingpicktheball` | 19.49 | 61.70 | +42.21 | 0.792 |
| `runningcameragirl` | 35.16 | 58.07 | +22.91 | 0.953 |
| `whiteskirtgirlcomingfromgoal` | 29.17 | 42.85 | +13.67 | 0.798 |
| `boy`sheadingreycol` | 4.92 | 18.43 | +13.51 | 0.889 |
| `girlrightthewautress` | 24.71 | 36.98 | +12.28 | 0.918 |

首帧指初始化框之后第一次 tracker 更新（预测文件第 2 个框）。在 48 条 AUC 下降超过 20 点的序列中，M1 首次更新 IoU 均值为 0.860，adaptive 为 0.032；前五次更新分别为 0.835 和 0.030。该组 adaptive 的框在约 30.3% 更新帧中宽或高不超过 10 像素，约 55.9% 更新帧贴近左/上边界，约 96.8% 更新帧 IoU 小于 0.05。

其余 196 条序列的 AUC 均值为 M1 54.09、adaptive 53.74，首次更新 IoU 均值分别为 0.840 和 0.797。故总均值退化主要由一组立即失锁的序列拉低，同时候选整体尚未显示稳定正收益。小框和边界框是预测轨迹的崩溃表型，不能单凭它们判定某个内部算子是首因。

## 原因判断

**已确认的现象**：全序列闭环中，很多严重负例在初始化后的第一次预测就失去目标，后续框持续贴边或缩成极小框。这支持“首次定位失配被状态/crop 回灌放大”的链条，且不同序列发生率差异很大。少量序列有明显收益，说明模块并非完全没有有效信号。

**尚未定位的首因**：全量预测和 evaluator cache 没有记录候选内部的 AMCM residual、`tbsi_fuse_search` 前后 margin、CENTER target/background logits、size/offset 解码和 crop clipping 中间量。当前不能判定错误最早出现在 target matching、残差方向/幅值、TBSI、CENTER 解码、边界裁剪还是状态更新；也不能把失败定性为 entropy 本身造成。

优先待检验的机制是假设而非结论：

1. 当前熵来自 foreground attention 的分散程度，不等价于“本帧目标定位可信度”。背景重复纹理也可能产生集中注意，遮挡目标也可能产生高熵。
2. global/foreground 两种相似度的混合只有在两者量纲、校准和空间支持一致时才有语义。未校准的凸组合可能在少数模板/外观分布下改变 residual 的通道方向或空间位置。
3. AMCM 在 TBSI 与 CENTER 前注入两路 search residual。即便 residual 有目标相关性，若它改变分类峰与框回归分支的相对尺度/几何，第一次解码就可能输出错误中心或退化尺寸；短训后缺少可靠的定位置信回退会让该错误进入 tracker state。
4. 候选只训练 3 epoch 且采用无监督 AMCM；full-244 结果表明当前训练合同没有学到可迁移的可靠决策，但不能据此推出更多 epoch 一定能修复。

## 成熟跟踪思想对路线的启发

- ATOM（Danelljan et al., CVPR 2019）和 DiMP（Bhat et al., ICCV 2019）把目标模型判别能力与在线更新分开处理，并使用更新/预测可信度约束在线模型变化。对本模块的启发是先估计本帧“是否值得改变融合/模板”，再决定幅度；相似度或熵不能直接当更新置信度。
- KeepTrack（野外目标跟踪中的 distractor-aware tracking，ICCV 2021）显式维护干扰物并区分目标与相似候选。启发是可靠性应由 GT/模板目标与 hard distractor 的相对排序校准，而不只是 RGB-T 一致性或 foreground attention 熵。
- STARK（Bertinetto et al., ICCV 2021）及 OSTrack（Ye et al., ECCV 2022）将 template-search 交互与分类/框预测作为一个一致的读出过程。启发是融合分支必须对最终 target score 和 box quality 负责，不能以 residual 变化或 matching map 看起来更集中作为成功标准。

这些方法提供设计原则，不构成在 M1/LasHeR 上的效果保证；其骨架、训练目标和在线策略与本实验不同。

## 下一步：只做首更新固定状态因果诊断

暂不训练、不继续换 matching 模块。选 12 条独立 train-side 序列：6 条代表首更新崩溃表型，6 条有相近目标/遮挡但没有崩溃的对照；以帧为单位固定 template、crop、tracker state 和搜索图像。对 M1 parent 与 adaptive epoch3 checkpoint 同时记录：

1. AMCM 输入两路 feature norm，以及 global/foreground/adaptive target 与 hard-background 匹配分数；
2. 每模态 residual 的 norm、GT 区与 hard-background 区的方向/幅值比；
3. `tbsi_fuse_search` 前后 target/background feature margin；
4. CENTER GT cell rank、GT/top-1 score margin、分类峰坐标、size/offset 原始输出；
5. 解码后框、裁剪前后框、IoU 及状态更新前后 crop 参数。

在完全相同输入上分别做 AMCM bypass、adaptive residual 替换为零、residual 符号翻转三种可逆诊断，每次只动一个环节；另做非目标位置等幅替换作为空间对照。若 bypass 或置零能救回 CENTER 排名/尺寸，则 residual 是首个可干预原因；若 CENTER 排名正确但框退化，查解码/size/offset/边界处理；若首框合理而下一状态后才跑偏，再查状态回灌。该诊断需使用独立 train-side 序列，避免 full-test 上调模块。

## 模块改造建议

保留“global 与 foreground 证据兼得”的核心，不把当前 entropy 权重直接包装成贡献。把模块改成**目标条件、候选校准、有拒绝/回退能力的有界 residual 修正器**：

`模板目标/干扰物证据 + 双模态局部证据 → 目标/背景排序与定位质量 → 校准后的修正幅度 → bounded pre-fusion residual；低置信时输出严格 no-op。`

这个顺序借鉴 DiMP/ATOM 的置信控制、KeepTrack 的干扰物区分和 STARK/OSTrack 的融合到读出一致性。初版不增加多项 loss：先用现有跟踪定位目标训练信号，并只在已存在且可审计的 CENTER 分类响应上监督一个 localization-quality/target-vs-hard-negative 校准头；残差需零初始化、范数上限和无效候选 no-op。先通过固定帧可拟合性与 train-side 中间量合同，再做与 frozen-M1 的匹配进度训练比较。

**准入条件**：独立 train-side 崩溃样本上 GT 排名或定位质量校准明显改善；非目标位置对照不产生相同响应；正常控制序列 residual 接近零且第一次框不退化。满足后才进入短训和冻结 manifest 验证；若只改变 residual 大小却不改善 CENTER GT 排序/框质量，应停止旁路残差方向，转到候选/定位质量与 tracker update 决策接口。

## 可比性与复现限制

- M1 固定权重为 `TBSITrack_ep0015.pth.tar`，保护 SHA256 以顶层 `AGENTS.md` 为准。
- 候选配置 SHA256：`9c360c75f44c23762ae7a1b31306e168e5c39e16cd468b042b2d5936f8dbe967`。
- 候选 epoch3 checkpoint SHA256：`15e0bf9b2a489cecd24f1d4994a8b44a535aae2c689eca2ba3e2be237981a1b2`。
- 代码参考 HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`；分析脚本来自当前工作树。
- 环境：1× NVIDIA GeForce RTX 4090 D；Python 3.8.20、PyTorch 2.0.0+cu118、CUDA 11.8；FP32、M1 串行评测口径。
- 对照评测缓存、bbox、GT、配对 CSV 和汇总 JSON 均单独保留；本分析未修改 M1 权重、配置或输出。
