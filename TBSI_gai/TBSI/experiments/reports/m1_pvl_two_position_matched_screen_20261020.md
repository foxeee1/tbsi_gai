# M1 PVL 双位置匹配筛选结果（2026-10-20）

## 结论

结论分类：**暂不支持晋级**。在同 seed、相同公开 SOT 初始化、相同训练进度和固定 13 条已观察 LasHeR test 序列下，模板 loc6 与搜索融合 Neck 两个 PVL 候选都低于同架构 control。loc6 的 AUC 差为 `-4.18`，Neck 为 `-8.26`；两者均为逐序列 5 胜、8 负。该证据足以停止当前两个实现进入长训，但不能推出 PVL 方法在其他监督、参数化或实验条件下普遍无效。

本实验的 control 与候选均从公开 `TBSITrack_SOT.pth.tar` 开始训练 5 epoch，**不是 M1 epoch15 checkpoint 的三路直接对比**。清单中的“原 M1 低 AUC”仅表示这些序列按历史 M1 表现选出；当前 control 在其中的表现明显不同。因此这里的差值只解释为同进度候选相对 control 的差值，不能说候选把 M1 原性能降低了相同幅度。该名单是回看后选取的筛选集，不是独立验证集。

## 运行与实现合同

| 组别 | 位置 | AUC | PR | NPR | OP50 | OP75 | FPS | 相对 control AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 同架构 control | 关闭 PVL | 52.33 | 63.86 | 62.07 | 62.09 | 45.57 | 60.06 | -- |
| PVL 模板 | TBSI loc6 模板 token，joint template MHSA 前 | 48.15 | 57.81 | 57.01 | 56.63 | 42.36 | 55.62 | -4.18 |
| PVL 搜索 Neck | 搜索特征进入 `tbsi_fuse_search` 前 | 44.07 | 52.31 | 51.30 | 50.92 | 39.17 | 52.73 | -8.26 |

三组均采用 seed 42、LasHeR train `512 samples/epoch × 5 epochs`、batch 2、FP32、worker 4、同一训练策略，并在同一 manifest 的 13 条序列测试：简单 3 条、中等 3 条、历史 M1 低 AUC 3 条、额外困难 4 条。候选和 control 有 267 个共享 state tensor 逐元素一致。M1 受保护配置与 epoch15 checkpoint 哈希分别仍为 `fab8e818...16b4e1e` 与 `1302de3b...f54e04ef`，与资产登记值一致。

初次 loc6 运行因 PVL 初始化消耗共享 RNG 而改变其他随机层初始化，已单独保留并登记为无效归因；本报告只使用隔离模块初始化后的 matched 运行。随机输入 forward、输入输出 shape/channel 检查、零门控 identity parity、反向梯度、冻结状态检查及确定性 eval 检查已通过。Dataset、划分、Loss、Optimizer 和训练策略未改。

模块 shape 合同为两路输入各 `[B,N,768]`、两路输出各 `[B,N,768]`；Neck 位置的 `N=256` token 在适配后还原为 `[B,768,16,16]` 接回原搜索融合。模块新增参数 `247,041`，两个位置相同。Forward 验证通过；通道不匹配、RGB/TIR shape 不同或 device/dtype 不同会显式报错。

## 分组与逐序列结果

| 序列组 | 数量 | control AUC | loc6 AUC（差值） | Neck AUC（差值） |
|---|---:|---:|---:|---:|
| 简单（按 M1 AUC） | 3 | 90.26 | 90.12（-0.14） | 90.75（+0.49） |
| 中等（接近 AUC 60） | 3 | 58.07 | 56.30（-1.77） | 61.66（+3.58） |
| 历史 M1 低 AUC | 3 | 60.91 | 44.82（-16.09） | 22.58（-38.33） |
| 额外困难 | 4 | 13.13 | 13.05（-0.08） | 11.98（-1.16） |

分组差值是小样本序列均值，不能当作总体估计。主要退化集中在历史低 AUC 组：`ab_bikeoccluded` 两候选相对 control 均约低 48 AUC；Neck 在 `3rdfatboy` 低 24.27、`leftfarboycomingpicktheball` 低 42.76；loc6 这两条接近 control。相反，Neck 在 `bluegirlbiketurn` 高 11.54，loc6 在 `leftunderbasket` 高 0.41、`boyunder2baskets` 高 1.44。收益与灾难性退化并存，平均提升不稳健。

历史 M1 低 AUC 桶里 control AUC 已达 `60.91`，与历史 M1 被用于筛选的低分差异很大。这提示初始化、训练进度及闭环状态对该组结果影响显著，也说明不能把本轮表现当成 M1 的原始低 AUC 成因诊断。需查看逐序列表与历史 M1 指标分开解释。

## 能力、成本与归因

| 维度 | loc6 模板位置 | 搜索融合 Neck 位置 |
|---|---|---|
| 表达能力 | 双向低秩（adapter dim 64）的概率 cross-attention，试图让模板 token 按另一模态的不确定 key/value 接收消息；增加条件跨模态交互，但没有测得目标/背景选择性提升 | 同一双向概率交互直接作用于进入搜索融合的特征，空间搜索 token 更直接参与决策；没有证据表明其排序能力变好 |
| 新增参数 | 247,041（约总参数 0.224%） | 247,041（约总参数 0.224%） |
| 理论计算增量 | 16.78M MAC/sample | 79.69M MAC/sample |
| 实测速度 | 55.62 FPS，较 control 低约 7.4% | 52.73 FPS，较 control 低约 12.2% |
| 稳定性 | 13 条中 5 胜、8 负；跨序列结果不稳定，单 seed 无法评价 seed 稳定性 | 13 条中 5 胜、8 负；最大退化接近 48 AUC，风险更高，单 seed 无法评价 seed 稳定性 |
| 可解释性 | 插入点明确，但当前没有 PVL 专属的有效特征变化记录，无法把性能变化连到目标选择性消息 | 插入点紧邻搜索融合主路，机制上易于定位；同样缺少可用的 PVL 专属特征变化记录 |

MAC 为理论估算，FPS 是本轮固定小名单的实测吞吐，受单次运行与评测时序影响，不能代替受控 profiling。整体梯度日志记录了模型梯度流，但 `feature_changes.csv` 只有表头；不能据此声称已测得 PVL 的实际学习幅度或特征方向。零初始化残差门说明初始函数是 identity，不等于训练后仍无影响；性能下降表明最终行为已经改变，但具体哪些 token/位置被改变尚未被记录。

### 对三个对照方向的解释

- **表达能力**：两个位置都加入了可学习的跨模态条件映射，Neck 位置理论计算开销约为 loc6 的 4.75 倍；当前数据没有支持这种表达增量改善目标/背景区分。
- **参数量与计算量**：参数增量相同且很小，但 Neck 的注意力计算量更高。该结果说明参数量不能代表实际推理成本。
- **稳定性与可解释性**：两组胜负数相同，但失败幅度分布不同；Neck 在三条历史低 AUC 序列上更差。两者都缺少跨 seed 和模块级激活/梯度轨迹，因此稳定性和机理解释只能作初步判断。

## 可支持与不可支持的归因

共享随机初始化匹配、forward、shape、parity 和梯度合同检查通过，支持“代码接口可运行且三组初始化可比”。两组五轮筛选结果均低于 control，支持“当前 PVL 参数化/插入点/监督合同不值得直接长训”。它们不支持“概率不确定性不适合 RGB-T 跟踪”的普遍结论，也不定位造成单序列退化的具体 token、阶段或状态更新环节。

本次把两个候选和 control 都从通用 SOT 权重训练到 5 epoch，改变变量只有 PVL 插入位置；所以候选间可做位置筛选归因。但由于只有一个 seed、测试序列回看选取、训练期短，且未记录 PVL 专属表示变化，结果不能支撑总体泛化、显著性或模块机制结论。FPS 下降是观察值，不单独证明 PVL 的固有部署速度。

## 决策与后续

**NO-GO：两个位置都不进入长训，也不继续扩大 PVL 旁路搜索。** Neck 优先停止，原因是总体 AUC 更低、历史低 AUC 桶退化更重且理论 MAC 更高。loc6 同样不晋级；简单与额外困难组近似持平不能抵消其总体下降和历史低 AUC 桶回退。

如果未来需要重新评估 PVL，应先在独立 train-side 固定帧做最小可逆诊断：记录 PVL 门值、消息幅值、前景/硬背景参与度与 GT rank，并在相同 crop/state 上比较注入前后经后续融合与 CENTER 读出的 margin。只有能确认模块把变化集中到目标位置、且不扰乱非目标对照，才有理由重新设计训练信号或讨论另一个插入点。该后续属于新假设、新实验 ID；本结果不授权长训。

## 可复核证据

- 汇总与逐序列结果：`output/experiments/M1_PVL_POSITION_SCREEN_MATCHED_20261020/analysis/{summary.json,paired_per_sequence.csv}`
- 固定序列名单：`experiments/manifests/m1_pvl_position_screen_20261020.txt`
- 配置：`experiments/tbsi_track/M1_PVL_TEMPLATE_LOC6_MATCHED_5EP_20261020.yaml`、`experiments/tbsi_track/M1_PVL_SEARCH_NECK_MATCHED_5EP_20261020.yaml`
- 候选训练与测试：`output/experiments/M1_PVL_TEMPLATE_LOC6_MATCHED_5EP_20261020/`、`output/experiments/M1_PVL_SEARCH_NECK_MATCHED_5EP_20261020/`
- 原始三组汇总与 control 结果：`output/experiments/M1_PVL_POSITION_SCREEN_MATCHED_20261020/analysis/summary.json`
- 修复版源码快照 SHA256：`d5e6278fc4205dd430273df625cf366ddae16a85f772266572f8ba1e2636b990`
