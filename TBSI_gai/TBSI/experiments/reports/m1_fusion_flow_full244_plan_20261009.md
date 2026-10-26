# M1 全量 LasHeR 融合信息流诊断：设计与预启动记录

日期：2026-10-09  
实验 ID：`M1-FUSION-FLOW-FULL244-DIAG-20261009`  
Parent：受保护 M1 `TBSI_E_MHSA_FULL_4090D_seed42`，epoch 15，seed 42  
当前状态：`DIAGNOSTIC-COMPLETE / CODE-MISMATCH / CAUSALITY-UNCONFIRMED`

## 目的与边界

对 LasHeR testingset 官方顺序中的全部 244 条有效序列做固定 M1 权重的 FP32 当前工作树回放，并在各序列等距抽 32 帧测量冻结模型对定位损失的梯度敏感度。目的为补齐全量序列的机制描述：观察模板 MHSA locations 3/6/9、head 前搜索融合、目标响应和最终定位的共变关系。该测试集诊断是观察性分析，不是接口干预实验；不训练、不改权重、不选择 DBF 注入位置。

已有 30 序列测试诊断发现当前工作树与受保护归档的逐帧跟踪输出不一致：12,014/42,636 帧最大坐标差超过 2 px，最大差 430.646 px。全量延伸可补充描述范围，但不能修复源码 parity，也不能称为 M1 严格复现。后续所有关联只解释“固定 M1 checkpoint + 当前工作树回放”。

## 固定数据与权重

- 成员：`LasHeRDataset(testingset)` 返回的全部 244 条序列，官方顺序；不按结果筛样。清单还附 19 个官方属性标签供描述性分层。
- 数据量：219,292 帧（包含各序列初始化帧），219,048 个闭环跟踪帧；梯度样本目标 7,808 行。
- checkpoint SHA256：`1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`。
- config SHA256：`fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`。
- 环境：1× RTX 4090 D 24 GiB，Python 3.8.20，PyTorch 1.9.0+cu111，CUDA 11.1，FP32；跟踪单主进程，无 DataLoader worker，PyTorch CPU 线程数为 4。
- manifest SHA256：`888751a18ace71e3422054293b5c5df19c15859e288e30a77165d598c901ec10`。
- 源码证据快照：`output/experiments/M1_FUSION_FLOW_FULL244_DIAG_20261009/provenance/source_snapshot_20261009.tar.gz`，SHA256 `bcc70a0204f43d3e22f37b1b39aa1b65970e6b5510d986fbb8d953938f10d3ca`。它保存本次工具、诊断改动、tracker 代码和 M1 config，不是 M1 训练时源码快照。

## 测量设计

1. **闭环追踪**：复用正式 tracker 初始化和逐帧状态更新，记录每帧状态、crop、预测框和 GT。每完成一条序列后验证 JSONL 帧号连续且长度正确，再原子更新 `run_state.json`。中断后只跳过校验通过的完整序列；残缺文件隔离后重跑。
2. **梯度与融合流**：每条序列在有效 GT 帧时间轴等距取 32 帧。对每帧固定 trace 的 `tracker_state_before` 重前向并计算与训练定位项一致的 `2*GIoU + 5*L1 + focal` 目标，反传但不更新参数。按 30 条序列分批保存，任意整批完成后可续跑。
3. **模板 MHSA 信息**：locations 3/6/9 分别记录 RGB/TIR 输入与输出范数、残差差值和比例、输入输出余弦、attention 熵、双向跨模态注意力质量、目标/背景 key 质量、各模态输入/输出梯度、梯度乘激活幅度，以及 QKV/Proj 参数梯度。
4. **搜索接口与定位响应**：记录 `tbsi_fuse_search` 两模态输入统计、融合输出和梯度、搜索融合参数梯度、backbone 特征/梯度；head 侧记录 peak response、GT-cell response、目标与背景响应差、峰值中心误差、crop 内 GT 覆盖、预测 IoU 和各损失分量。
5. **关联分析**：产出逐帧、逐序列和属性分层结果；计算逐序列 Spearman、pooled 与序列内中心化关系，以及以序列为 cluster 的中位相关 bootstrap 区间。逐帧强时间相关、属性成员重叠和多重比较意味着该结果仍为描述性证据，不作显著性或因果声明。
6. **parity 审核**：梯度重前向预测必须与对应 trace 框在预设 0.01 px 容差内一致。另逐帧比较当前闭环结果与 M1 归档结果，输出坐标差和 IoU 差；归档差异不满足 M1 源码 parity 时明确标 `NOT-DIRECTLY-COMPARABLE`。

## 预启动结果

全量 preflight 两次通过：清单和数据均为 244 条；219,292 帧；checkpoint/config 哈希吻合；Python/PyTorch/CUDA/GPU 与登记环境一致；图像路径存在；preflight 时可用空间 5.294 GiB，smoke 后约 5.293 GiB。无其他 GPU 计算进程。

两条 smoke 序列为官方顺序开头的 `10runone`、`11leftboy`，合计 454 跟踪帧。64 个梯度样本全部完成，24 个 `*_grad_norm` 列在 64/64 行非零；trace 重放 parity 最大框差 0.000122 px，最大 IoU 差 0.00000290，均通过 0.01 px 容差。smoke 是工程通路检查，不能代表全量指标或模块效果。

## 产物

- 完整序列/属性清单：`output/experiments/M1_FUSION_FLOW_FULL244_DIAG_20261009/manifest.json`。
- 可续跑闭环：`closed_loop_trace/run_state.json`、`closed_loop_trace/trace/`、`closed_loop_trace/trace_summary.json`。
- smoke 梯度验证：`gradient_smoke/{summary.json,per_frame.csv,metric_correlations.csv,excluded_frames.csv}`。
- 全量梯度完成后：`gradient_sampled_32/{summary.json,per_frame.csv,metric_correlations.csv,excluded_frames.csv,batches/}`。
- 最终分析：`analysis/{analysis_summary.json,per_sequence_summary.csv,sequence_cluster_correlations.csv,archive_prediction_comparison.csv}`。
- launcher 日志建议写入：`logs/full244.log`。

## 资源预算与运行状态

此前 30 序列闭环约 29.26 FPS；按 219,048 跟踪帧粗估闭环约 2.1 小时。梯度阶段有 7,808 次重前向/反传，smoke 帧速不稳定且只覆盖 64 帧，预估再需约 0.5–1 小时。总预算按约 2.5–3.5 小时安排，实际时间以 launcher 日志为准。full trace 和分批 CSV 预计低于 1 GiB；当前仍有约 5.3 GiB 空间。

## 全量完成与续跑记录

2026-10-09 全量 launcher 完成，日志以 `ALL_STAGES_COMPLETE` 结束。闭环为 244/244 条、219,048 帧；9/9 梯度批次全部 parity 通过，最终得到 7,807 个梯度样本。`lowerfoamboard` 在 trace 中有 31 个可采样有效 GT 帧，其余 243 条各有 32 个；初版聚合器错误地要求每序列恰为 32 行，因此第一次在批次产物全部完成后于合并校验退出（7,807/7,808）。修复后按每序列有效 trace 帧数校验并续跑，没有重算闭环或梯度批次，也没有复制样本。分析器同步接受该实际样本数。

汇总相关计算原实现反复扫描全部帧，续跑中观察到单核长时间满载；改为按序列预分组的 NumPy 中心化计算后，梯度汇总在约 38 秒完成。最终修订脚本快照：`output/experiments/M1_FUSION_FLOW_FULL244_DIAG_20261009/provenance/resume_fix_source_v2_20261009.tar.gz`，SHA256 `73fe7c9afbcb18a41e306da38b171dce7c82286294efc409454b1ac2ece70edd`。它还将分析 summary 中的 32 帧目标数与实际 31–32 帧范围分开记录。原始预启动源码快照和第一版续跑快照均保留。

全量结果位于 `output/experiments/M1_FUSION_FLOW_FULL244_DIAG_20261009/analysis/`。24 个梯度范数列在 7,807/7,807 样本中均为非零；所有梯度批次的 trace 重前向 parity 通过，最大框差 0.000122 px。当前工作树回放与归档预测仍明显不同：82,219/219,048 帧坐标差超过 2 px，平均最大坐标差 15.029 px，最大 849.103 px；因此不是 M1 严格复现或直接可比结果。描述性序列中位 Spearman 例子：响应峰值与 IoU +0.439，loc9 QKV 梯度范数与 IoU -0.425，搜索 RGB/TIR 输入梯度范数分别为 -0.437/-0.448。所有均为测试集观察性关联，不支持因果结论，也不得用于选择模块或注入位置。正式跟踪 AUC/PR/NPR/OP50/OP75 未在本诊断中重算，登记为 NA；GPUHours 未单独记录，登记为 NA。

最终环境与权重/config 哈希仍与 M1 登记一致：1× NVIDIA GeForce RTX 4090 D 24 GiB，Python 3.8.20、PyTorch 1.9.0+cu111、CUDA 11.1、FP32；checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`，config SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`。

## 结论口径

全量完成后只报告本次工作树回放中“模态信息/梯度/响应与定位指标的统计关系”，并一并报告与 M1 归档的逐帧偏差。即便观察到相关性，也不能称为某融合模块造成增益/退化，不能以测试集选定模块或注入位置。要回答“接口处哪种信息导致定位改变”，后续仍需在 train/dev 侧固定同一 crop/state 做受控接口反事实；那是另一项实验。
