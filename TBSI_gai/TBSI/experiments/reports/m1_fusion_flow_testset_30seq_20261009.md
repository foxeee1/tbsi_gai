# M1 融合信息流测试集 30 序列诊断（2026-10-09）

## 研究问题与可比边界

在 M1 epoch-15 固定 checkpoint 上，跨不同 LasHeR 场景查看模板 MHSA、head 前搜索融合、定位响应和逐帧定位结果之间的关系，并验证前次 3 序列 train-side 观察是否有跨序列一致趋势。本次只是描述性机制诊断，不训练、不更新权重、不用于选择 DBF 模块或插入位置。

配置为 `TBSI_E_MHSA_FULL_4090D_seed42`，seed 42，checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`，config SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`。环境为 1× NVIDIA GeForce RTX 4090 D，Python 3.8.20、PyTorch 1.9.0+cu111、CUDA 11.1、FP32。

序列清单由官方 244 序列属性表按加权属性覆盖算法预先选出，选择程序只读 `sequence` 和 `attributes` 字段，不读取逐序列指标或预测结果。30 条序列覆盖 19 个属性标签，但覆盖不均：例如 `PO=28`、`TC=21`、`LI=2`、`HI=1`、`HO=1`、`OV=1`。因此少数标签只是被代表，不能据此作属性子组结论。完整成员与逐序列标签见 `output/experiments/M1_FUSION_FLOW_TESTSET_DIAG_20261009/manifest.json`。

先在全部所选序列运行正常闭环 tracker，记录 42,636 个跟踪帧的输入状态、crop、预测框和 GT；再在每序列有效帧时间轴等距抽取 32 帧，共 960 帧，对定位损失反向求梯度并记录 location 3/6/9 模板 MHSA、`tbsi_fuse_search`、backbone 和 score response。采样点没有修改闭环状态。梯度是冻结权重对单帧定位损失的敏感度，不是训练更新或跨帧反传。

**重要可比性限制：** M1 原训练/全量评测时的完整源码快照未找到。本次用了相同 M1 checkpoint 和 config，但使用当前工作树代码。与受保护 full-244 归档逐帧比较，42,636 帧中 12,014 帧预测框最大坐标差超过 2 px，平均最大坐标差 8.370 px，最大 430.646 px；平均绝对 IoU 差为 0.03005。虽然匹配帧的合并平均 IoU 接近（当前重放 0.49196、归档 0.49064），逐帧差异和闭环漂移使本次结果**不可直接称为 M1 全量评测复现**。归档坐标为整数输出；超过 2 px 的差异表明取整本身不足以解释全部偏离，但偏离根因尚未定位。本次所有关联仅对“相同 M1 权重/config、当前工作树的这次闭环重放”成立。

## 结果

| 观察 | 数值 | 解释边界 |
|---|---:|---|
| 闭环覆盖 | 30 条序列，42,636 跟踪帧；每序列 32 个梯度样本，共 960 帧 | 逐序列抽样长度相同；帧仍有强时间相关，统计以序列为单位。 |
| 当前重放定位 | 有效 GT 帧平均 IoU 0.49196；相同帧的归档平均 IoU 0.49064 | 当前重放未通过归档逐帧 parity，接近的合并均值不能掩盖逐帧闭环分歧。 |
| 反向梯度 | 3/6/9 的 RGB/TIR MHSA 输入、输出、QKV/Proj 参数，以及搜索融合输出/参数梯度在 960/960 帧非零 | 被测路径在本次状态下可反传；不代表梯度训练时健康或有效更新。 |
| 模板 MHSA 激活改变量 | residual/input 范数均值：loc3 RGB/TIR `0.117/0.132`，loc6 `0.122/0.136`，loc9 `0.125/0.135` | 两路模板均有非零且低于输入范数的改变量；幅度不能单独判定融合不足/过度。 |
| GT 位置响应 | 每序列 GT-cell 响应与 IoU 的 Spearman 中位数 `+0.250`；30 序列 bootstrap 95% CI `[+0.173,+0.365]`；25/30 条为正 | 是本次跨序列较一致的定位相关信号，仍属 head 响应关联，不定位上游因果。 |
| 响应峰值 | 每序列 peak response 与 IoU 的 Spearman 中位数 `+0.426`；95% CI `[+0.274,+0.518]`；28/30 条为正 | 响应越强通常定位越好；高峰是否落在目标上仍需结合 GT-cell 指标。 |
| location 9 entropy | 每序列 Spearman 中位数 `-0.144`；95% CI `[-0.287,+0.045]`；11/30 条为正 | 区间跨零；前次 3 序列观察到的负关联没有在这 30 条上形成稳定序列级信号。 |
| location 9 跨模态消息 | TIR cross-message 中位数 `+0.026`，95% CI `[-0.164,+0.131]`；RGB 中位数 `+0.065`，95% CI `[-0.034,+0.369]` | 两个区间均跨零，当前不能说消息量更大就带来更好或更差定位。 |
| 定位损失梯度 | QKV grad norm 中位数相关 `-0.332`，95% CI `[-0.533,-0.154]`；搜索 RGB/TIR 输入梯度中位数 `-0.364/-0.394`，区间分别 `[-0.470,-0.134]`、`[-0.470,-0.191]` | 较差 IoU 常对应更大的损失敏感度，可能来自定位误差更大；不能将梯度范数解释为模块欠训练、贡献度或优化效果。 |

以上区间通过以序列为 cluster 的 bootstrap 估计中位 Spearman 不确定性；每条序列只有 32 个等距帧样本。区间不处理序列内部时间自相关，也不校正多重比较；结果用于描述一致性，不构成正式显著性检验。

## 判断与下一步

30 序列数据支持“定位 response 与该次重放的 IoU 同步变化”，没有支持“location 9 的 entropy 或跨模态 message 量与定位存在稳定关系”。较差定位时梯度范数上升，更像误差增大后损失敏感度变大，不能作为 DBF 应替换 MHSA 或搜索融合的证据。模态输入/梯度路径均存在，仍没有单模态可用性反事实，也没有固定状态的接口前后干预。

状态为 `DIAGNOSTIC-COMPLETE / CODE-MISMATCH / CAUSALITY-UNCONFIRMED`。**不据本次测试集结果决定 DBF 注入位置或启动训练。** 下一步最重要的是解决代码可比性：恢复/审计 M1 原评测代码快照，或明确建立一个当前工作树的新 parent 并重新做 baseline 评测；在逐序列预测 parity 达到预先设定容差后，再把梯度与定位关联用于接口级因果诊断。测试集本次已经作为诊断数据使用，不应用来挑选模块。

## 复现证据

- 属性清单：`output/experiments/M1_FUSION_FLOW_TESTSET_DIAG_20261009/manifest.json`，canonical sequence/attribute manifest SHA256 `899aa554ba7d02b84289f17390461b6cc687d230b133d32e13371e444b301143`。
- 闭环 trace：`output/experiments/M1_FUSION_FLOW_TESTSET_DIAG_20261009/closed_loop_trace/trace/`；summary 记录 42,636 帧和闭环耗时 1,457.17 秒，约 0.405 GPU 小时。
- 梯度采样：`output/experiments/M1_FUSION_FLOW_TESTSET_DIAG_20261009/gradient_sampled_32/{summary.json,per_frame.csv,metric_correlations.csv,excluded_frames.csv}`；重前向相对本次 trace 最大框差 0.000122 px。
- 汇总：`output/experiments/M1_FUSION_FLOW_TESTSET_DIAG_20261009/analysis/{analysis_summary.json,per_sequence_summary.csv,sequence_cluster_correlations.csv,archive_prediction_comparison.csv}`。
- 本次可复现源码快照：`output/experiments/M1_FUSION_FLOW_TESTSET_DIAG_20261009/provenance/current_source_snapshot.tar.gz`，SHA256 `86cd08a4df1c6bbfb9dd7a9e335c2e4e0753320c48e9814e183b03f8014ff453`；它记录当前工作树，不是 M1 训练时源码快照。
- 完整命令（均从 `/root/autodl-tmp` 执行）：

```bash
python3 TBSI_gai/TBSI/tools/build_m1_test_flow_manifest_20261009.py --per-sequence-csv TBSI_gai/TBSI/output/experiments/M1_FULL244_SEQUENCE_AUDIT_20261009/per_sequence.csv --output TBSI_gai/TBSI/output/experiments/M1_FUSION_FLOW_TESTSET_DIAG_20261009/manifest.json --count 30

/root/autodl-tmp/conda_envs/tbsi/bin/python -u TBSI_gai/TBSI/tools/run_m1_test_flow_trace_20261009.py --manifest TBSI_gai/TBSI/output/experiments/M1_FUSION_FLOW_TESTSET_DIAG_20261009/manifest.json --output-dir TBSI_gai/TBSI/output/experiments/M1_FUSION_FLOW_TESTSET_DIAG_20261009/closed_loop_trace

/root/autodl-tmp/conda_envs/tbsi/bin/python -u TBSI_gai/TBSI/tools/m1_fusion_flow_testset_diagnostic_20261009.py --trace-dir TBSI_gai/TBSI/output/experiments/M1_FUSION_FLOW_TESTSET_DIAG_20261009/closed_loop_trace/trace --checkpoint TBSI_gai/TBSI/output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar --config TBSI_gai/TBSI/experiments/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42.yaml --split testingset --frames-per-sequence 32 --output-dir TBSI_gai/TBSI/output/experiments/M1_FUSION_FLOW_TESTSET_DIAG_20261009/gradient_sampled_32

python3 TBSI_gai/TBSI/tools/analyze_m1_test_flow_20261009.py --root TBSI_gai/TBSI/output/experiments/M1_FUSION_FLOW_TESTSET_DIAG_20261009 --output-dir TBSI_gai/TBSI/output/experiments/M1_FUSION_FLOW_TESTSET_DIAG_20261009/analysis --archive-dir TBSI_gai/TBSI/output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42 --full-per-sequence-csv TBSI_gai/TBSI/output/experiments/M1_FULL244_SEQUENCE_AUDIT_20261009/per_sequence.csv
```
