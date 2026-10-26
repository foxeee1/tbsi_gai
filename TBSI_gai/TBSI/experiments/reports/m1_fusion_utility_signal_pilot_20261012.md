# M1 融合前信号效用快速筛查（2026-10-12）

> 2026-10-12 合同审计修正：以下衰减实验只测目标/背景 score margin 代理。背景控制虽等面积，却固定落在图像左上方；多数 severe 状态没有可判别的净 GT 排名变化，且未评分反事实预测框。旧准确率不能排除目标条件响应或学习型效用信号。详见 `experiments/reports/m1_fusion_signal_screen_contract_audit_20261012.md`；原始结果保留。

## 结论

**旧代理合同下的三个手工信号未获准进入门控训练。** 初始 17 序列筛查曾留下模板支持差这一小样本候选；随后在 72 条无重叠 train-side 序列、37 个 severe onset 上统一重提取并复测后，模板支持差对旧 margin 代理方向的命中率为 51.4%，评估集多数方向为 64.9%。能量差在全体固定状态有一定相关，但 severe onset 只有 45.9%。该对照不构成对实际框定位效用或其他任务信号的否定。

因此，本次快速实验**不支持马上训练融合 gate/MCCA**。它也没有否定 pre-fusion 位置；它只表明现有候选统计量对所定义的局部 CENTER 效用缺乏稳定、跨状态的增量预测能力。

## 假设与干预

- 假设：可部署的融合前信号能预测当前 RGB/TIR 哪一路的目标区域证据更有助于 CENTER 正确选择目标，而不只是预测特征幅值、模板响应或独立 probe 可分性。
- Parent：受保护 M1 epoch15，固定输入来自此前采集的 17 序列、70 状态固定 crop；GT 仅用于离线目标/背景掩码和效用评分。
- 单一实验变量：在最终 `tbsi_fuse_search` 输入处，对某一路目标掩码内特征以强度 `0.5` 向该路背景均值衰减；控制臂在等面积背景位置做同样衰减。
- 局部效用定义：`U_m = margin_control,m - margin_target-attenuation,m`，其中 margin 是冻结 CENTER Hann 后目标区最高分减背景区最高分。`U_rgb-U_tir > 0` 表示当前 crop 的 RGB 目标证据受到衰减时，CENTER 的目标/背景选择 margin 相对控制臂损失更大。
- 比较信号：局部能量差、模板支持差、模板 PSR 差。局部冲突强度没有方向，故只报告关联，不把其幅值符号解释为 RGB/TIR 胜者。

## 执行与合同

- 使用 M1 受保护配置与 epoch15 checkpoint，哈希分别为 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1` 和 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`；实验未训练、未写 checkpoint、未运行闭环评测。
- 固定状态特征包包含半精度存档的融合输入。重放融合输出相对存档 fused feature 的最大绝对差为 `0.00387`，用于界定该离线重放精度。
- 硬件/软件：1× NVIDIA GeForce RTX 4090 D 24 GiB；Python 3.8.20、PyTorch 2.0.0+cu118、CUDA 11.8；FP32；GPU 推理约 40.2 秒（`0.0112 GPU h`）。
- 首次启动在第一个样本的掩码索引处中断：NumPy bool 掩码不能直接索引 CUDA 张量。修复为设备内 torch bool 掩码后，同一冻结样本完整跑完 70/70；中断没有产生结果文件、更新权重或改变 M1 资产。
- 所有序列与旧 MCCA validation96 重叠；这不是独立验证集。实验结果是筛选性机制诊断，不可作为泛化或论文提升结论。

## 结果

| 划分 | 数量 | RGB 效用为正 | 恒选多数模态准确率 | 能量差方向准确率 / Spearman | 模板支持差方向准确率 / Spearman | 模板 PSR 差方向准确率 / Spearman |
|---|---:|---:|---:|---:|---:|---:|
| 固定状态 | 70 | 51 | 72.9% | 74.3% / 0.250 | 57.1% / 0.168 | 57.1% / 0.006 |
| 严重起点 | 10 | 6 | 60.0% | 60.0% / 0.079 | 80.0% / 0.491 | 50.0% / 0.055 |
| 非严重状态 | 60 | 45 | 75.0% | 76.7% / 0.302 | 53.3% / 0.084 | 58.3% / -0.015 |
| 序列均值 | 17 | 14 | 82.4% | 82.4% / 0.252 | 64.7% / 0.625 | 64.7% / 0.194 |

序列均值行中，每条序列先对其状态效用及信号求均值，再以序列作统计单位。严重起点只有 10 条序列，模板支持差的 8/10 是待复核线索；当前不能排除小样本波动、序列重叠或状态选择造成的乐观偏差。70 状态级结果还显示效用方向本身偏 RGB（51/70），所以能量差的 52/70 预测没有超越常数预测器多少。

## 独立扩样复核（72 序列，144 状态）

按冻结清单复用 72 条 outcome-blind train-side 序列的既有闭环 trace，在 rank 0/24/48 三批统一使用 `onset-3-iou-0.5` 规则重提取 144 个固定 crop 的融合前特征与模板 token。该队列与已审计的 MCCA validation96、identity calibration/holdout、此前融合信号与 pre-fusion admission 队列及已登记 MCCA attribution 选择均无序列重叠；清单及排除审计见 `experiments/diagnostics/m1_fusion_utility_signal_pilot_20261012/expansion72/manifest.json`。这是新的 train-side 诊断队列，但 M1 训练暴露仍未知，不能称为模型训练集外部验证。

对 144 个状态沿用相同的固定 crop 干预：在最终融合输入处分别将 RGB/TIR 目标区域特征以 `0.5` 强度衰减到该路背景均值，并与等面积背景衰减控制比较冻结 CENTER Hann 后目标/背景 margin。重放融合输出相对 half-precision 存档最大绝对差为 `0.003888`。GPU 为 1× RTX 4090 D 24 GiB；Python 3.8.20、PyTorch 2.0.0+cu118、CUDA 11.8、FP32。三批特征重提取累计约 `0.00814 GPU h`，144 状态反事实前向约 `0.00065 GPU h`；没有训练、闭环跟踪或改写 checkpoint。

| 划分 | 单位数 | 恒选多数效用方向 | 能量差 | 模板支持差 | 模板 PSR 差 |
|---|---:|---:|---:|---:|---:|
| 全部状态 | 144 状态 | 52.8% | 62.5% / Spearman 0.346 | 56.9% / 0.054 | 50.7% / -0.008 |
| Severe onset | 37 状态/序列 | 64.9% | 45.9% / 0.090 | 51.4% / -0.129 | 43.2% / -0.135 |
| 非 severe | 107 状态 | 58.9% | 68.2% / 0.403 | 58.9% / 0.070 | 53.3% / 0.026 |
| 按序列聚合 | 72 序列 | 55.6% | 73.6% / 0.406 | 48.6% / 0.002 | 54.2% / 0.027 |

全状态中 RGB utility 为正 76/144，效用方向没有强烈总体单模态偏置；但在最关心的 severe onset 上，TIR utility 为正 24/37。能量差的全体/按序列结果显示它能读出部分一般状态变化，却没有识别 severe 状态的模态效用方向。模板支持差和 PSR 差在 severe 状态均未超过恒选多数基线。冲突幅值与效用差绝对值的 Spearman 在 severe 子集仅 `0.041`。逐状态干预结果、序列聚合结果和完整汇总分别保存在 `output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/utility_replay144/per_state_utility.csv`、`.../utility_analysis/per_sequence_utility.csv` 和 `.../utility_analysis/summary.json`。辅助 probe AUC 的旧 `.../signals_analysis/` 输出只作历史证据；按序列留出并修正 cohort 元数据后的结果在 `.../signals_analysis_sequence_holdout_v2/`。

## 判定与下一步

- **原 17 序列口径暂不支持：**局部能量差、模板 PSR 差作为该 cohort 的普遍 margin 代理方向信号；能量差未超过该 cohort 的事后多数方向。
- **暂不支持：**局部特征能量差、模板支持差、模板 PSR 差作为 severe onset 的融合选择信号；72 序列扩样中三者都低于恒选多数基线。
- **尚未区分：**目标区衰减对 CENTER margin 的效用是否足以代表最终预测框效用。当前指标只测冻结 CENTER 的目标/背景选择 margin，背景控制位置不匹配，也没有逐框反事实 IoU 或闭环跟踪收益。事后多数方向只能作描述性参照。
- **本轮停止条件触发：**不训练 gate/MCCA，不在当前 cohort 上调阈值或拼接信号。新的 72 序列结果已检验之前保留的模板支持候选，因此不再把它列为准入前优先项。
- **下一步：**按合同审计报告先修正空间匹配控制和 CENTER 联合 score/offset/size 的离线效用定义，再按序列训练/留出目标条件响应信号；在此之前不启动融合门控训练。

## 复现材料

- 运行脚本：`tools/run_m1_fusion_utility_signal_pilot_20261012.py`。
- 分析脚本：`tools/analyze_m1_fusion_utility_signal_pilot_20261012.py`。
- 输入特征：`output/experiments/M1_FUSION_SIGNAL_EXPANSION_DIAG_20261012/combined_feature_capture_17/selected_features.npz`。
- 信号表：`output/experiments/M1_FUSION_SIGNAL_EXPANSION_DIAG_20261012/analysis_combined17_v2/per_state_signals.csv`。
- 逐状态结果：`output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/per_state_utility.csv`。
- 汇总：`output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/analysis_v2/summary.json`；按序列结果：`per_sequence_utility.csv`。
- 首次失败保留在会话日志；失败发生在有效前向输出前。最终运行命令：

```bash
python3.8 tools/run_m1_fusion_utility_signal_pilot_20261012.py \
  --features output/experiments/M1_FUSION_SIGNAL_EXPANSION_DIAG_20261012/combined_feature_capture_17/selected_features.npz \
  --signal-csv output/experiments/M1_FUSION_SIGNAL_EXPANSION_DIAG_20261012/analysis_combined17_v2/per_state_signals.csv \
  --output-dir output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012
python3.8 tools/analyze_m1_fusion_utility_signal_pilot_20261012.py \
  --input output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/per_state_utility.csv \
  --output-dir output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/analysis_v2
```

- 扩样补提取、信号重算、反事实复测与扩样分析命令：

```bash
TBSI_INFER_FP16=0 python3.8 tools/diagnose_m1_search_stage_20261009.py --trace-dir output/experiments/M1_SEARCH_STAGE_TRAIN_DIAG_20261009/trace_run/trace --output-dir output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/capture_rank0 --no-perturb --probe-reference-frames 0 --selection-rule onset-3-iou-0.5 --diagnostic-id M1-FUSION-UTILITY-SIGNAL-PILOT-20261012 --capture-template-tokens
TBSI_INFER_FP16=0 python3.8 tools/diagnose_m1_search_stage_20261009.py --trace-dir output/experiments/M1_SEARCH_STAGE_TRAIN_DIAG_20261009/trace_run_rank24/trace --output-dir output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/capture_rank24 --no-perturb --probe-reference-frames 0 --selection-rule onset-3-iou-0.5 --diagnostic-id M1-FUSION-UTILITY-SIGNAL-PILOT-20261012 --capture-template-tokens
TBSI_INFER_FP16=0 python3.8 tools/diagnose_m1_search_stage_20261009.py --trace-dir output/experiments/M1_SEARCH_STAGE_TRAIN_DIAG_20261009/trace_run_rank48/trace --output-dir output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/capture_rank48 --no-perturb --probe-reference-frames 0 --selection-rule onset-3-iou-0.5 --diagnostic-id M1-FUSION-UTILITY-SIGNAL-PILOT-20261012 --capture-template-tokens
python3.8 tools/merge_m1_fusion_signal_expansion_20261012.py --base-features output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/capture_rank24/selected_features.npz --base-center-csv output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/capture_rank24/per_arm.csv --extension-features output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/capture_rank48/selected_features.npz --extension-center-csv output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/capture_rank48/per_arm.csv --output-dir output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/merged_24_48
python3.8 tools/merge_m1_fusion_signal_expansion_20261012.py --base-features output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/merged_24_48/selected_features.npz --base-center-csv output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/merged_24_48/per_arm.csv --extension-features output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/capture_rank0/selected_features.npz --extension-center-csv output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/capture_rank0/per_arm.csv --extension-sequence-list output/experiments/M1_SEARCH_STAGE_TRAIN_DIAG_20261009/trace_run/trace/trace_sequences.txt --output-dir output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/merged_all72
python3.8 tools/analyze_m1_fusion_signal_admission_20261012.py --features output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/merged_all72/selected_features.npz --center-csv output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/merged_all72/per_arm.csv --output-dir output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/signals_analysis
python3.8 tools/run_m1_fusion_utility_signal_pilot_20261012.py --features output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/merged_all72/selected_features.npz --signal-csv output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/signals_analysis/per_state_signals.csv --output-dir output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/utility_replay144 --provenance-note '72 outcome-blind train-side sequences; 144 fixed states under unified onset-3-iou-0.5 selection; see frozen expansion72 manifest.'
python3.8 tools/analyze_m1_fusion_utility_signal_pilot_20261012.py --input output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/utility_replay144/per_state_utility.csv --output-dir output/experiments/M1_FUSION_UTILITY_SIGNAL_PILOT_20261012/expansion72/utility_analysis --provenance-note '72 outcome-blind train-side sequences; 144 fixed states under unified onset-3-iou-0.5 selection; zero overlap with audited earlier validation/screening cohorts per frozen expansion72 manifest; M1 training exposure unknown.'
```

- 扩样反事实汇总 SHA256 `454cb886d57211fcfd82f41b6983f6e63d800f80ab88067e68b47fbb39fa216b`；序列级分析汇总 SHA256 `38ff0be923d566b3f7cddc69e8746fe39c8729abf294c68ec5a1a42d046d810c`。受保护 M1 config/checkpoint 哈希在扩样前后仍分别为 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1` / `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`。
