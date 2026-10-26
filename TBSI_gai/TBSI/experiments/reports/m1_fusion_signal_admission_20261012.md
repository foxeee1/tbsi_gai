# M1 帧内融合信号准入诊断（2026-10-12）

> 2026-10-12 合同审计修正：此报告所用 probe 实际只排除当前状态，未排除同序列其他帧；以下 probe AUC 数值是历史探索结果，不能作为按序列留出结论。效用 AUC 也不等于 CENTER 实际框定位收益。详见 `experiments/reports/m1_fusion_signal_screen_contract_audit_20261012.md`；原数值保留追溯。

## 结论

**判定：当前“预测峰附近局部冲突 + 模板余弦峰显著性差”未通过信号准入；不启动空间门控或 MCCA 训练。** 在 12 条 train-side 序列的 48 个固定状态中，只有 6 个首个 crop 内 severe onset。逐位置局部冲突没有显示其强度能识别哪路更有目标判别力；模板支持差仅在 `1/6` 个 severe onset 中选中 held-out probe AUC 较高的一路，低于固定选 RGB 的 `4/6`。多数 severe 帧两路 probe AUC 都接近 1，效用差本身常接近零。

这是一次**探索性信号筛查**，不是对帧内条件融合的普遍否定。所有剩余合格序列都属于之前 MCCA validation96 的序列集合，故本轮不是独立序列验证；每个固定帧是新采集的，但序列级结果曾用于旧 MCCA 评估。M1 工作树重放也不等价于归档 checkpoint 的逐帧轨迹。

## 预设问题与信号定义

- **H1 局部冲突：**最终 RGB/TIR search 特征在 M1 Hann top-1 周围的逐位置余弦不一致，能否标出两路目标判别效用差较大的帧？冲突图为 `1-cos(S_rgb(p), S_tir(p))`；取 M1 预测峰周围 3×3 均值作为推理时可计算的帧内标量。
- **H2 目标条件支持差：**分别用初始模板前景 token 加权均值，与对应模态当前 search 特征计算 cosine support map；每一路的信号为其最高峰减去峰周围 3×3 外最强竞争峰。RGB/TIR 两路峰显著性差的符号预测哪一路更有判别力。
- **离线效用标签（历史口径）：**每个模态独立训练容量相同的线性 probe，原实现只对当前状态留出，同序列其他帧仍参与拟合；先前写为 leave-one-sequence-out 是错误的。GT 只定义目标支持区和背景区、拟合 probe 与评分。效用主量使用当前状态的 target-vs-background AUC，避免直接相减两个独立 probe 的未校准 logit margin。该 AUC 仅用于历史诊断，不进入运行时信号，也不能作为 CENTER 定位效用标签。
- **真实定位核对：**同一固定 crop 上配对 M1 Hann GT rank、选中格 offset/size 误差、最终 IoU 与 GT-cell oracle IoU；oracle 只作离线上界。

## 数据与执行

- Parent：受保护 `TBSI_E_MHSA_FULL_4090D_seed42` epoch15；config SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1`，checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`，前后未变。
- Cohort：LasHeR trainingset；排除登记过的 train-side trace/fixed-state 诊断及 MCCA attribution 序列。剩余 17 条合格序列全部与 MCCA validation96 列表重叠；outcome-blind 确定性均衡抽取 12 条。sequence list SHA256 `f30e7adf1ffe5fe317186f9cc9eb0e91f043deaa494f02bcfe522bf4ae466442`。
- 独立性处理：严格排除 MCCA validation96 后，新 cohort builder 找到 0 条合格序列，未启动该配置；随后明确把研究范围降为探索性“新帧、旧序列”筛查，并在资产中标明重叠。不能把本次表现当作独立验证。
- 完成 12 序列、3,841 个闭环 tracking frame；固定 48 个状态：6 severe onset、6 pre-onset、36 normal reference。固定状态重放相对原 trace 的最大框误差 `5.29e-5 px`。
- 实际环境：1× NVIDIA GeForce RTX 4090 D 24 GiB，Python `3.8.20`、PyTorch `2.0.0+cu118`、CUDA `11.8`、FP32。trace 131.49 秒，feature capture 8.34 秒，约 `0.039 GPU小时`；线性 probe / 汇总在 CPU 上。没有训练、模块干预、checkpoint 写入或 full-test 评测；AUC/PR/NPR/OP50/OP75/FPS 对本诊断记 `NA`。
- 首次 trace launcher 命令含不支持的 `--diagnostic-id` 参数，在 argparse 阶段退出，未加载模型或启动 GPU；随后移除该参数正常完成。初版 `analysis/` 将未校准的两路 probe logit margin 直接相减，发现口径不成立后保留原始目录并标为 superseded；结论只引用 `analysis_v2/` 中尺度无关的 AUC 版本。

## 结果

### 补充信号筛查（analysis_v3）

在同一 48 个固定状态上追加筛查了模板支持图 PSR/熵、RGB/TIR 支持图相关性与峰距、M1 预测峰附近的局部特征能量比/邻域一致性，以及 CENTER 融合后 score 熵。该补充分析复用原 cohort，不构成独立验证，也没有训练模块。

- 48 状态中只有 7 个满足“两路 held-out probe AUC 差至少 0.02 且信号非零”；6 个 severe onset 中仅 1 个达到该效用差门槛。因此所有方向准确率都必须结合极小分母看，不能用总体状态数作有效样本数。
- 局部特征能量比差在这 7 个可判别状态中方向命中 `7/7`，全状态 Spearman `0.281`；其中 severe 的命中率也是 `1/1`。这是值得后续独立复核的线索，但其有效样本实际上只有一个 severe 状态，且该量是特征幅值代理，不能解释为模态质量或证明能改善 CENTER 定位。
- 模板 PSR 差在可判别状态命中 `4/7`，Spearman `-0.271`；模板支持峰显著性差命中 `6/7`，但全状态 Spearman `-0.102`，severe 可判别例 `0/1`。这些样本内命中率易受同一数据筛查与效用定义影响，不据此准入。
- 邻域特征一致性差命中 `2/7`，Spearman `-0.056`；模板支持熵差命中 `2/7`，Spearman `0.015`。目前没有稳定支持。
- RGB/TIR 模板支持图的峰距中位数 `7.41` token、图相关性中位数 `0.375`，说明两路模板响应图常不一致；但“不一致”本身仍不能判断哪一路有利。
- 融合后 CENTER score 熵与当前 `1-IoU` 的全状态 Spearman 为 `0.772`，severe/normal 熵中位数分别为 `0.565/0.414`。这是当前较清晰的隐式不确定性信号，可用于后续**后融合置信/回退**研究；它在融合之后才产生，不能条件化单次前向中的融合前模块，也不证明回退机制会带来收益。

补充结果和逐状态数值见 `output/experiments/M1_FUSION_SIGNAL_ADMISSION_DIAG_20261012/analysis_v3/summary.json` 与 `per_state_signals.csv`。分析脚本 SHA256 为 `221cd364488e933b49a0e9f7d3bbf36eebee2db8ea94b30cddcb1fe3c1031a08`；summary SHA256 为 `ad59ab871662d4fcbc9d71617699da724e8c61bb07851a65237ae84bdc963af2`。

### 信号与模态效用

- 6 个 severe onset 的局部冲突均值中位数为 `0.623`，但与两路 held-out AUC 绝对差的 Spearman 为 `-0.37`（`n=6`）；高冲突两例的 AUC 差绝对值中位数 `0.0026`，其余四例为 `0.0016`。数值小且方向不支持“冲突越强，越需要选择性融合”。
- 模板 cosine support 两路的 GT target-vs-background margin 中位数均为负（RGB `-0.122`、TIR `-0.111`），只有 `1/6` 帧的任一路 cosine support top 落在 GT 区域。原始模板 cosine 在此 cohort 仍不是可靠目标支持标签。
- 6 个 severe onset 中 RGB/TIR held-out AUC 分别为 `0.982–1.000` 和 `0.742–1.000`。仅一个状态的 AUC 绝对差超过 `0.02`，该状态 RGB AUC `0.996`、TIR AUC `0.742`，但模板峰显著性差指向 TIR，方向错误。按所有 6 个 AUC 差的符号，模板支持差选对较高 AUC 模态 `1/6`；常量 RGB 选择为 `4/6`。小 AUC 差不应解释成确定的模态胜负。
- 由于大部分两路 probe AUC 接近天花板，本 cohort 对“哪一路更有用”的监督判别力很弱；因此上面的低准确率只能说明**这版支持信号未通过准入**，不能据此否定所有目标条件可靠性信号。

### 与 CENTER 定位对应

- Severe onset CENTER Hann GT rank 中位数 `5.5`；预测格 offset 后中心误差中位数 `0.621` token，size L1 中位数 `1.555` token；实际 IoU 中位数 `0.446`。强制 GT 格仅在 `2/6` 帧令 IoU 增加超过 `0.1`，GT-cell oracle IoU 中位数 `0.507`。
- 这批帧同时包含选格错误和框回归限制，不能把当前 severe 框错误全归给融合信号。正常参考帧的 M1 IoU 中位数 `0.847`；本诊断没有做任何调制，故不能据此判定候选模块对正常帧是否安全。
- 六个 severe 个例、逐状态局部冲突/模板支持量、probe AUC 和 CENTER 指标见 `output/experiments/M1_FUSION_SIGNAL_ADMISSION_DIAG_20261012/analysis_v2/per_state_signals.csv`。

## 归因与决策

- **已排除：**这轮不是接线、梯度、训练或状态合同问题；没有新模块或训练。
- **暂不支持：**仅用预测峰附近 cosine 冲突强度加上初始模板 cosine 峰显著性差，作为逐位置双模态平衡门控的依据。它没有稳定识别出较有用模态，且效用 AUC 多数天花板化。
- **尚未区分：**学习式目标匹配支持量能否优于 raw cosine；特征冲突是否应通过 CENTER 的真实候选竞争而非分支 probe 定义；原融合输入被重加权后能否提高最终框。
- **允许的下一步：**优先找不与 MCCA validation96 重叠的独立序列，并预注册检验局部特征能量信号是否在严重失配帧中能预测分支定位效用；需要同时做特征幅值归一化/扰动诊断，排除它只是激活尺度代理。CENTER 熵可另立后融合置信研究。两者都先做只读信号验证，不直接训练当前未通过的 gate/MCCA，也不在本 cohort 上调到正向。
- **不能声称：**本诊断证明融合无用、IGF 原理无效、MCCA 所有算子无效，或该 12 序列结果代表 LasHeR/full-244 的总体性能。

## 复现身份与产物

- Git HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`；相关工作树源码快照 `experiments/snapshots/m1_fusion_signal_admission_sources_20261012.tar.gz` SHA256 `becfe8cae93b81e24d13808fa675fe1e94de5d5d17f43991b2272dc908fbd9b4`；trace runner SHA256 `fee4c4bb5e68bda0390bd651e8ec1c09d56f02add81eee07d441da4d4092ebba`；新 cohort builder SHA256 `9ea133ccde8279a0abe2bfc7b41b3430948768911fbabd3714c3e7cc59c352fb`；固定状态捕获工具 SHA256 `0eae248de318bca3d9356d384054217aaaec4e731f3188bed34176398d08be41`；分析工具 SHA256 `4413da266f3c6adce52a50372071554baf06b247ec7c80c96f556e9a1c34ef5f`。
- Manifest：`experiments/diagnostics/m1_fusion_signal_admission_20261012/manifest.json`，SHA256 `579e3a9f4fd0da575ede35ffcbdd687c6b6a0c5372e6e16e02e2a00b2a46084e`。
- 轨迹：`output/experiments/M1_FUSION_SIGNAL_ADMISSION_DIAG_20261012/trace/`；固定状态 feature capture：`.../feature_capture/selected_features.npz`，SHA256 `02b867895bca793e4e7c034a11c75795e6a89897b81a7032eafb3cd54defdcda`。
- 权威分析：`output/experiments/M1_FUSION_SIGNAL_ADMISSION_DIAG_20261012/analysis_v2/summary.json`，SHA256 `566054488a7be68c1d7317e87046a9e0698229d1ecf5b35352e72a7d07717efb`。初版 `analysis/` 保留但已 superseded。
- 命令：

```bash
python3.8 tools/build_m1_fusion_signal_admission_cohort_20261012.py
TBSI_INFER_FP16=0 python3.8 tools/run_m1_search_stage_train_trace_20261009.py --output-dir output/experiments/M1_FUSION_SIGNAL_ADMISSION_DIAG_20261012/trace --sequences 12 --sequence-list experiments/diagnostics/m1_fusion_signal_admission_20261012/sequence_list.txt
TBSI_INFER_FP16=0 python3.8 tools/diagnose_m1_search_stage_20261009.py --trace-dir output/experiments/M1_FUSION_SIGNAL_ADMISSION_DIAG_20261012/trace/trace --output-dir output/experiments/M1_FUSION_SIGNAL_ADMISSION_DIAG_20261012/feature_capture --no-perturb --probe-reference-frames 2 --selection-rule onset-3-iou-0.5 --diagnostic-id M1-FUSION-SIGNAL-ADMISSION-DIAG-20261012 --capture-template-tokens
python3.8 tools/analyze_m1_fusion_signal_admission_20261012.py --features output/experiments/M1_FUSION_SIGNAL_ADMISSION_DIAG_20261012/feature_capture/selected_features.npz --center-csv output/experiments/M1_FUSION_SIGNAL_ADMISSION_DIAG_20261012/feature_capture/per_arm.csv --output-dir output/experiments/M1_FUSION_SIGNAL_ADMISSION_DIAG_20261012/analysis_v2
```

- 唯一资产 ID：`M1-FUSION-SIGNAL-ADMISSION-DIAG-20261012`；状态 `DIAGNOSTIC-COMPLETE / SIGNAL-NOT-ADMITTED`。
