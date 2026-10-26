# M1 帧内融合信号扩展筛查（2026-10-12）

> 2026-10-12 合同审计修正：本报告复用的 probe 分析脚本只排除当前状态，并非按序列留出；下面的 probe AUC 方向命中率保留作历史探索结果，不具备所述留出效力，也未检验实际 CENTER 框定位收益。修正与 72 序列另队列复核见 `experiments/reports/m1_fusion_signal_screen_contract_audit_20261012.md`。

## 结论

**主判定：严格排除既有 MCCA attribution 序列后，17 序列结果仍不支持当前信号准入融合前门控或 MCCA 训练。** 主集 70 固定状态中有 14 个状态的两路 probe AUC 差至少 `0.02`；10 个 severe 序列中只有 4 个达到此门槛。局部特征能量差总体命中 `11/14`，但 severe 仅 `2/4`，低于恒选多数模态的 `3/4`。另 3 条历史 attribution 序列只作为敏感性分析，不纳入主结论。

融合后 CENTER score 熵与 IoU 误差仍有较强关联：主集状态级 Spearman `0.707`，按序列均值后 17 序列 Spearman `0.809`。它是这轮最稳定的隐式风险信号，但在融合后才可见，不能用于同次前向的融合前调制。当前不应用这些结果训练门控；若继续帧内融合研究，应先在独立序列上验证可部署的融合前目标响应信号。

本扩展仍是 train-side 探索筛查。主集新增 5 条序列及敏感性集 3 条序列都与旧 MCCA validation96 重叠，所以**没有形成独立验证集**。执行后才发现初始 8 条清单中有 3 条已在旧 MCCA attribution 选择集中；已按原始排除清单重算 17 序列主集，并将 20 序列结果降为敏感性分析。

## 设计与执行

- 新版本 ID：`M1-FUSION-SIGNAL-EXPANSION-DIAG-20261012`；父诊断：`M1-FUSION-SIGNAL-ADMISSION-DIAG-20261012`。
- 唯一变量是序列集扩展。执行后 provenance audit 发现 8 条中 `blkbikefromnorth`、`boyleave`、`man2startmoto` 出现在旧 MCCA attribution 选择集；主扩展因此限定为 `2runfour`、`Blue_in_line_1202`、`The_girl_with_the_cup_1202`、`bikeout`、`carfarstart` 五条，清单 SHA256 `6ea38a73e8d21d4894ae9e718b68342aa2b5bbed4c91ccc78e786a4700c268df`。三条重叠序列单独保留在敏感性清单，SHA256 `521c7722a8139770303f5146e905f6856dcd46474141be6f5b6229c7a935063b`。
- 使用受保护 M1 epoch15；配置 SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1`，checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`。两者前后保持不变；所有新增模块均关闭。
- 采集覆盖全部新增 8 条：2,529 个闭环帧；trace `81.573 s`，固定状态捕获 `6.811 s`，合计约 `0.02455 GPU小时`。扩展捕获 37 个状态：7 severe onset、7 pre-onset、23 normal/reference。按排除规则保留 5 条主扩展后，与原 12 条合并为 17 条、5,216 个闭环帧、70 个固定状态：10 severe、10 pre-onset、50 normal/reference。另 3 条重叠序列加入敏感性结果时共 20 条、6,370 个闭环帧和 85 个状态。
- 固定状态重放最大框 parity 误差 `5.34e-5 px`。环境为 1× NVIDIA GeForce RTX 4090 D 24 GiB、Python `3.8.20`、PyTorch `2.0.0+cu118`、CUDA `11.8`、FP32。没有训练、模块干预、full-test 评测或 FPS 测量。
- 保持原状态抽取、目标/背景掩码、两个分支当前状态留出 probe、`AUC gap >= 0.02` 信息量门槛和信号定义。原报告误写为 leave-one-sequence-out；GT 仅用于离线状态选择、probe 拟合和评分。

## 结果

### 分支效用与融合前信号

17 序列主集的 70 个状态中有 14 个状态的两路 held-out probe AUC 差至少 `0.02` 且候选信号非零；10 个 severe 序列中只有 4 个 severe 状态达到该效用差门槛。后续方向结论以这 4 个 severe 状态为主要单位。

| 信号 | 全部可判别状态 | severe 序列状态 | 判读 |
|---|---:|---:|---|
| 局部特征能量比差 | `11/14`，Spearman `0.271` | `2/4` 可判别 severe，severe 全样本 Spearman `-0.370`（n=10） | 不稳定；低于恒选多数模态 `3/4` |
| 模板峰显著性差 | `9/14` | `1/4` | 不支持 |
| 模板 PSR 差 | `8/14` | `3/4` | 仅等于恒选多数模态基线 |
| 模板熵差 | `4/14` | `2/4` | 当前不支持 |
| 邻域特征一致性差 | `4/14` | `1/4` | 当前不支持 |

局部冲突强度与 severe 状态两路效用差绝对值的 Spearman 为 `-0.394`（n=10）。最高冲突三个 severe 状态的 AUC gap 绝对值中位数为 `0.0041`，其余七个为 `0.0082`。因此本批数据不支持“冲突越强，越应调节模态权重”的假设；模态表示不一致并不等价于一边更可靠。

RGB/TIR 模板支持图的 severe 峰距中位数为 `6.52` token，相关性中位数为 `0.223`。两路响应图分歧明显，但当前 raw cosine 形状仍没有提供稳定的胜出模态判别。

### CENTER 不确定性信号

- 融合后 CENTER score 熵对当前状态 `1-IoU` 的状态级 Spearman 为 `0.707`。
- 对每条序列内固定状态的 entropy 与 `1-IoU` 分别取均值后，17 条序列的 Spearman 为 `0.809`。这是描述性相关，不是独立校准结果；仍需在未参与筛查的序列上评估校准、召回和误报。
- severe CENTER Hann GT rank 中位数为 `1.5`，offset 后中心误差中位数 `0.712` token，size L1 中位数 `1.319` token，IoU 中位数 `0.433`，GT-cell oracle IoU 中位数 `0.477`。仍有选格与框读出两方面限制，不能单归因于融合。

### 新增序列敏感性分析

新增 8 序列 capture 有 37 个状态，其中 17 个对两路 probe AUC gap 达到 `0.02`。由于其中 3 条已进入旧 MCCA attribution 选择集，这组只做敏感性分析：局部能量差全状态命中 `12/17`，severe 命中 `3/6`；CENTER entropy 与 IoU error 的状态级 Spearman 为 `0.584`。包含这 3 条后的 20 序列合并结果是能量差 `15/22` 总体、`2/7` severe，CENTER entropy 状态级/序列均值相关分别 `0.652/0.802`。这些敏感性结果不改变主集“不准入”的决定。

## 归因与决定

- **已排除：**原先局部能量差 `7/7` 的表现不是稳定的 severe 起点规律；严格排除历史 attribution 后主集 severe 命中 `2/4`，加入重叠序列的敏感性结果为 `2/7`。
- **暂不支持：**局部冲突、raw template cosine/PSR/entropy、特征能量差或邻域一致性作为融合前模态门控的依据。都没有在 severe 状态上稳定优于固定多数模态基线。
- **支持下一步单变量检验：**CENTER score entropy 可作为帧级错误风险指标候选，但它只能服务于融合后置信估计/回退，不可声称为融合前调节信号；必须在独立序列上预先固定阈值并报告风险召回、误报和正常帧损伤。
- **尚未区分：**学习式、目标条件分支响应图能否提供优于 raw cosine 的融合前证据；当前独立 probe AUC 是否能预测实际 CENTER 在单路证据上的定位效用。
- **项目决定：**不启动本候选的 gate/MCCA 训练，也不扩展本 cohort 上的阈值或信号组合搜索。要继续融合创新，应优先在未筛查序列上检验“融合前每模态候选峰 margin/目标支持与真实定位效用的对应关系”，明确它能否识别该帧哪一路带来更好的 CENTER 选格和框回归。

## 复现材料

- cohort 协议与清单：`experiments/diagnostics/m1_fusion_signal_expansion_20261012/protocol.md`、`sequence_list.txt`、`primary_sequence_list.txt`、`sensitivity_overlap_sequence_list.txt`、`manifest.json`（manifest SHA256 `f69a5e2527e2e4c138e125813e1370fe168ae1d83729e0fb40386607b2735daf`）。
- 扩展 trace：`output/experiments/M1_FUSION_SIGNAL_EXPANSION_DIAG_20261012/trace_run/`；trace manifest SHA256 `1e409746dabbb3a8a563945dd128e11aa32582adcb550af185a8c2f0096873c8`，run state SHA256 `50f96f5b53275b46f1fdafcad4189193a3d5a41057812b64e3b0896ed62092e8`。
- 新 8 序列 feature capture：`.../feature_capture/`；summary SHA256 `ed0a2893aade8ff3f3da1daa1db3989f3533e999ba87035c3b5e6845846a3de5`，features SHA256 `4875554b5d522715dbdbcb380664b3d6a7175ab429fc49600387ee9e3512a6fb`。
- 新 8 序列单独敏感性分析：`.../analysis_new8/summary.json`，SHA256 `99fd9a660d1a261d7e42aa992f6369cc7637c41f15c58559603a37cc2195e3f3`，使用初版 analyzer SHA256 `221cd364488e933b49a0e9f7d3bbf36eebee2db8ea94b30cddcb1fe3c1031a08`。
- 17 序列主结果：`.../combined_feature_capture_17/`、`.../analysis_combined17_v2/summary.json`，summary SHA256 `a28dfca7e2da746193be3598c1e69f5af516eab9c5bfc819b9e7650e5da02c06`；主合并 feature SHA256 `85a57ce49378febeeb68f58c343c8ff18bb0c5ce3f4e8bf558c97173afc5f544`。20 序列全量结果只作敏感性分析，保存在 `.../analysis_combined20/summary.json`，summary SHA256 `eebc2ff4d07e4f9ba32841ebcee200e1936e758c702210900f345d29bbb815b4`，使用同一初版 analyzer。原 12 序列 `analysis_v3` 与原始 feature archive 均保留。
- Git HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`；protocol SHA256 `81535c06823e71a554ca4bfb5b7c928a3f657ba8e87ac9483bda712abf35d68e`；主序列清单 SHA256 `6ea38a73e8d21d4894ae9e718b68342aa2b5bbed4c91ccc78e786a4700c268df`；merge 工具 SHA256 `26578537fd0d840de2958d113f3b61848e270de61a8fedc9f76d770520640291`；analysis 工具 SHA256 `35dca7a580ec9035c8fce692c990dd633fe345890d6a3e6e93aaa9b2e133f252`；trace runner SHA256 `fee4c4bb5e68bda0390bd651e8ec1c09d56f02add81eee07d441da4d4092ebba`；capture tool SHA256 `0eae248de318bca3d9356d384054217aaaec4e731f3188bed34176398d08be41`。
- 命令：

```bash
TBSI_INFER_FP16=0 python3.8 tools/run_m1_search_stage_train_trace_20261009.py --output-dir output/experiments/M1_FUSION_SIGNAL_EXPANSION_DIAG_20261012/trace_run --sequences 8 --sequence-list experiments/diagnostics/m1_fusion_signal_expansion_20261012/sequence_list.txt
TBSI_INFER_FP16=0 python3.8 tools/diagnose_m1_search_stage_20261009.py --trace-dir output/experiments/M1_FUSION_SIGNAL_EXPANSION_DIAG_20261012/trace_run/trace --output-dir output/experiments/M1_FUSION_SIGNAL_EXPANSION_DIAG_20261012/feature_capture --no-perturb --probe-reference-frames 2 --selection-rule onset-3-iou-0.5 --diagnostic-id M1-FUSION-SIGNAL-EXPANSION-DIAG-20261012 --capture-template-tokens
python3.8 tools/analyze_m1_fusion_signal_admission_20261012.py --features output/experiments/M1_FUSION_SIGNAL_EXPANSION_DIAG_20261012/feature_capture/selected_features.npz --center-csv output/experiments/M1_FUSION_SIGNAL_EXPANSION_DIAG_20261012/feature_capture/per_arm.csv --output-dir output/experiments/M1_FUSION_SIGNAL_EXPANSION_DIAG_20261012/analysis_new8
python3.8 tools/merge_m1_fusion_signal_expansion_20261012.py --base-features output/experiments/M1_FUSION_SIGNAL_ADMISSION_DIAG_20261012/feature_capture/selected_features.npz --base-center-csv output/experiments/M1_FUSION_SIGNAL_ADMISSION_DIAG_20261012/feature_capture/per_arm.csv --extension-features output/experiments/M1_FUSION_SIGNAL_EXPANSION_DIAG_20261012/feature_capture/selected_features.npz --extension-center-csv output/experiments/M1_FUSION_SIGNAL_EXPANSION_DIAG_20261012/feature_capture/per_arm.csv --extension-sequence-list experiments/diagnostics/m1_fusion_signal_expansion_20261012/primary_sequence_list.txt --output-dir output/experiments/M1_FUSION_SIGNAL_EXPANSION_DIAG_20261012/combined_feature_capture_17
python3.8 tools/analyze_m1_fusion_signal_admission_20261012.py --features output/experiments/M1_FUSION_SIGNAL_EXPANSION_DIAG_20261012/combined_feature_capture_17/selected_features.npz --center-csv output/experiments/M1_FUSION_SIGNAL_EXPANSION_DIAG_20261012/combined_feature_capture_17/per_arm.csv --output-dir output/experiments/M1_FUSION_SIGNAL_EXPANSION_DIAG_20261012/analysis_combined17_v2
python3.8 tools/merge_m1_fusion_signal_expansion_20261012.py --base-features output/experiments/M1_FUSION_SIGNAL_ADMISSION_DIAG_20261012/feature_capture/selected_features.npz --base-center-csv output/experiments/M1_FUSION_SIGNAL_ADMISSION_DIAG_20261012/feature_capture/per_arm.csv --extension-features output/experiments/M1_FUSION_SIGNAL_EXPANSION_DIAG_20261012/feature_capture/selected_features.npz --extension-center-csv output/experiments/M1_FUSION_SIGNAL_EXPANSION_DIAG_20261012/feature_capture/per_arm.csv --output-dir output/experiments/M1_FUSION_SIGNAL_EXPANSION_DIAG_20261012/combined_feature_capture
python3.8 tools/analyze_m1_fusion_signal_admission_20261012.py --features output/experiments/M1_FUSION_SIGNAL_EXPANSION_DIAG_20261012/combined_feature_capture/selected_features.npz --center-csv output/experiments/M1_FUSION_SIGNAL_EXPANSION_DIAG_20261012/combined_feature_capture/per_arm.csv --output-dir output/experiments/M1_FUSION_SIGNAL_EXPANSION_DIAG_20261012/analysis_combined20
```
