# Early gate + learnable template Value reconstruction quick3（2026-10-26）

> **2026-10-26 后续审计更正：** 下方原始指标保留，原“稳定负增益候选”“Early evidence不足”归因与完整preflight通过结论不再适用。外层ViT初始化覆盖了新增up层及普通gate末层的构造器零初始化；preflight公共state拷贝未延续到正式两臂训练；旧重构同时改变self/cross Value；所谓16样本拟合是两份合成输入重复8步拟合教师，非真实GT跟踪过拟合。每臂仅12个optimizer更新边界，且已有同control重复差11.15 pp。当前状态改为 `PROXY-NEGATIVE / INITIALIZATION-AND-PATH-CONFOUNDED / MECHANISM-INCONCLUSIVE`。完整依据与最小配对短训设计见 [信息流与初始化审计](m1_early_gate_bottleneck_audit_20261026.md)。本报告后文原判定作为历史记录保留，不代表当前机制结论。

## 目的与配对合同

该实验检验通过 SOT-origin preflight 的 `EarlyTemplateValueReconstruction` 是否能在已有 Early-conditioned MSHA gate 上改善跨模态 Value 内容。control 与 candidate 都从 `TBSITrack_SOT_Pretrained.pth.tar`、seed42、fresh AdamW/StepLR 开始，联合训练主干和 head；唯一结构变量是 candidate 开启 `EARLY_TEMPLATE_VALUE_RECONSTRUCTION`。两臂均为 LasHeR train `512 samples/epoch × 3 epochs`、batch16、累积8、worker4、AMP、模板更新关闭，测试使用固定 `m1_dcvr_quick7_legacy_20261025.txt` 的 7 序列、FP32、`threads=0`。

## 结果

| 指标 | control | candidate | delta |
|---|---:|---:|---:|
| AUC | 48.5779 | 44.1906 | -4.3874 pp |
| AO | 49.4654 | 45.5285 | -3.9369 pp |
| Precision | 61.4272 | 55.9468 | -5.4804 pp |
| Norm Precision | 59.9351 | 54.1783 | -5.7568 pp |
| OP50 | 60.8934 | 55.0805 | -5.8129 pp |
| OP75 | 38.9574 | 36.4082 | -2.5491 pp |

AUC 序列胜负为 `3/4/0`。逐序列 AUC 变化：`3rdfatboy -24.89 pp`、`4thboywithwhite -1.41 pp`、`ab_bikeoccluded +28.75 pp`、`boydownplatform +0.88 pp`、`leftfarboycomingpicktheball -43.75 pp`、`shinycarcoming -5.73 pp`、`womanback2car +15.44 pp`。

## 信息流与归因

- candidate 的 12 个 reconstruction tensor 确实被纳入联合训练；训练日志每次记录 2/255 个无梯度参数，和 control 的基线现象相同，未发现新增分支整体断图。
- preflight 已证明 zero-init、非零输出 probe、单 batch 梯度、冻结状态和 16 样本拟合均通过；因此 quick3 负结果不能归因为“模块未执行”或“无法学习”。
- 正向收益集中在遮挡/恢复类序列，负向退化集中在 `3rdfatboy` 与 `leftfarboycomingpicktheball`。这说明 source/counterpart 差分重构在普通目标和背景干扰下改变了模板 Value 的方向，但没有稳定改善目标消息；它把同一套重构规则施加到所有 token，没有可靠的场景选择机制。
- 宏观指标、4 个明显负序列和 candidate 的低于 control 的整体结果，排除了当前 adapter 作为 Early gate 的稳定增益候选。7 序列仍是开发筛查，不能外推 full LasHeR。

## 判定

**`PROXY-COMPLETE / NEGATIVE / STOP-CANDIDATE`**。

不进入 full15，也不继续调 rank、`max_delta` 或学习率。该结果与 fixed28 中 receiver-specific 路由未通过相互印证：当前 Early gate 证据没有足够可靠性来驱动额外 Value 内容重构。保留 Early-conditioned MSHA gate 作为主方案，Value reconstruction 作为负结果和合同资产。

证据：`output/experiments/M1_EARLY_GATE_VALUE_RECON_QUICK3_{CONTROL,CANDIDATE}_20261026/`、`experiments/snapshots/m1_early_gate_value_recon_quick3_20261026/analysis/`、`output/experiments/M1_EARLY_GATE_VALUE_RECON_SOT_PREFLIGHT_20261026/preflight.json`。
