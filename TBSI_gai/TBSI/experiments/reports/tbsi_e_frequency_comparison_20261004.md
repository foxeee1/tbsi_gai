# TBSI-E 后频域改进实验结果对比分析

日期：2026-10-04  
Parent：`TBSI-E-MHSA-FULL-20260927`   
代码快照：`09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9` + working tree  
评测口径：当前仓库 LasHeR 244 条测试序列，15 epochs，seed 42，serial FP32 tracker，`TBSI_INFER_FP16=0`。这不是 canonical baseline 复现；canonical baseline 仍保持独立且未修改。

## 1. Full-244 结果

| 实验 | 主要改动 | AUC | OP50 | OP75 | PR | NPR | 相对 M1 AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| TBSI-E M1 | template-only joint MHSA | 56.21 | 68.33 | 47.13 | 70.21 | 66.20 | -- |
| CFVI P1 | location 9 固定 Haar 四频带 Value 注入 | 56.20 | 68.38 | 47.01 | 70.39 | 66.43 | -0.01 |
| TAFH | location 9 目标锚定 LL 地址与高频交换 | 55.19 | 67.12 | 46.40 | 68.81 | 65.18 | -1.02 |
| FMR | location 9 projected message 高频残差细化 | 55.91 | 67.93 | 46.92 | 69.81 | 66.08 | -0.30 |
| HF-Relation R0 | location 9 跨模态 MHSA logits 高频关系 bias | 56.47 | 68.56 | 47.49 | 70.51 | 66.62 | +0.26 |
| R2 Scale-Calibrated | location 9 原生 logit 尺度校准的频率关系 bias | 56.17 | 68.51 | 47.06 | 70.34 | 66.46 | -0.04 |

相对 M1 的完整差值（AUC / OP50 / OP75 / PR / NPR）为：CFVI `-0.01 / +0.05 / -0.12 / +0.18 / +0.23`；TAFH `-1.02 / -1.21 / -0.73 / -1.40 / -1.02`；FMR `-0.30 / -0.40 / -0.21 / -0.40 / -0.12`；HF-Relation `+0.26 / +0.23 / +0.36 / +0.30 / +0.42`。

## 2. 配对证据与机制边界

HF-Relation 是当前唯一五项指标同时为正的 full-244 候选，但其逐序列 AUC 均值为 `+0.2634`，中位数 `+0.0128`，10% trimmed mean `-0.0720`，胜/平/负为 `122/1/121`，bootstrap 95% CI 为 `[-0.8991, +1.4704]`，且有 21 条序列的 AUC 退化不低于 5 个百分点。该结果应标为 `VALID-FULL-MATCHED-PROVISIONAL`，不能称为稳定增益。

CFVI 的 paired mean 为 `+0.000214`，median `-0.001110`，胜率 `47.95%`，bootstrap 95% CI `[-0.012137,+0.012972]`，与 M1 基本持平。源频带能量 LL 占主导，且没有目标/背景 mask 或跨模态局部对应约束，因此不能据此声称某一频带有效。

TAFH 通过了 Haar、zero-gamma、梯度和局部寻址 P0，但 full 结果五项全部下降；epoch-15 注入比例仅约 RGB `0.001484`、TIR `0.000537`。gamma=0 只读审计相对独立 M1 仍为 AUC `-0.35`，所以已学习 residual 是主要额外损失的证据仍不足以排除训练轨迹或合同差异。

FMR 分支确认有学习参数和非零注入，但 full 结果 AUC `-0.30`，预注册的 `AUC >= 57.21` 门槛未通过，停止第二 seed、动态 gate 和新的频率模块。

HF-Relation 的 R0 机制审计在 24 个固定序列、48 个事件上发现 learned bias 虽数值存在，但 normal 相对 beta0 的平均 IoU 差异仅约 `+2.18e-6`，shuffle 为 `-2.30e-6`，sign-flip 为 `+0.30e-6`，未显示可辨识的频率使用效应。因此 R1 full training 按预注册 gate 停止。

R2 的固定尺度审计满足预注册的可观测性门槛：alpha=0.10 的 median attention L1/KL/message relative delta 分别为 `6.730e-4 / 9.547e-3 / 3.190e-2`，均超过 `1e-3 / 1e-5 / 1e-4` 中相应门槛的要求（attention L1 以 alpha=0.20 达到 `1.358e-3`）。因此 runner 按预注册规则完成了唯一一次 full-244。可是 alpha=0.10 的 shuffle 条件几乎保留同等扰动幅度（median `6.710e-4 / 9.673e-3 / 3.184e-2`），未显示关系特异性的削弱；sign-flip 使事件平均 IoU 从 `+2.834e-5` 反转为 `-3.070e-5`，但 attention/message 幅度仍近似不变。R2 full 结果 AUC `56.17`，相对 M1 `-0.04`，因此不能把尺度校准解释为有效跟踪改进。

## 3. 结论与后续边界

当前证据支持的结论是：频域分支可以接通并产生可测算子差异，但 CFVI、TAFH、FMR 均未在匹配 full-244 单 seed 中带来增益；HF-Relation 的小幅正差值尚未通过逐序列稳定性和机制使用验证。频域方向暂不应继续堆叠模块或宣称贡献。

R2 已完成预注册尺度审计和一次 full-244 开发性验证；结果不支持继续第二 seed、动态 gate 或新的频率模块。代理-60、48-event 诊断和属性分组只用于筛选/机制判断，不能替代 full confirmation 或论文主结果。

## 4. 证据路径

- Parent：`experiments/reports/tbsi_e_mhsa_full_20260927.md`
- CFVI：`experiments/reports/cfvi_p1_full_matched_m1_20261001.md`、`experiments/reports/cfvi_p1_mechanism_redefinition_audit_20261001.md`
- TAFH：`experiments/reports/tafh_full_matched_m1_analysis_20261002.md`、`experiments/reports/tafh_mechanism_audit_20261002.md`
- FMR：`experiments/reports/fmr_full_matched_m1_20261003.md`
- HF-Relation：`experiments/reports/hf_relation_loc9_full_matched_m1_20261003.md`、`experiments/reports/hf_relation_r1_mechanism_audit_20261003.md`
- 频率诊断：`output/experiments/freq_guidance_v1/response_signed/response_summary.json`、`output/experiments/frcb_admission_v1/admission_summary.json`
- R2：`experiments/reports/r2_scale_calibrated_plan_20261004.md`、`output/experiments/r2_scale_p0_20261004.json`
- R2 审计：`output/experiments/r2_scale_audit_v1_20261004/scale_summary.json`、`output/experiments/r2_scale_audit_v1_20261004/scale_events.csv`
- R2 full pipeline：`output/experiments/r2_scale_calibrated_full_20261004/`、`output/experiments/R2_SCALE_CALIBRATED_LOC9_seed42/checkpoints/TBSITrack_ep0015.pth.tar`
