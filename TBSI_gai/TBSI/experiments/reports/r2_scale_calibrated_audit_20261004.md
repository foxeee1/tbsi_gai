# R2 Scale-Calibrated Frequency Relation Audit

日期：2026-10-04  
实验：`R2_SCALE_CALIBRATED_LOC9_seed42`  
Parent：`HF-RELATION-R0-AUDIT-20261003` / TBSI-E M1  
代码：`09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9` + working tree  
硬件：1× RTX4090D 24GB；Python 3.8.20；PyTorch 1.9.0+cu111；CUDA 11.1；FP32 tracker

## 审计合同

固定开发集为 24 条 LasHeR train 序列、48 个事件。R2 只改变 location 9 跨模态频率关系 bias 的尺度校准，不增加 Value 注入、message residual、可靠性 gate、target mask 或第二套 attention。审计条件固定为 `alpha=0/0.05/0.10/0.20`，并在 `alpha=0.10` 下执行 `shuffle` 和 `sign_flip` 对照。

预注册的继续门槛为至少一个非零 alpha 同时满足：median attention L1 > `1e-3`、median attention KL > `1e-5`、median message relative delta > `1e-4`。该门槛由 runner 在训练前自动判断。

## 机制审计结果

| 条件 | median attention L1 | median attention KL | median message relative delta | mean event delta IoU |
|---|---:|---:|---:|---:|
| R0 normal | 3.762e-6 | 4.074e-7 | 2.256e-4 | -2.177e-6 |
| R2 alpha 0 | 0 | 0 | 0 | 0 |
| R2 alpha 0.05 | 3.354e-4 | 2.382e-3 | 1.591e-2 | 7.866e-6 |
| R2 alpha 0.10 | 6.730e-4 | 9.547e-3 | 3.190e-2 | 2.834e-5 |
| R2 alpha 0.20 | 1.358e-3 | 3.842e-2 | 6.436e-2 | 6.158e-5 |
| alpha 0.10 shuffle | 6.710e-4 | 9.673e-3 | 3.184e-2 | 2.658e-5 |
| alpha 0.10 sign-flip | 6.690e-4 | 9.483e-3 | 3.168e-2 | -3.070e-5 |

尺度可观测性 gate 通过：alpha 0.20 的 attention L1、KL 和 message delta 均超过门槛，runner 因此按预注册规则继续了唯一一次 full-244 训练。可是条件特异性没有完整通过：shuffle 几乎保留了 alpha 0.10 的 attention/message 扰动幅度，未表现出“正确对应关系被打乱后明显减弱”；sign-flip 反转了平均事件 IoU 方向，但没有显著消除 attention/message 扰动本身。结论是“尺度校准使算子可观测”，不是“已证明模型使用了正确频率对应关系”。

## Full-244 结果

训练 15 epochs，SOT pretrained initialization，serial FP32 test，244/244 sequences 完成分析。

| 指标 | R2 | TBSI-E M1 | R2 - M1 |
|---|---:|---:|---:|
| AUC | 56.17 | 56.21 | -0.04 |
| OP50 | 68.51 | 68.33 | +0.18 |
| OP75 | 47.06 | 47.13 | -0.07 |
| Precision | 70.34 | 70.21 | +0.13 |
| Norm Precision | 66.46 | 66.20 | +0.26 |

R2 没有形成 AUC 增益，且仅为单 seed、已参与方案筛选的 full-244 开发性验证；不能作为最终泛化确认或论文主结果。

训练日志在早期出现过一次 `clip_grad_norm_` 的 non-finite norm warning，训练过程未中断并最终完成 15 epochs；该 warning 作为 provenance 风险保留，不据此把结果升级为稳定方法结论。

## 决定

状态：`VALID-FULL-MATCHED-NEGATIVE`。

停止 R2 第二 seed、动态 gate、频率模块扩展和新的 full training。频域支线保留为已完成的机制诊断和负向开发结果；后续论文主线转向更具体的融合问题。R2 P0、尺度审计和 full pipeline 的证据均保留，不覆盖 canonical baseline。

## Provenance

- Runner：`tools/run_r2_scale_calibrated_serial.sh`
- P0：`output/experiments/r2_scale_p0_20261004.json`
- 审计摘要：`output/experiments/r2_scale_audit_v1_20261004/scale_summary.json`
- 审计逐事件：`output/experiments/r2_scale_audit_v1_20261004/scale_events.csv`
- Pipeline：`output/experiments/r2_scale_calibrated_launcher.log`
- Full output：`output/experiments/r2_scale_calibrated_full_20261004/`
- Checkpoint：`output/experiments/R2_SCALE_CALIBRATED_LOC9_seed42/checkpoints/TBSITrack_ep0015.pth.tar`
- Tracking：`output/test/tracking_results/tbsi_track/R2_SCALE_CALIBRATED_LOC9_seed42/`
- Checkpoint SHA256：`c2ed603f16012aeb54674c1d5ab00e1c396a58b1fe77d544035b4d2c8ec6cdfa`
- Config SHA256：`b85b4eeb54449ab2ff66268102a39559df87b1b574a2a42483f501174ab278a1`
- Canonical baseline YAML SHA256：`4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b`
- 训练日志注意事项：`output/experiments/r2_scale_calibrated_full_20261004/logs/training.log` 记录一次 non-finite gradient norm warning；训练仍完成，需保留为结果限制。
