# 废弃实验权重清理记录

本次删除 6 个已停止或已否定路线的 checkpoint，共约 1.3 GiB。保留所有日志、指标、逐序列结果、配置和报告作为审计证据。

删除项：

- `M1_CENTER_RANK_LOSS_PROXY20_20261016/checkpoints/TBSITrack_ep0003.pth.tar`，全 244 负结果。
- `M1_IDENTITY_CALIBRATION_V2_20261010/results/calibrated_ranker.pth`，校准候选已停止。
- `M1_JOINT_LOCAL_SIGNAL_DECISION_20261013/model_artifacts_retry2/s1_response_heads.pth`，信号未准入。
- `M1_SINGLE_SIGNAL_LOW_AUC_20261012/train72_screen_v1/s1_response_heads.pth`，代理筛选证据已完成。
- `tam_trace_smoke_v1_20261007/tam_smoke_checkpoint.pth`，仅训练合同 smoke。
- `tam_online_replay_train_audit_v1_20261007/tam_image_replay_smoke_checkpoint.pth`，仅回放合同 smoke。

保留：受保护 M1、4090D 等效复现组和当前 IGF proxy 的 checkpoint。删除前 SHA256 已记录在终端审计中，资产表和本报告保留路径与结果信息。

## 第二批清理（2026-10-17）

为扩大 E2 验证释放磁盘空间，删除已完成否定或仅用于筛选的 AMCM checkpoint；配置、日志、指标和报告保留：

- `M1_AMCM_E1_POST_FUSION_PROXY_20261017/checkpoints/*.pth.tar`
- `M1_AMCM_E3_JOINT_FUSION_PROXY_20261017/checkpoints/*.pth.tar`
- `M1_AMCM_E4_JOINT_HEAD_PROXY_20261017/checkpoints/*.pth.tar`
- `M1_AMCM_LITE_RELIABILITY_PROXY_20261017/checkpoints/*.pth.tar`
- `M1_AMCM_LITE_RELIABILITY_PROXY_20261017_BNDRIFT/checkpoints/*.pth.tar`

保留 `M1_AMCM_E2_TARGET_SUPERVISED_PROXY_20261017`、正在运行的扩大 E2、M1、4090D 等效复现组和 IGF proxy checkpoint。

随后删除原始 E2 proxy 的中间 `TBSITrack_ep0001.pth.tar` 和 `TBSITrack_ep0002.pth.tar`，保留其最终 `ep0003` 作为 difficult5 结果证据，并保留扩大实验的 `ep0005`。
