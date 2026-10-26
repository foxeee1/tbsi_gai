# 权重清理记录

按用户授权删除代理实验和小实验组 checkpoint；未删除 canonical baseline、M1、v1.0.1/v1.0.3 full 实验或固定预训练权重。

删除对象：

- `output/experiments/BTP_RPP_v1_b16_seed42/checkpoints/TBSITrack_ep0015.pth.tar`
- `output/experiments/BTP_RPP_v1_b16_seed3407/checkpoints/TBSITrack_ep0015.pth.tar`
- `output/experiments/BTP_PRUNE_ONLY_M2_seed42/checkpoints/TBSITrack_ep0015.pth.tar`
- `output/experiments/BTP_PRUNE_ONLY_M3_seed42/checkpoints/TBSITrack_ep0015.pth.tar`
- `output/template_expert_p1_retry/experiments/TEMPLATE_EXPERT_C_P1_seed42/checkpoints/TBSITrack_ep0003.pth.tar`
- `output/template_expert_p1_retry/experiments/TEMPLATE_EXPERT_G_P1_seed42/checkpoints/TBSITrack_ep0003.pth.tar`
- `output/template_expert_p1_retry/experiments/TEMPLATE_EXPERT_T_P1_seed42/checkpoints/TBSITrack_ep0003.pth.tar`
- `output/template_expert_p2_retry_20260928/experiments/TEMPLATE_EXPERT_DISPONLY_P2_seed42/checkpoints/TBSITrack_ep0003.pth.tar`
- `output/template_expert_p2_retry_20260928/experiments/TEMPLATE_EXPERT_SOFT_P2_seed42/checkpoints/TBSITrack_ep0003.pth.tar`

释放空间约 13.4 GiB；删除前可用空间约 2.9 GiB，删除后约 16 GiB。历史日志、指标和结果目录保留用于审计，权重不再可恢复。
