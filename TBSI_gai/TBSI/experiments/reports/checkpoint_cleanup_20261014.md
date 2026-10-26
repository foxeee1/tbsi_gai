# 旧实验权重清理记录

日期：2026-10-14

清理了已由 V2 替代且已判无效的 router P0 V1 权重：

- 删除前路径：`output/experiments/M1_ADAPTIVE_POST_ROUTER_P0_QUICK_20261013/experiments/M1_ADAPTIVE_POST_ROUTER_P0_QUICK_seed42/checkpoints/TBSITrack_ep0001.pth.tar`
- 大小：444,078,192 bytes
- SHA256：`d019a9c7d76ca43b4407a28966263c5a6fb3ccb6b3120feec3091a448fbf4cae`
- 原因：V1 因冻结 M1 时 BN buffer 漂移而失效；资产报告记录的有效 V2 checkpoint 仍保留，V1 的训练日志、梯度与内部信号 CSV 仍保留。
- 磁盘可用空间由约 7.8 GB 增至约 8.2 GB。

M1 checkpoint、4090D 复现实验组 checkpoint、已完成正式结果的 Cdiag/ABC 权重、有效 P0 V2 权重及 negative20 router 权重均保留。没有删除仍承担当前比较或登记证据职责的资产。
