# M1 废弃实验权重清理

## 删除范围

仅删除已判定为 `NOT-PROMOTED` 或已停止路线的 `.pth.tar`：4 个 PVL 5-epoch checkpoint，以及 6 个 TargetRankGate/CenterRank diagnostic checkpoint。删除前已核对这些文件不属于 M1 受保护 checkpoint、4090D 等效复现组或当前 global/foreground/adaptive 对照组。

配置、日志、报告、CSV、manifest 和诊断结果均保留。

## 保留范围

- M1 `TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar`；
- 4090D 等效复现组 `v1.0.1-单卡4090D-24G-等效全局128加速`；
- foreground-only、mix50、adaptive 三个当前 AMCM 对照 checkpoint。

清理后 checkpoint 文件总数为 5。受保护资产未修改。

状态：`MAINTENANCE-COMPLETE`。adaptive full-244 尚未运行。
