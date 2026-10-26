# 废弃代理路线权重清理记录（2026-10-13）

用户授权清理已停止或仅作工程诊断的代理路线 checkpoint。删除前按资产登记逐项核对 SHA256；历史报告、配置、日志、manifest、预测和分析结果保留。

| 文件 | SHA256 | 字节数 | 原因 |
|---|---|---:|---|
| `output/experiments/M1_MCCA_G0_MATCHED_seed42/checkpoints/TBSITrack_ep0003.pth.tar` | `d0fdb062f05217a388d82d6f1eda9523ac588f1f738f9ba754d53f6efc1f5e2f` | 1,321,826,403 | MCCA 路线已停止；matched control 结果已归档 |
| `output/experiments/M1_MCCA_G1_BIDIR_seed42/checkpoints/TBSITrack_ep0003.pth.tar` | `3595f7da6e1634cc2a5063cd8fd6715bb20a88f8dae55c0d2ca48fd51f8db498` | 1,336,067,027 | MCCA 路线已停止；代理正向信号未晋级 |
| `output/experiments/DBF_G1_LEGACY_seed42/checkpoints/TBSITrack_ep0003.pth.tar` | `a864d8d6369979fb07b9a4a1f2814546c75b390cec1e829cbd4c6a206157cd73` | 1,321,826,403 | DBF 路线已停止；仅探索性 paired screen |
| `output/experiments/DBF_G1_VANILLA_WORKER4_seed42/checkpoints/TBSITrack_ep0003.pth.tar` | `5e419c49251a5f1ed2cfbf0deec39c138c96af5dbcc6cad7549c02359fcadbfe` | 1,342,391,443 | DBF worker-4 重试为不可归因的代理筛选 |
| `output/experiments/tam_rpp_engineering_probe_v1_20261007/TBSITrack_M1_ep0015_TAM100.pth.tar` | `76b9f297a20328533ed679028d1e4e8ced6a3b354e470ceb69dca237d8b5ce44` | 1,325,159,889 | TAM 工程探针中止，未完成代理评测 |

合计删除 6,647,271,165 bytes（约 6.19 GiB）。M1 epoch-15、4090D 等效复现组、预训练权重和其他仍在使用的诊断权重未删除。删除后需重新训练才能恢复上述 checkpoint；已归档指标和分析不变。
