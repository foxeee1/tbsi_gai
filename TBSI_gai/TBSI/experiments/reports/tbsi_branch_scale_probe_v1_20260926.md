# TBSI Branch Scale Probe v1

日期：2026-09-26  
状态：`DIAGNOSTIC`；分支缩放归因未支持继续设计质量门控

## 研究问题

在固定的 `single_modality_blur_v2` 事件上，RGB 模糊时直接降低 TBSILayer 的 RGB search 分支权重，是否能够恢复当帧定位性能，同时保持 clean 输入稳定？

## 固定协议

- 72 个事件、24 条固定 train 序列、offset 0--5、RGB 全图 Gaussian blur sigma=3.0。
- 同一 epoch-15 checkpoint、同一 tracker 状态快照和 RNG 快照；未重新选择事件。
- 干预位置为 `TBSILayer.forward(x_v, x_i, lens_z, ...)` 的真实输入边界：`x_v[:, lens_z:]` 为 RGB search，`x_i[:, lens_z:]` 为 TIR search。
- 条件：`F`、`B`、`B-R1.00/0.75/0.50/0.25/0.00`、`B-T0.50`、`F-R1.00/0.50`、`F-T0.50`。
- 仅使用 inference forward hook，不修改权重、模型源码或 canonical YAML。

## 执行与审计

脚本：`tools/tbsi_branch_scale_probe_v1.py`  
Smoke：8 事件，528 行，raw-hook parity `True`，clean `F-R1.00` parity `True`。  
Batch：72 事件，4,752 行，raw-hook parity `True`，0 跳过。  
配置 SHA256：`4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b`。  
Checkpoint SHA256：`df7390978514b4a8e97924a20097cb932314cdf00d6feed47d1dc338b04d7689`。  
环境：1× NVIDIA GeForce RTX 4090 D；Python 3.8.20；PyTorch 1.9.0+cu111；`TBSI_INFER_FP16=0`。

第一次启动因命令遗漏 `TBSI_INFER_FP16=0` 在前向阶段出现 Float/Half 错误，未产生结果；按固定 FP32 语义重启后完成，失败命令和重启命令均可由终端记录重建。

## 结果

| 条件 | offset 0 IoU 均值 | 相对模糊 `B-R1.00` | 相对 clean `F-R1.00` |
|---|---:|---:|---:|
| `B-R1.00` | 0.65650 | 0 | NA |
| `B-R0.75` | 0.62964 | -0.02686 | NA |
| `B-R0.50` | 0.58070 | -0.07580 | NA |
| `B-R0.25` | 0.54244 | -0.11406 | NA |
| `B-R0.00` | 0.53839 | -0.11811 | NA |
| `B-T0.50` | 0.54527 | -0.11123 | NA |
| `F-R1.00` | 0.76911 | NA | 0 |
| `F-R0.50` | 0.65799 | NA | -0.11113 |
| `F-T0.50` | 0.70799 | NA | -0.06112 |

`B-R0.50` 的 offset 1--5 相对差值依次为 `-0.15018、-0.18496、-0.17913、-0.18044、-0.18353`。offset 0 的序列均值提升仅 8/24；去掉最大 3 个事件后，事件均值差为 `-0.09808`，仍为负。

## 结论与路由

本实验反驳了“固定降低 RGB 分支即可恢复 RGB 模糊定位”的当前假设。缩放 RGB 或 TIR 都会损害 clean/blur tracking，且 RGB 缩放没有正向趋势；因此停止固定分支重加权路线，不启动质量感知 gate 训练。该结果只适用于本固定 train 事件集和受控全图模糊诊断，不能替代 full-244 tracking 结果。

证据：`output/experiments/tbsi_branch_scale_probe_v1/smoke/`、`output/experiments/tbsi_branch_scale_probe_v1/batch/`。
