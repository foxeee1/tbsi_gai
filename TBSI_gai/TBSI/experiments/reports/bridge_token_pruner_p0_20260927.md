# Bridge Token Pruner P0（2026-09-27）

## 目的

验证第一轮固定长度、类型感知 soft token pruning 能否在 TBSI 中安全接入，且不破坏关闭状态的原始 forward。该实验只验证数值与梯度契约，不代表 LasHeR tracking 结果，也不代表真实 FLOPs/FPS 降低。

## 实现与隔离

- 新模块：`lib/models/layers/bridge_token_pruner.py`
- 接入位置：`TBSILayer` 的跨模态桥接前；默认关闭
- 新配置块：`MODEL.BRIDGE_TOKEN_PRUNER`；目标 TBSI location 9 对应 `LAYER_IDS: [2]`
- P0 脚本：`tools/validate_bridge_pruner_p0.py`
- 证据：`output/experiments/bridge_token_pruner_p0/p0_result.json`
- parent：B0/canonical TBSI 结构；本次未修改 canonical YAML、canonical checkpoint 或 canonical 输出目录

## P0 结果

运行环境为 1× RTX 4090 D 24GB，Python 3.8.20，PyTorch 1.9.0+cu111，seed 42。用 `B=2, N=20, C=32, lens_z=4` 的确定性 TBSI layer smoke 检查：

| 检查 | 结果 |
|---|---:|
| pruner disabled 最大绝对差 | 0.0 |
| keep=1.0 identity 最大绝对差 | 0.0 |
| keep=0.75 开启后最大绝对差 | 1.35303354 |
| pruner 参数梯度 | 8/8 非零 |
| pruner 参数量 | 6,658 |
| P0 状态 | PASS |

模型构建回归也通过：`TBSI_LOC=[3,6,9]` 时仅第 3 个 TBSI layer（实际 location 9）启用 pruner。

## 边界与下一步

当前磁盘约 17GB，低于 RPP-v1 正式脚本的 100GB 预留阈值，因此没有启动 proxy train/test，也没有产生 AUC、PR、NPR、OP50、OP75 或 FPS。下一步必须先满足独立 run 目录和磁盘安全条件，再创建新的 pruner RPP 配置，按固定 seed 42/3407、冻结 manifest 和 15 epoch 协议运行；在此之前结论仅为 `P0 PASS / UNCONFIRMED`。

## 运行阻断记录

用户要求的可删除范围内没有可删除 checkpoint：`output/experiments` 仅有 canonical、`v1.0.1` 和 `v1.0.3` 三个约 2.4GB 权重，均按要求保留；`pretrained_models/TBSITrack_SOT.pth.tar` 是固定预训练资产，也未删除。正式 RPP-v1 启动因可用空间约 17GB 小于脚本的 100GB 阈值而被安全门阻断，未产生训练或测试进程。
