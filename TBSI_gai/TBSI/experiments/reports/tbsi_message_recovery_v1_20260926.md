# TBSI Message Recovery v1：RGB 模糊路径归因

日期：2026-09-26  
状态：`DIAGNOSTIC`；不启动训练

## 归因位置与协议

代码审计确认当前顺序为：TIR search -> fused template -> RGB search，随后 RGB search -> fused template -> TIR search。因此 RGB 模糊待归因的模板消息是第一层 `tbsi_layers[0].ca_s2t_v2f.attn_reshape` 输出的前 64 个 template tokens；返回张量形状为 `[B,320,C]`，后 256 个 search tokens 始终保留模糊运行的原值。

每个事件使用同一干净 tracker 状态、模板、时间缓存和 CPU/CUDA RNG 分叉，运行：`F`、`B`、`B+RGB消息恢复`、`B+原样消息hook`、`B+TIR路径对照`、`B+匹配随机消息对照`。clean message 只作测试时机制探针，不是可部署方法。消息恢复量为 `R_e=IoU(B+恢复)-IoU(B)`；恢复比例只对原始 blur loss 非零事件计算。

## 审计门槛

8 事件 smoke 和 72 事件 batch 均通过 raw-hook parity：raw hook 与 B 的预测 hash/IoU 完全一致；hook 调用次数和形状一致；目标输出为 `[1,320,768]`，只替换前 64 个 tokens。完整结果为 2,592 行 = 72 事件 × 6 条件 × 6 offsets。

## 结果

| 指标 | offset 0 均值 | 序列 bootstrap 95% CI | offset 1--5 均值 | 序列 bootstrap 95% CI |
|---|---:|---:|---:|---:|
| 原始 blur loss | 0.11261 | [0.08128, 0.14797] | 0.00920 | [0.00310, 0.01658] |
| `R_e` RGB 消息恢复 | 0.000124 | [-0.00000365, 0.00026590] | 0.000788 | [-0.000239, 0.001851] |
| TIR 路径对照 | 0.000002 | [-0.00000406, 0.00000815] | -0.000286 | [-0.000739, 0.000005] |
| 匹配随机消息 | -0.000125 | [-0.000563, 0.000292] | 0.000208 | [-0.001442, 0.001934] |
| `R_e - random` | 0.000249 | [-0.000140, 0.000694] | 0.000580 | [-0.001144, 0.002390] |
| `R_e - TIR` | 0.000122 | [-0.000006, 0.000262] | 0.001074 | [-0.000013, 0.002240] |

offset 0 的 `R_e` 中位数为 0；去掉最大 3 个事件后均值为 `0.000108`。恢复比例的 offset 0 中位数约 `0.00106`，序列区间跨零，且比例对接近零的原始 loss 敏感。

## 结论与路由

新序列复核确认强 RGB 模糊的当帧损失是可复现现象，但第一层 RGB search -> fused template 的 clean-message 恢复没有稳定恢复 IoU，也没有相对随机消息对照的稳定优势。因此本轮不支持“该模板消息路径承载了主要 RGB 模糊损失”的机制主张。

`B+TIR路径对照` 接近零与预期一致，但这不足以挽救 RGB 路径归因；最终跟踪框也不能单独证明 TIR 分支定位变差。停止该消息恢复方向的训练计划，不新增注意力模块，不调整 blur 强度。后续若继续当帧问题，应先审计其他明确路径或转向真实低质量视频验证，并保持诊断/方法边界分离。

## 证据与复现

- Smoke：`output/experiments/tbsi_message_recovery_v1/smoke/`。
- Batch：`output/experiments/tbsi_message_recovery_v1/batch/`。
- 原始结果：`message_recovery_event_results.csv`；离线汇总：`analysis_v1/analysis.json`。
- 脚本：`tools/tbsi_message_recovery_v1.py`、`tools/analyze_tbsi_message_recovery_v1.py`。
- checkpoint/config 仅只读使用；canonical baseline YAML 未修改。环境为 1× RTX 4090 D、Python 3.8.20、PyTorch 1.9.0+cu111、FP32 tracker inference。
