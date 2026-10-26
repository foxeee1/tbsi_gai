# G0-Hard：困难/偏心条件下首次模板桥接后的 RGB/TIR 互补审计

日期：2026-10-05  
Parent：`G0-POST-TBSI-COMPLEMENT-20261005`  
性质：冻结 M1 的一次边界补充诊断；不跑 full training/full-244。  
当前决策：`G0_HARD_CONDITIONAL_PASS_FFM_P0_ONLY`

## 1. 选样合同

在固定的 176 条 LasHeR train 序列清单上按序运行原 M1 FP32 闭环。选样只使用 M1 预测框、GT 和实际 search crop，不使用 RGB/TIR 特征分数、FFM 或任何候选模块结果。

- `rescue_failure`：M1 event-frame IoU `<0.5`，且 GT 目标 token mask 非空；
- `offcenter`：目标中心距实际 crop 中心 `>2` 个 token，且 GT 目标 token mask 非空；
- `easy_centered`：IoU `>=0.5` 且中心距 `<=2` token；
- crop 外事件单独计数，不进入三个主要组；
- 每组每序列只取按时间顺序第一个符合事件，每组最多 48 个事件。

最终每组 48 个事件，均来自 48 条不同序列；部分事件同时属于多个组，共保存 141 个事件。另有 15,996 个 crop-out 帧单独计数，未混入可救援统计。

第一次选样运行误将组名写入去重集合，造成前 3 条序列集中填满上限，标记为无效；修复后重跑。随后一次命令因序列清单路径拼写错误在启动阶段失败，也未产生模型结果。二者均不进入统计。

## 2. 特征和对照

每个事件保存 post-TBSI-0 / post-block-5 的 RGB/TIR search token。除原有对角 token AP 外，新增更接近 FFM 假设的目标邻域取回指标：

- RGB query → TIR key：目标 RGB query 从 TIR 目标邻域而非背景 key 取回的 margin 和 top-5 命中；
- TIR query → RGB key：反向同理；
- `event_shuffle`：同一条件组内固定循环错配另一事件的 TIR；
- `spatial_shift`：同一事件 TIR 沿水平方向固定平移一个 token；
- `XY_only`：search crop 中心位置先验。

GT 只在 forward 完成后生成目标区域标签；未用于 attention、query/key、gate 或 crop 选择。

## 3. 选样覆盖和主要结果

| 组 | 事件/序列 | M1 IoU | center distance | token AP | XY-only AP |
|---|---:|---:|---:|---:|---:|
| rescue_failure | 48/48 | `<0.5` | 目标在 crop 内 | 0.06776 | 0.77450 |
| offcenter | 48/48 | 依选样定义 | `>2` token | 0.06038 | 0.39860 |
| easy_centered | 48/48 | `>=0.5` | `<=2` token | 0.07425 | 0.85040 |

Token AP 本身没有支持正确配对：rescue_failure 中 correct-event shuffle AP 差为 `-0.00931`，offcenter 为 `-0.01550`。因此不能把结果表述为“冻结对角对应已经证明目标定位增益”。

但目标邻域取回指标在 `rescue_failure` 组出现了困难条件特异的正信号：

| 组 / 方向 | correct margin | event shuffle margin | correct-event 差 | correct-spatial 差 |
|---|---:|---:|---:|---:|
| rescue RGB→TIR | 0.02972 | 0.01389 | +0.01583 | +0.00632 |
| rescue TIR→RGB | 0.01679 | 0.00217 | +0.01462 | +0.00399 |
| offcenter RGB→TIR | 0.02491 | 0.01474 | +0.01017 | +0.01277 |
| offcenter TIR→RGB | 0.01190 | 0.01132 | +0.00058 | +0.00625 |

rescue_failure 组按序列 bootstrap 的 correct-event retrieval margin 95% CI 为：

- RGB→TIR：`+0.01583`，CI `[+0.00314,+0.02892]`；
- TIR→RGB：`+0.01462`，CI `[+0.00466,+0.02467]`。

相对 spatial shift 的 95% CI 也为正：RGB→TIR `[+0.00305,+0.01082]`，TIR→RGB `[+0.00087,+0.00698]`。offcenter 组 TIR→RGB 相对 event shuffle 的 CI 跨零，因此不能把偏心组的双向互补说成稳定成立。

## 4. 决策边界

G0-Hard 没有证明 FFM 必然有效，且 token AP 和 XY-only 对照仍然提示强位置/读出偏差。但在真正的 `rescue_failure` 困难组，双向目标邻域取回相对 event shuffle 和 spatial shift 均有跨序列同向信号。这满足“允许一次低成本 P0/短程诊断”的证据门槛，不满足 full training 门槛。

```text
G0_HARD_CONDITIONAL_PASS_FFM_P0_ONLY
```

下一步最多移植双向 CrossAttention 消息算子做 P0 和一次短程 Cdiag/Cspatial 对照；仍需保持 zero-gate、完整 tracker parity、M1 参数/buffer 冻结和困难组闭环审计。若 P0 或短程闭环没有减少 rescue_failure 退化，立即封存 FFM。此时不跑 full-244、不加第二个位置、不改 loss、不调参搜索。

## 5. 证据路径

- 选样结果：[selection.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_post_tbsi_hard_20261005_retry1/selection.json)
- 困难组分析：[hard_analysis.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_post_tbsi_hard_20261005_retry1/hard_analysis.json)
- 选样脚本：[g0_post_tbsi_hard.py](/root/autodl-tmp/TBSI_gai/TBSI/tools/g0_post_tbsi_hard.py)
- 分析脚本：[g0_post_tbsi_hard_analyze.py](/root/autodl-tmp/TBSI_gai/TBSI/tools/g0_post_tbsi_hard_analyze.py)
