# G0 Cscsu 越闸探索性实验分析

日期：2026-10-05  
Parent：`G0-CPLAIN-CLOSEDLOOP-STOP-20261005`  
性质：明确登记的单次越闸探索，不是已获准改进方案，不是论文主结果  
最终决策：`CSCSU_STOP_NO_GO`

## 1. 研究问题与边界

本实验只回答一个问题：在普通 Cplain 早期 search 旁路已经出现广泛退化的情况下，把给定的 SCSU 放入早期 RGB/TIR search 旁路，是否能在同一写入位置、同一训练预算和同一开发序列上减少退化。

固定对照：

- `C0`：原 M1，冻结；
- `Cplain`：early search -> Linear projection -> zero-gated residual -> location 9 search；
- `Cscsu`：early search -> 每模态独立 SCSU -> 同一 Linear projection -> 同一 zero-gated residual -> 同一 location 9 search；
- `Cscsu-gamma0`：同一个训练后 Cscsu checkpoint，两个 gate 强制为零。

没有改变原 M1 的模板、MHSA、TBSI、head、crop 或预测流程；没有加入新 loss、位置编码、动态 gate 或第二个 seed；没有运行 full-244。

## 2. SCSU 接口与实现

用户提供的 `SpatialChannelSelection` 已保存为 `lib/models/layers/scsu.py`。每个模态的 early token 为 `[B,256,768]`，实验中重排为 `[B,768,16,16]`，通过 SCSU 后再恢复为 token，保持旁路输入输出合同。

SCSU 使用：

- `k1=3`、`k2=5`；
- 两个半通道 SCSU 分支；
- 深度可分离 square/horizontal-strip/vertical-strip 卷积；
- 自适应平均池化生成三分支选择权重；
- 最后 1x1 projection；
- RGB/TIR 各使用同一个旁路结构的独立参数。

旁路可训练参数总数：`2,691,074`。M1 epoch-15 checkpoint：`1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`。

## 3. P0 合同审计

P0 synthetic 输入为 `[2,256,768]`，对应 `16x16` search grid。

| 检查 | 结果 |
|---|---:|
| shape/finite forward | 通过 |
| SCSU enabled 输出有限 | 通过 |
| gamma=0 最大差 | `0.0` |
| 旁路参数梯度 | 非零且有限 |
| SCSU、projection、gate 梯度 | 全部可见 |
| P0 | `PASS` |

证据：[p0_result.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_cscsu_20261005/p0_result.json)

第一次 runner 启动因新配置漏写 `TBSI_DROP_PATH` 三层列表而在模型构建时失败，没有产生训练或 tracking 指标；该失败作为实现/配置启动失败保留在 runner log，修复后才开始正式一次短程运行。

## 4. 冻结状态审计

沿用修复后的 Cplain-only 状态冻结逻辑：冻结的 M1 模块保持 `eval()`，只有 early residual 旁路处于 train mode。训练后 checkpoint 与原 M1 checkpoint 对所有非 `early_search_residual` 状态逐项比较：

- 非旁路参数和 buffer 差异数：`0`；
- 最大非旁路绝对差：`0.0`；
- 状态 parity：通过。

证据：[state_parity.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_cscsu_20261005/state_parity.json)

训练后 gate：

- `gamma_rgb = -0.0137533`；
- `gamma_tir = +0.0156001`。

gate 非零说明旁路被优化，但不代表闭环有效。

## 5. 闭环结果

协议：152 条固定训练序列、24 条与训练不相交的开发评估序列、1 epoch、seed 42、FP32 tracker。

| 条件 | Mean AUC | Mean IoU |
|---|---:|---:|
| C0 | 0.718779 | 0.720613 |
| Cplain | 0.719125 | 0.720971 |
| Cscsu | 0.667146 | 0.668144 |
| Cscsu-gamma0 | 0.718779 | 0.720613 |

### Cscsu 相对 C0

- Mean ΔAUC：`-0.051633`
- Mean ΔIoU：`-0.052469`
- AUC 胜负：`10 win / 14 loss / 0 tie`
- AUC 下降至少 `0.05`：4 条
- IoU 下降至少 `0.05`：4 条
- 去掉最大获益序列后的 Mean ΔAUC：`-0.054695`

最严重的 Cscsu 相对 C0 退化：

| 序列 | ΔAUC | ΔIoU |
|---|---:|---:|
| `blkboywillstand` | -0.791668 | -0.804211 |
| `left4thgirlwithwhitepants` | -0.264256 | -0.268659 |
| `1strowleft3rdgirl` | -0.094730 | -0.096253 |
| `rightbhindbike-gai` | -0.067738 | -0.069290 |
| `boybackpack` | -0.042971 | -0.043500 |

### Cscsu 相对 Cplain

- Mean ΔAUC：`-0.051979`
- Mean ΔIoU：`-0.052826`
- 序列胜负：`12 win / 12 loss`
- 相对 Cplain 下降至少 `0.05`：3 条
- 去掉最大差异序列后的均值仍为明显负值，约 `-0.019817` AUC。

因此 SCSU 没有救回普通 Cplain 的广泛退化，反而引入了更严重的闭环损失。

## 6. gamma=0 完整 tracker parity

Cscsu-gamma0 与 C0 比较全部 24 条序列：

- 24/24 序列长度一致；
- 最大逐框绝对框差：`0.0`；
- parity：通过。

这排除了“关闭 residual 时 SCSU 改坏 M1”的解释。问题发生在 residual 开启、SCSU 参与实际预测之后。

## 7. 决策

```text
CSCSU_STOP_NO_GO
```

原因：

1. P0 算子合同通过；
2. M1 参数和 buffer 冻结合同通过；
3. gamma=0 完整 tracker parity 通过；
4. Cscsu 相对 C0 的 AUC/IoU 均显著下降；
5. Cscsu 相对 Cplain 也下降约 `0.052` AUC；
6. 去掉最大获益或最大异常序列后，结论仍为负；
7. 存在 `-0.791668` 的单序列灾难性退化。

本轮不启动第二 seed、不做 full-244、不继续更换 SCSU 超参数或旁路位置。该结果支持封存“当前 SCSU 早期旁路接法”，不能推出所有 SCSU 变体必然无效。

## 8. 证据路径

- P0：[p0_result.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_cscsu_20261005/p0_result.json)
- 状态 parity：[state_parity.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_cscsu_20261005/state_parity.json)
- 四方汇总：[closed_loop_summary.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_cscsu_20261005/closed_loop_summary.json)
- C0：[TBSI_E_MHSA_FULL_4090D_seed42_g0_cscsu_c0](/root/autodl-tmp/TBSI_gai/TBSI/output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42_g0_cscsu_c0/)
- Cplain：[G0_CPLAIN_DIAGNOSTIC_seed42_g0_cscsu_cplain](/root/autodl-tmp/TBSI_gai/TBSI/output/test/tracking_results/tbsi_track/G0_CPLAIN_DIAGNOSTIC_seed42_g0_cscsu_cplain/)
- Cscsu：[G0_CSCSU_DIAGNOSTIC_seed42_g0_cscsu_trained](/root/autodl-tmp/TBSI_gai/TBSI/output/test/tracking_results/tbsi_track/G0_CSCSU_DIAGNOSTIC_seed42_g0_cscsu_trained/)
- gamma=0：[G0_CSCSU_GAMMA0_seed42_g0_cscsu_gamma0](/root/autodl-tmp/TBSI_gai/TBSI/output/test/tracking_results/tbsi_track/G0_CSCSU_GAMMA0_seed42_g0_cscsu_gamma0/)
- runner：[run_g0_cscsu.sh](/root/autodl-tmp/TBSI_gai/TBSI/tools/run_g0_cscsu.sh)
- analyzer：[g0_cscsu_analyze.py](/root/autodl-tmp/TBSI_gai/TBSI/tools/g0_cscsu_analyze.py)

## 9. 失败归因审计：`blkboywillstand`

本节为不训练的只读审计，不改变 `CSCSU_STOP_NO_GO`。对象是 Cscsu 相对 C0 退化最大的 `blkboywillstand`，使用同一 FP32 tracker、同一 crop 规则和同一 epoch-1 Cscsu checkpoint。

逐帧结果显示，首次预测框最大差超过 1 px 出现在 frame 14；此时 C0/Cscsu IoU 为 `0.543663/0.539930`，目标中心仍在 crop 内，RGB/TIR residual 与 late 特征范数比为 `0.007139/0.014148`。首次框差超过 5 px 出现在 frame 15。首次 IoU 下降超过 0.05 出现在 frame 25：C0 IoU=`0.763734`，Cscsu IoU=`0.0`，但目标中心仍在该帧 crop 内。该顺序支持“早期预测分叉后发生闭环漂移”的解释；不能把后续 335 个 crop 外帧当作最初原因。

全序列 residual/late 范数比仍较小（RGB mean/max=`0.007672/0.009010`，TIR mean/max=`0.010866/0.015901`），因此目前证据不能把退化简单归因于残差数值爆炸。逐帧证据：[summary.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_cscsu_20261005/failure_audit/summary.json)、[frame_rows.jsonl](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_cscsu_20261005/failure_audit/frame_rows.jsonl)。

## 10. Gate 缩放只读干预

四个 checkpoint 均从正确的训练后 Cscsu epoch-1 checkpoint 派生，只修改 RGB/TIR 两个 gamma；没有训练或修改原 checkpoint。第一次派生曾误用了初始化 checkpoint，四个文件 gate 全为零，所得推理结果已标记无效并排除；随后已用正确源 checkpoint 重建并重新推理。

| gate 倍率 | AUC | Mean IoU | ΔAUC vs C0 | IoU<0.5 帧 | 首次框差>1 px |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.843696 | 0.847106 | 0.000000 | 15 | 无 |
| 0.25 | 0.051096 | 0.041944 | -0.792600 | 346 | 14 |
| 0.50 | 0.051562 | 0.042408 | -0.792134 | 346 | 13 |
| 1.00 | 0.052028 | 0.042895 | -0.791668 | 346 | 14 |

在该失败序列上，缩小 gate 没有消除灾难性退化；三种非零倍率均在相近的早期帧分叉并进入漂移。这个结果只说明本 checkpoint 的问题不是简单的 gate 幅度过大，不能据此选择新的倍率，也不能外推为 full-244 结论。证据目录：[valid gamma tracking results](/root/autodl-tmp/TBSI_gai/TBSI/output/test/tracking_results/tbsi_track/)。

## 11. SCSU 16x16 padding 结构审计

实现中的 `strip_k = 3*k + 2`，所以 `k2=5` 产生长度为 17 的水平/垂直条带卷积，而 search 特征网格为 `16x16`。对单轴 16 位置统计：边缘位置只有 `9/17` 个真实网格输入，中心最好为 `16/17`；平均有效覆盖为 `12.5/17 = 73.53%`。每个输出位置至少有一个 kernel slot 使用 zero padding。该结构失配是合理的 L3 位置/尺度风险假设，但不是本次退化的已证实因果原因；本轮不修改 `k2` 或重训。

统计脚本：[g0_cscsu_padding_audit.py](/root/autodl-tmp/TBSI_gai/TBSI/tools/g0_cscsu_padding_audit.py)，结果：[scscu_padding_audit.json](/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/g0_cscsu_20261005/failure_audit/scscu_padding_audit.json)。

## 12. 审计结论

失败归因审计没有找到足以支持继续迭代的证据：首个明显闭环分叉发生在目标仍可见且 residual/late 范数比不大的阶段；gate 缩放未恢复该序列；`k2=5` 的长条带确有 16x16 边界 padding 风险，但尚未被因果验证。因此维持：

```text
CSCSU_STOP_NO_GO
```

不启动 full training、full-244、第二 seed、kernel 修改或新的 gate/loss 搜索。
