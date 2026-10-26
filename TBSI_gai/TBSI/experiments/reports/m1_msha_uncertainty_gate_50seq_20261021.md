# M1 MHSA 不确定性消息门控 50 序列诊断（2026-10-21）

## 结论

结论分类：**NOT-PROMOTED / 暂不训练**。

本次在预先固定的 50 条 LasHeR testingset 序列上，比较冻结 M1 epoch15 的原始 `off` 路径与新的 `message_gate` 路径。每条序列使用首个满足“固定状态下预测 IoU < 0.30 且 GT 完全在 crop 内”的事件；50 条中 26 条满足，24 条没有合法事件，未按结果替换或删除。

`message_gate` 只对模板 MHSA 的跨模态 Value 消息做 uncertainty 衰减，self message、query、原始 residual、CENTER、状态更新均保持不变。它不是训练后的 PVL 权重实验，而是验证“PVL 风格不确定性是否能安全控制 MHSA 跨模态消息”的冻结功能诊断。

在 26 个配对事件上：

- GT-hard-background margin 平均变化：`-0.000266`，14 条上升、12 条下降；
- GT rank 改善 `3/26`，退化 `2/26`，其余不变；
- top-1 从非 GT 变成 GT：`1/26`；
- fixed-state IoU 平均变化 `+0.02227`，但中位数为 `0`，最大单事件增益 `+0.44658`，不能视为稳定收益；
- loc9 query foreground/background 差值：`-0.06303`；
- loc9 participation foreground/background 差值：`-0.05155`。

因此 gate 没有形成目标选择性，且不确定性估计仍偏向背景。50 序列覆盖降低了小样本过拟合风险，但本次仍是固定状态机制筛选，不是完整跟踪性能结论。

## 实验合同

- Parent：受保护 `TBSI_E_MHSA_FULL_4090D_seed42` epoch15。
- 配置与 checkpoint 哈希：`fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`、`1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`。
- 序列清单：`experiments/manifests/m1_msha_uncertainty_gate_50seq_20261021.txt`，SHA256 `a76af65939e9940aa491cb5cd1e2a645cfcb7c2d9ca0a99309656c894940e795`。
- 选择规则：244 条 test trace 按名称排序取前 50 条，先固定名单，再发现事件；不使用 AUC、失败程度或候选结果排序。
- 事件数：请求 50 条，合法固定事件 26 条；固定状态和 hard-background 对照复用同一帧与同一 crop。
- 环境：1× RTX 4090D，FP32，PyTorch 2.0.0+cu118；无训练。
- 不修改 Dataset、划分、Loss、Optimizer、CENTER head 或状态更新。

## 诊断结果

### 不确定性方向

当前 gate 使用模板 RGB/TIR agreement 与搜索 query match 组成 reliability，再对跨模态 Value 消息做逐 token 衰减。50 序列事件上，loc3/6/9 的 participation foreground/background 差值均为负，loc9 最明显：

| 位置 | query foreground 均值 | query background 均值 | participation foreground-background |
|---|---:|---:|---:|
| loc3 | 0.69446 | 0.72419 | -0.03761 |
| loc6 | 0.60088 | 0.62744 | -0.03444 |
| loc9 | 0.65360 | 0.71663 | -0.05155 |

这不是“gate 强度不够”的证据，而是 gate 输入本身缺少目标条件。继续增大或减小 gate 温度只会改变背景偏置的幅度，不能把它变成目标选择器。

### 目标排序变化

14/26 条事件的 margin 有微小上升，但总体均值略低于零，且 `4thboywithwhite` 一条事件贡献了最大 `+0.02809` margin 和 GT rank `5→3`。另一方面，`1strowleftboyturning`、`ab_whiteboywithbluebag`、`bikefromlight` 等事件出现下降。结果呈现“许多近零扰动 + 少数较大单例”，不满足稳定性要求。

### 与固定状态 IoU 的关系

fixed-state IoU 平均提升主要由单个事件贡献，中位数为零，仍有 8/26 下降。即使当前 gate 能在部分事件改善 IoU，也不能证明它改善了目标/背景表征；它可能只是扰动了 CENTER 峰。

## 归因

本实验支持：

1. 只控制跨模态 Value 消息的 gate 不会自动产生目标选择性。
2. 当前 PVL 风格 reliability 在 50 序列事件中存在稳定的背景偏置。
3. 不能把问题归因于简单的消息注入过强；输入信号方向本身不可靠。

本实验不支持：

- 训练当前 `message_gate`；
- 继续搜索 gate bias、temperature 或 message scale；
- 将当前结果解释成 PVL 在所有任务上无效；
- 将 fixed-state IoU 的单例增益解释成闭环跟踪收益。

## 失败运行

首次运行在新分支的 multi-head 消息重排处触发 shape error，未产生结果；该运行目录保留为实现合同失败证据。随后修复为 `[B,H,L,head_dim] -> [B,L,C]` 重排并在独立 retry 目录完成 50 序列诊断。

## 下一步方案

### 方案 A：停止当前 uncertainty gate，先做目标条件信号审计（推荐）

不再调整 gate 强度。对每个 loc3/6/9，分别记录：

- GT 与 hard-background 的 template-to-search signed margin；
- 跨模态消息前后 target/background feature margin；
- CENTER 的 size/offset 解码误差；
- gate 与实际 CENTER margin 的相关性。

只有当某个信号在独立 50 序列上对 GT/hard-background 有正向区分，才允许把它接入 MHSA。

### 方案 B：若继续保留 PVL，改变职责

PVL 只做模态可用性估计，不直接形成空间 gate。每个模态使用独立的 source reliability，并把跨模态消息写成：

\[
m_{rgb\leftarrow tir}=r_{tir}m_{rgb\leftarrow tir},\quad
m_{tir\leftarrow rgb}=r_{rgb}m_{tir\leftarrow rgb}
\]

其中 `r` 必须先通过 50 序列的目标/硬背景审计；不能继续使用当前共享的 foreground/background 反向 reliability。该方案仍应先固定状态验证，不进入训练。

### 方案 C：若表征已正确但 CENTER 错位

转向 CENTER size/offset 与状态更新诊断。当前证据中已经存在“top-1 接近 GT 中心但 IoU 仍低”的案例，融合 gate 对此没有直接作用。

## 最终路线

**停止当前 PVL uncertainty gate 的训练和强度搜索。** 下一步优先做 50 序列的 target-conditioned signed-margin 审计；如果没有正向目标选择性，直接转向 CENTER 解码与状态恢复机制。当前实验已完成的唯一可靠结论是：不确定性可以抑制跨模态消息，但现有不确定性估计不能决定该消息是否有利于目标。

## 证据路径

- `output/experiments/M1_MSHA_UNCERTAINTY_GATE_50SEQ_20261021_RETRY/{summary.json,per_event.csv,validation_events.json,sequence_list.txt}`
- `experiments/manifests/m1_msha_uncertainty_gate_50seq_20261021.txt`
- `lib/models/layers/tbsi_layer.py`
- `tools/run_m1_template_query_stage_diagnostic_20261020.py`
