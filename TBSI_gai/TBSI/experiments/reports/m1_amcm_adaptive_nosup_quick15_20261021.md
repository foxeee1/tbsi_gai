# M1 AMCM Entropy-Adaptive Global/Foreground Matching Quick 实验

## 目的

在固定 50/50 global-foreground matching 的基础上，验证 token-level 不确定性回退是否能进一步稳定前景匹配。唯一新增机制是 foreground matching attention entropy 产生的确定性混合权重；没有新增 loss、可学习门控或外部链路修改。

## 核心算子

分别计算 global matching 和 foreground weighted matching：

\[
M_i^g=\operatorname{Match}(x_i,t),\qquad
M_i^f=\operatorname{Match}(x_i,t;w_{fg}).
\]

对 RGB/TIR 前景匹配 attention 计算归一化熵 (H_i^{rgb},H_i^{tir})，使用较保守的共同置信度：

\[
\lambda_i=1-\max(H_i^{rgb},H_i^{tir}),
\qquad
M_i=\lambda_iM_i^f+(1-\lambda_i)M_i^g.
\]

高熵位置自动退回 global evidence，低熵位置才使用 foreground evidence。随后沿用 E5 directed residual pre-fusion 路径。

## 实验合同

- 配置：`M1_AMCM_DR_ADAPTIVE_NOSUP_QUICK15_3EP_20261021`
- 初始化：M1 epoch15 checkpoint；仅训练 AMCM
- LasHeR proxy train：512 samples/epoch × 3 epochs，batch 2，FP32，worker 4
- 评测：同一 20 failure + 5 control，25-event closed-loop
- 随机 Forward/Backward：通过；输出 shape 与原 AMCM 一致

## 结果

| 子集 | 事件数 | 平均 event IoU delta | 胜/负 |
|---|---:|---:|---:|
| failure | 20 | `+0.00048` | 8 / 7 |
| control | 5 | `+0.00280` | 2 / 3 |
| all | 25 | `+0.00095` | 10 / 10 |

代表性变化：

- `basketball849`：`-0.0036`，继续保持稳定，较 foreground-only 的 `-0.5033` 明显改善。
- `middrillmaster`：`+0.1789`。
- `rightgirltakingcup`：`+0.2828`。
- `rightcar-chongT`：仍为 `-0.3811`。

## 判定

**QUICK-COMPLETE / MECHANISM-SUPPORTED / NOT-PROMOTED**。

Entropy fallback 比固定 50/50 有轻微整体改善，但幅度很小，且没有修复最主要的极端退化序列。该结果支持“global fallback 能抑制 foreground-only 过激响应”，不支持“matching uncertainty 已经能够完成目标/背景判别”。

停止继续搜索固定比例、温度或 entropy 阈值。下一步若继续，应转向固定 state/crop/frame 的 GT/hard-background peak 与 residual margin 诊断，确认 `rightcar-chongT` 的错误是否在 matching 之前已经存在；若 matching 前后均错误，则该旁路不能解决其根因。

