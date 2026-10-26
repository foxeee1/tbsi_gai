# M1 AMCM Global/Foreground 50-50 Matching Quick 实验

## 目的

验证 foreground-only matching 的退化是否来自过度依赖前景 mask。唯一改动是将全模板匹配与前景加权模板匹配固定按 50/50 混合，保留 E5 pre-fusion directed residual、无 supervision、原始 tracking loss、TBSI、CENTER 和状态更新。

## 核心算子

对每个 search token 分别计算：

\[
M_i^{g}=\operatorname{Match}(x_i,t),\qquad
M_i^{f}=\operatorname{Match}(x_i,t;w_{fg})
\]

再使用固定混合：

\[
M_i=0.5M_i^{g}+0.5M_i^{f}.
\]

混合结果进入原 directed residual message 和 pre-fusion 注入路径。输入输出 shape 不变：search `[B,C,16,16]`，template `[B,64,C]`，输出双流均为 `[B,C,16,16]`。

## 实验合同

- 配置：`M1_AMCM_DR_MIX50_NOSUP_QUICK15_3EP_20261021`
- 初始化：M1 epoch15 checkpoint；仅训练 AMCM
- LasHeR proxy train：512 samples/epoch × 3 epochs，batch 2，FP32，worker 4
- 评测：与 foreground-only 相同的 20 failure + 5 control，25-event closed-loop
- 新增参数量和训练策略与 foreground-only 相同

## 结果

| 子集 | 事件数 | 平均 event IoU delta | 胜/负 |
|---|---:|---:|---:|
| failure | 20 | `-0.00111` | 7 / 8 |
| control | 5 | `+0.00703` | 3 / 2 |
| all | 25 | `+0.00052` | 10 / 10 |

代表性变化：

- `basketball849`：foreground-only `-0.5033`，50/50 混合 `-0.0268`；极端退化明显被抑制。
- `middrillmaster`：`+0.1789`。
- `rightgirltakingcup`：`+0.2583`。
- `rightcar-chongT`：仍为 `-0.3810`，说明 global fallback 不能修复该类错误峰。

## 判定

**QUICK-COMPLETE / MECHANISM-SUPPORTED / NOT-PROMOTED**。

固定混合明显比 foreground-only 稳定，支持“foreground-only 过度依赖前景 mask”这一解释；但 failure 子集仍略低于零，且存在单序列大幅退化，不能进入 full-244 或论文主结果。

下一步若继续，只允许测试一个更保守的 token-level adaptive mixture：在前景匹配熵高、RGB/TIR 不一致或 foreground/global peak margin 冲突时退回 global；不建议继续搜索固定比例或增大 residual。应先在固定 state/crop/frame 上验证混合匹配是否改善 GT/background peak margin，再决定是否训练。

