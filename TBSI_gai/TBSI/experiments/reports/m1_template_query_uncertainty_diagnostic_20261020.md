# M1 Query-conditioned 模板不确定性诊断（2026-10-20）

## 唯一改变量

在 `V2TemplateFusion` 内将模板 token participation 从 agreement-only 改为：

\[
q_i=\frac12\max_j\cos(t_i^{rgb},s_j^{rgb})+
\frac12\max_j\cos(t_i^{tir},s_j^{tir}),
\]
\[
r_i=\frac12\cos(t_i^{rgb},t_i^{tir})+\frac12q_i,
\qquad p_i=\sigma((r_i-\tau)/T).
\]

`p_i` 只作为 template-key attention logit 的 `log(p_i)` bias。`tbsi_fuse_search`、CENTER head、loss、tracker 状态和 M1 epoch15 权重均未改变。配置为 `tau=0.5,T=0.15`，同一固定 25-event manifest，FP32 闭环前缀。

## 逐序列结果

相对同入口 `off`，all 25 events 平均 event IoU `+0.02587`，11 升、10 降；failure 组平均 `+0.03412`，7 升、9 降；control 组 `-0.00715`。收益由少数大幅事件主导：`whitesuvturn +0.9376`、`rightcar-chongT +0.3776`；同时 `leftunderbasket -0.6570`、`basketball849 -0.0527`、`boyunder2baskets -0.0594` 明显退化。因此尚未形成稳定的序列级改善证据。

## 内部信号

query match 均值 `0.71323`，综合 reliability 均值 `0.67276`，participation 均值 `0.74527`，最低 participation 的均值 `0.57410`。但跨模态 route mass 与 message/input norm 仍接近 parent：`r_to_t=0.33324`、`t_to_r=0.72099`、message/input norm `0.12414`。这说明 query 信息已经进入 participation，但尚未将模板消息重排成明确的目标区域选择性。

## 决策

**保留为诊断候选，不晋级训练。** 相比 agreement-only，query-conditioned 版本更符合目标条件建模方向，并在困难组出现初步正信号；但 25 个事件中升降接近，且存在同等幅度的灾难性退化，不能证明解决 M1 的目标/背景区分问题。下一步若继续，只做固定失败帧的 GT/background template participation、GT rank、peak margin 和 post-`tbsi_fuse_search` 特征 margin 分解，重点验证 `whitesuvturn` 与 `leftunderbasket` 的差异。只有目标选择性和下游 margin 同向，才允许短训。

证据：`output/experiments/M1_TEMPLATE_QUERY_UNCERTAINTY_T05_20261020/`。
