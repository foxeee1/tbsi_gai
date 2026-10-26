# M1 模板不确定性参与度诊断（2026-10-20）

## 目的

检验 M1 的模板 MHSA 是否因无条件纳入全部 RGB/TIR 模板 token 而污染查询。唯一改变量是 `V2TemplateFusion` 的 template-key soft participation：

\[
a_i=\cos(z_i^{rgb},z_i^{tir}),\qquad p_i=\sigma((a_i-\tau)/T),\qquad A'_{q,k}=A_{q,k}+\log p_k.
\]

`tbsi_fuse_search`、CENTER head、loss、tracker 状态和 checkpoint 均未改；权重为受保护 M1 epoch15。`off` 与 `soft(\tau=0.5,T=0.15)` 使用同一固定 25-event manifest，FP32 闭环前缀评测。

## 结果

`soft` 相对同一入口 `off` 的平均 event IoU 变化为 **-0.01647**；8/25 上升、13/25 下降。明显退化包括 `leftunderbasket -0.6645`、`rightgirltakingcup -0.2396`、`boy\`sheadingreycol -0.1100`；明显上升包括 `whitesuvturn +0.9250`、`middrillmaster +0.1554`，呈现高方差而非可重复收益。相对磁盘中旧 M1 full tracking 结果的 delta 不能作为模块增益，因为该结果与当前入口合同不完全一致。

## 模块内部证据

在 4,923 次模板 MHSA 调用上：

| 指标 | off | soft | 变化 |
|---|---:|---:|---:|
| RGB→TIR attention mass | 0.33220 | 0.33645 | +0.00425 |
| TIR→RGB attention mass | 0.72176 | 0.71762 | -0.00414 |
| attention entropy | 3.40914 | 3.42674 | +0.01760 |
| message/input norm | 0.12364 | 0.12420 | +0.00055 |

soft participation 本身确实被触发：agreement 均值 `0.63239`，participation 均值 `0.68329`，最小值均值 `0.41296`，最低可到 `0.14339`。但 off/soft 的 route mass、entropy、message norm 相关性均约 `0.99`，说明它主要是轻微 logit 重标定，没有形成目标区域特异的模板消息重排。当前诊断没有观察到 GT/background residual、GT rank 或 post-`tbsi_fuse_search` margin 的一致改善。

## 结论

**暂不支持晋级训练。** agreement-only uncertainty 没有目标条件或空间监督，无法判断低一致性是否就是背景；当前结果满足“参与度改变”，但不满足目标选择性、下游同向和跨序列可重复三条证据。

下一步若继续，只允许在 MSHA 内以模板与当前 search 的对应相似度形成 query-conditioned participation，并先测 GT/background rank 与 post-`tbsi_fuse_search` margin；不增加外部支路、不增大强度、不做 full training。若仍无目标选择性，应放弃 uncertainty 路线，回到 M1 的模板状态/目标条件信息缺失这一系统瓶颈。

证据目录：

- `output/experiments/M1_TEMPLATE_UNCERTAINTY_OFF_20261020/`
- `output/experiments/M1_TEMPLATE_UNCERTAINTY_SOFT_T05_20261020/`
- `output/experiments/M1_TEMPLATE_UNCERTAINTY_OFF_DIAG_20261020/diagnostics.jsonl`
- `output/experiments/M1_TEMPLATE_UNCERTAINTY_SOFT_T05_DIAG_20261020/diagnostics.jsonl`
