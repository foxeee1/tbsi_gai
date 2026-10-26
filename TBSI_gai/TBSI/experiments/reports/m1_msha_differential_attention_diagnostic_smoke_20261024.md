# M1 MSHA Differential Attention Diagnostic Smoke

日期：2026-10-24  
Parent：`TBSI_E_MHSA_FULL_4090D_seed42` epoch15  
目的：在真实 `tbsi_e` 模板 MSHA 路径中，检查 self/cross attention 是否存在可测的共同响应，为差分消息算子提供或否定机制依据。

## 实施

新增了默认关闭的只读开关 `TBSI_MSHA_DIFF_DIAGNOSTICS_PATH`，位于 `V2TemplateFusion` 的原生模板联合 MSHA 路径。它不修改 logits、attention、Value、投影或输出，只记录 RGB/TIR 两个方向的 self/cross attention 分布重合、self/cross message 余弦相似度、跨模态 attention mass，以及有模板前景 mask 时的前景 query 分层统计。

`python -m py_compile` 和小型随机 tensor forward 均通过，zero/default mode 不启用该文件写入逻辑。

## Smoke 合同

- 运行器：`tools/run_m1_msha_query_role_fixed_state_20261022.py --smoke`
- 事件：1 个 matched-good 与 1 个 severe-onset train-side fixed state；
- 对照臂：M1、foreground-query、shifted-query；
- 3 个 `tbsi_e` MSHA 层；FP32，单线程，M1 checkpoint 不变。
- 输出：`output/experiments/M1_MSHA_DIFF_ATTENTION_DIAGNOSTIC_SMOKE_20261024/`

## 观察结果

诊断写出 36 行，覆盖 2 事件 × 3 对照臂 × 3 层。对 smoke 全部记录的平均值为：

| 量 | RGB query | TIR query |
|---|---:|---:|
| self/cross attention overlap | 0.785 | 0.732 |
| self/cross message cosine | 0.285 | 0.321 |
| cross attention mass | 0.296 | 0.817 |

按层和事件看，overlap 处于中高水平，但 message cosine 从接近 0 到约 0.57 变化；两个方向的 cross mass 也明显不对称。说明“存在共同注意力分布”是可测的，但“共同成分是有害冗余”尚未得到证明。当前结果也不支持使用单一全局差分强度。

## 结论

`DIAGNOSTIC-SMOKE-COMPLETE / MECHANISM-SIGNAL-PRELIMINARY / NO-TRAINING`

Smoke 只证明真实路径可观测且存在事件、方向异质性，不能支持实现差分算子或启动训练。下一步扩展到预登记的 28 个 fixed-state 事件，按 matched-good/severe-onset、RGB/TIR 方向和 MSHA 层分别统计；只有干扰事件中共同响应与定位退化具有一致关系，且 matched-good 不出现同等退化，才进入固定小幅差分对照。

## 28-event tagged follow-up

为避免把所有普通帧日志错误配到事件，runner 增加了 `sequence:frame:role:arm` 标签，并在同一合同下重跑了 28 个事件。最终输出位于 `output/experiments/M1_MSHA_DIFF_ATTENTION_DIAGNOSTIC_FIXED28_TAGGED_20261024/`，共 252 条 tagged MSHA 记录（28 事件 × M1/FG_QUERY/SHIFTED_QUERY × 3 层）。

M1 臂按三层平均后的结果如下：

| 量 | matched-good | severe-onset |
|---|---:|---:|
| RGB self/cross attention overlap | 0.77846 | 0.78065 |
| TIR self/cross attention overlap | 0.73701 | 0.73922 |
| RGB self/cross message cosine | 0.37337 | 0.37865 |
| TIR self/cross message cosine | 0.37619 | 0.38100 |
| RGB cross mass | 0.32886 | 0.32854 |
| TIR cross mass | 0.73380 | 0.73349 |

这些量在两类事件之间几乎没有差异。它们与现有同状态 FG_QUERY IoU 变化的相关系数也很弱：RGB/TIR attention overlap 分别约 `-0.234/-0.270`，RGB/TIR message cosine 约 `-0.064/-0.075`；cross mass 的相关方向相反且不稳定（约 `-0.002/-0.531`）。因此没有证据表明 severe-onset 事件具有更强的共同跨模态响应，也没有证据支持统一 subtractive differential attention 能针对该失败场景。

## Updated decision

`DIAGNOSTIC-COMPLETE / DIFFERENTIAL-OVERLAP-HYPOTHESIS-NOT-SUPPORTED / STOP-BEFORE-TRAINING`

停止当前“统一差分注意力”候选，不实现可训练 lambda，也不做 full training。诊断实现和 tagged fixed-state 产物保留为实验资产。若继续优化 MSHA，下一候选必须使用能区分事件的条件信号，例如目标 token 与局部 hard-background 的空间关系，且先通过同一类 tagged fixed-state 检验；仅凭 self/cross overlap 或消息余弦不能作为路由依据。
