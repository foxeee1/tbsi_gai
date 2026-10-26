# M1 PVL-MHSA 不确定性诊断与下一步方案（2026-10-21）

## 诊断对象与证据

本诊断使用冻结的 M1 epoch15 checkpoint，在相同 template、state、crop 和 frame 下分析四条代表序列的 pre-onset/onset 帧。记录了 block00/02、TBSI loc3/6/9、backbone norm、`tbsi_fuse_search` 前后 feature geometry 和 CENTER 输出。数据来自：

- `output/experiments/M1_LOW_AUC_INFORMATION_FLOW_CASES_20261020/{per_event.csv,stage_metrics.csv,summary.json}`
- `output/experiments/M1_TEMPLATE_QUERY_STAGE_DIAGNOSTIC_20261020/summary.json`
- `experiments/reports/m1_lowauc_information_flow_cases_20261020.md`

这是固定状态机制诊断，不是总体性能评估，也没有改变 M1 代码或权重。

## 结果

| 序列 | 最早可见问题 | TBSI 后变化 | CENTER/闭环含义 |
|---|---|---|---|
| `ab_bikeoccluded` | block00 RGB target/background margin `-0.1005`，TIR `+0.2118` | block02 后 RGB 回到 `+0.1859`，loc3/6/9 继续为正 | onset top-1 已接近 GT 中心，但 IoU 仍低，优先怀疑 size/offset 解码，不是错峰 |
| `3rdfatboy` | block00 RGB/TIR margin 降到 `+0.0761/+0.0580` | loc3/6 局部回升，loc9 RGB 降至 `-0.0019`，最终 RGB `-0.0092` | 早层证据变弱并传到读出，top-1 偏离约 1 个 cell |
| `leftfarboycomingpicktheball` | RGB 保持正，TIR 从 block00 `+0.0706` 到 block02 `-0.0159` | loc3 短暂接近 0，loc6/9 仍接近或低于 0 | 模态分化没有被持续修复，框误差和后续状态反馈仍需单独拆解 |
| `rightbottlecomes` | RGB/TIR 从 block00 起接近 0 | TIR 到 backbone norm 增至 `+0.1135`，但没有对应中心回归 | 表征增强不等于 CENTER 选择正确；该例是闭环退化案例 |

已有 query-conditioned uncertainty 诊断进一步显示：validation 5 事件中 GT rank 改善 `0/5`，CENTER margin 均值变化 `-0.00090`，query foreground/background 差值变化为 `-0.0944`；即当前 query estimator 更偏向背景，而不是目标。固定状态下它的影响很小，闭环中变化变大，说明状态反馈在放大微小峰扰动。

## 归因

### 已支持

1. **存在早层模态证据不足或分化。** 这不是后融合模块能够普遍恢复的，因为某些信息在 block00/02 已经弱化。
2. **TBSI 不是统一的破坏器。** 有局部回升，也有回升后再次衰减；不能直接把 loc3/6/9 某一处定为统一首因。
3. **`tbsi_fuse_search` 不是已证实的普遍二次放大器。** 既有 search-stage 资产不支持“融合后总是抹掉 target margin”。
4. **CENTER 与状态更新是独立风险。** 目标峰接近中心但 IoU 仍低的案例说明 size/offset 读出不能由融合模块自动修复；错误峰之后的 state/crop 更新会把局部错误变成连续漂移。

### 已排除或暂不支持

- 暂不支持“所有低 AUC 都源于 MHSA 跨模态融合强度过大”。
- 暂不支持“只用 RGB/TIR agreement 或 query match 作为不确定性，就能得到目标选择性”。现有 estimator 在验证事件中出现背景偏置。
- 不支持继续用 PVL 旁路强度搜索代替目标条件信号设计。

## 对 PVL 放在 MHSA 中的判断

PVL 放在 MHSA 内部有一个合理用途：**控制跨模态消息的可信注入量**。它不应直接承担目标/背景判别。当前 PVL 的概率 key/value 可以估计“另一模态消息是否可靠”，但其 attention score 本身没有可靠的目标标签条件，因此直接用它做 spatial target selector 会把背景也当成高匹配对象。

推荐把 PVL 改成 MHSA cross-modal branch 的门控信号：保留原始单模态/主回路，只有跨模态消息按不确定性衰减。设第 `l` 个 block、位置 `i` 的跨模态消息为 `m_i^l`，定义：

\[
u_i^l = \operatorname{norm}\left(\tfrac12 \operatorname{Var}_{k}(s_{ik}^{rgb}) +
\tfrac12 \operatorname{Var}_{k}(s_{ik}^{tir})\right)
\]

并加入跨模态 disagreement 和 attention entropy 的校准项：

\[
g_i^l = \sigma(a_l - b_l u_i^l),\qquad
\tilde m_i^l = g_i^l m_i^l
\]

\[
y_i^l = y_{i,\mathrm{main}}^l + \gamma_l\tilde m_i^l
\]

其中 `gamma_l` 零初始化，`b_l >= 0` 通过 softplus 参数化；不确定性只控制消息强度，不改变 query 的空间位置，不直接重排 CENTER 峰。

更稳妥的实现是对 source-to-query 的 K/V 消息做 gate，而不是对整个 MHSA 输出做 gate：

\[
K' = gK + (1-g)K_{\mathrm{self}},\quad
V' = gV + (1-g)V_{\mathrm{self}}
\]

这样高不确定位置退回单模态 self message，低不确定位置才接收跨模态补充。必须保留原始 residual 和 zero-gate identity，避免把不确定性错误变成信息删除。

### 为什么不是直接用 uncertainty 做 target selector

当前证据中的 query foreground/background 差值为负，说明“不确定性低”不等于“目标”。遮挡、热目标弱响应或两模态视差都可能使真实目标更不确定。若直接使用 `p_i` 选择目标，会进一步偏向背景上下文。PVL 在 MHSA 中的合理边界是“拒绝不可靠的跨模态消息”，不是“决定哪个位置是目标”。

## 建议的最小单变量实验

只做一个改变量：在现有 `V2TemplateFusion`/MHSA 的跨模态消息输出上加入 **PVL uncertainty gate**，不改 Dataset、Loss、Optimizer、CENTER head、状态更新和 TBSI 主回路。

### 输入信号

- PVL key variance；
- RGB/TIR cross-modal disagreement；
- attention entropy；
- 可选的模板一致性只作为校准项，不作为目标标签。

### 合同

- 输入输出 token shape 和 channel 不变；
- `gamma=0` 时与原 M1 bitwise/数值等价；
- gate 只作用于 cross-modal message；
- 记录每个 loc3/6/9 的 `u`、`g`、message norm、GT/hard-background gate 均值和 CENTER margin；
- 先做 fixed-state diagnostic，不训练。

### 判定标准

候选只有在同一 fixed state 下同时满足以下条件，才允许进入短训：

1. GT/hard-background gate 差值不低于 control，且不能只由单个事件贡献；
2. 至少一半代表事件的 GT rank 或 CENTER GT-hard-BG margin 改善；
3. 非目标位置的 gate 不整体升高；
4. 固定状态改善不能以 CENTER size/offset 明显恶化为代价。

若 gate 只降低所有消息幅度、但 GT rank 不变，应判为“抑制有效但没有目标选择性”，停止训练；若 fixed-state 有效而闭环无效，转去状态更新/恢复机制，不继续改 MHSA。

## 后融合旁路与主回路边界

| 诊断结果 | 适合的解决位置 |
|---|---|
| 跨模态消息不可靠，但单模态仍保留目标证据 | MHSA 内 uncertainty gate；这是 PVL 当前最合理的用途 |
| 目标证据在 block00/02 已丢失 | 主干 Encoder/浅层跨模态交互；后融合无可恢复信息 |
| TBSI 后局部回升但 CENTER 仍偏移 | CENTER size/offset 或读出校准；融合模块不是首选 |
| 固定状态改善、闭环继续漂移 | 状态更新拒绝、恢复和重定位机制 |
| 融合前 target margin 正、融合后稳定变负 | 才考虑改 `tbsi_fuse_search` 主回路或其输入归一化 |

## 最终建议

当前不要替换新的通用模块。先做一次 **PVL-MHSA uncertainty gate 的固定状态合同诊断**。该方案只验证“按不确定性降低跨模态消息强度”这一假设，风险和改动面小于继续替换整个融合模块。

如果该诊断不能提高 GT/hard-background margin，就停止 PVL 线，转向：

1. 早层模态证据保持；
2. CENTER size/offset 解码审计；
3. 状态更新和恢复机制。

当前推荐路线是：**不换新模块，先验证 uncertainty gate；验证失败则回到主回路/闭环，而不是继续旁路搜索。**
