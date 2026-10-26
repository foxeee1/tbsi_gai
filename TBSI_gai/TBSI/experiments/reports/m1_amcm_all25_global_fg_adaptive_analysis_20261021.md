# M1 AMCM 三组全量 25 事件分析

## 范围

对已完成的 foreground-only、mix50、entropy-adaptive 三组 25-event closed-loop CSV 做逐序列对照。结果均来自同一 M1 parent、同一 3 epoch proxy 合同和同一事件清单；不是 full-244 性能结论。

## 汇总

| 组别 | >+0.005 | <-0.005 | 近零 |
|---|---:|---:|---:|
| foreground-only | 8 | 8 | 9 |
| mix50 | 9 | 9 | 7 |
| adaptive | 10 | 9 | 6 |

adaptive 相对 mix50 的逐序列相关为 `0.9949`，说明它没有产生独立的序列级行为。

## 全量事件分类

### A. global fallback 明确修复

- `basketball849`：`-0.5033 → -0.0268 → -0.0036`。这是最强证据，说明 foreground-only 在该序列中产生了错误的局部响应，global evidence 起到了保护作用。
- `boyunder2baskets`：`-0.0591 → -0.0251 → -0.0578`。mix50 有缓解，但 adaptive 又回到较差结果，说明 entropy 不是可靠的选择信号。
- `girlafterglassdoor`：`-0.0245 → +0.0032 → -0.0165`。同样说明固定回退有效，而 entropy adaptive 可能重新信任了不可靠 foreground evidence。

### B. adaptive 有局部帮助，但不是稳定规律

- `leftunderbasket`：`+0.0024 → -0.0083 → +0.0104`。
- `rightgirltakingcup`：`+0.3111 → +0.2583 → +0.2828`。
- `middrillmaster`：`0 → +0.1789 → +0.1789`。

这些事件表明 mixture 能改变闭环路径，但收益仍由少数序列贡献，不能证明目标/背景排序已经稳定改善。

### C. 三组共同失败，matching 线无法解释

- `rightcar-chongT`：约 `-0.381`，三组完全一致。
- `1stcol4thboy`、`lastleftgirl`、`leftboy2jointhe4`、`leftpingpongball`、`whitesuvturn`：三组均保持 0 IoU。

这些序列更可能属于状态/crop、模板身份失配或目标出界后的反馈链问题。global/foreground/adaptive matching 都没有改变结果，继续改 AMCM matching 不具备区分能力。

### D. 低幅度随机/解码扰动

`1strowleftboyturning`、`ab_pingpongball2`、`bottlebetweenboy\`sfeet`、`2girlup`、`ab_bikeoccluded`、`whitecarturnleft` 等序列的变化通常在 `0.01--0.06` 范围，且 adaptive 与 mix50 方向相近或略有波动。这些结果不足以判断稳定机制，可能来自局部峰、框解码和闭环状态的联合变化。

### E. 稳定正向但不具备归因价值

`boy\`sheadingreycol`、`biketurnright`、`girl\`sblkbag`、`silvercarturn`、`darkcarturn` 等在三组中都保持小幅正向或接近零。由于 M1 parent 本身在这些事件上并未发生严重错误，不能把它们解释为目标/背景修复证据。

## 剩余瓶颈判断

全量结果把问题分成两类：

1. **前景信号过激类**：global fallback 有效，例如 `basketball849`；这条线已经得到机制支持，但收益上限有限。
2. **错误峰/状态链类**：三组共同失败，例如 `rightcar-chongT` 和多条 0 IoU 事件；matching 权重变化无法改变它们。

因此当前 AMCM 旁路已经达到它在无目标/背景监督和小残差注入合同下的能力边界。继续添加更复杂的 matching、attention 或 entropy 算子，预计只会在第一类事件上微调，不能解决第二类事件。

## 决策

**ANALYSIS-COMPLETE / FALLBACK-EFFECT-CONFIRMED / MATCHING-LINE-BOUNDED**。

下一步不再扩展 global/foreground matching。只需对 `rightcar-chongT`、`basketball849`、`middrillmaster` 各选一个固定 state/crop/frame，核对 matching 前后 GT/background margin 和 CENTER rank，作为最终判断：若 `rightcar-chongT` 在 matching 前已无目标优势，则停止旁路融合线，转向模板身份或状态恢复；若 matching 有优势而 CENTER 不变，才研究 fusion-to-readout 的可逆耦合。

