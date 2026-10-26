# M1 AMCM 前景加权模板匹配 quick 实验

## 实验目的

验证一个单变量假设：在 E5 `pre_fusion directed residual` 保持不变的情况下，使用 tracker 提供的模板前景 mask 对 RGB/TIR template token matching 加权，能否让 residual 更集中于目标区域。关闭 AMCM reliability/KL supervision，不新增 loss，不修改 TBSI、CENTER head、数据划分和状态更新。

## 唯一改变量

`AMCM_RELIABILITY.MODE=directed_residual_fg`。对模板 token 的余弦匹配 logits 加入归一化前景权重的对数偏置：

\[
 a_{ij}=\operatorname{softmax}_j\left(\frac{\cos(s_i,t_j)}{T}+\log w_j\right),
 \qquad w_j=\operatorname{normalize}(\text{template foreground mask}) .
\]

RGB 和 TIR 分支分别计算匹配图，再按 E5 的 directed residual 路径生成有界 pre-fusion residual。模块输入为 search `[B,C,16,16]` 和 template `[B,64,C]`（RGB/TIR 各一支），输出 shape 与输入相同。残差上限为 `0.10`，其余训练合同保持不变。

## 训练与评测

- 配置：`M1_AMCM_DR_FG_MATCH_NOSUP_QUICK15_3EP_20261021`
- 初始化：受保护 M1 epoch15 checkpoint；仅训练 AMCM 参数
- LasHeR proxy train：512 samples/epoch × 3 epochs，batch 2，FP32，worker 4
- 评测：同一 20 failure + 5 control 固定事件，FP32 closed-loop；25-event proxy，不是 full-244
- 输出：`output/experiments/M1_AMCM_DR_FG_MATCH_NOSUP_QUICK15_3EP_20261021/closed_loop_25/`

## 结果

| 子集 | 事件数 | 平均 event IoU delta | 胜/负 |
|---|---:|---:|---:|
| failure | 20 | -0.03017 | 7 / 7 |
| control | 5 | +0.00349 | 3 / 2 |
| all | 25 | -0.02344 | 10 / 9 |

显著退化：`basketball849` `-0.5033`、`rightcar-chongT` `-0.3811`。可见提升：`rightgirltakingcup` `+0.3111`；其余提升较小。多个原本已失效的序列保持 0 IoU，说明该改动没有修复出界/错误状态反馈链。

## 归因

前景 mask 确实改变了 template matching 的权重，但没有建立“前景 token 相似 → search 目标位置”的可靠对应关系。模板前景区域可能包含目标内部的非稳定部分，或 mask 与当前 crop 中的目标位置发生外观/尺度错配；因此 residual 的方向仍由错误的局部相似峰决定。该实验没有新增监督，训练只能适应 tracking loss，无法把前景加权信号约束成目标/硬背景排序。

相对 E5 的结论是：前景加权没有提供稳定的目标选择性，且在少数事件上放大了错误峰。它不能支持继续调大 residual、温度或 mask 权重，也不应直接进入 50-event 或 full-244 训练。

## 判定

**QUICK-COMPLETE / NOT-PROMOTED**。保留实现作为可逆诊断候选；停止该 foreground matching 路线的强度搜索。下一步若继续研究，应先在固定 state/crop/frame 上记录 GT 与 hard-background 的 matching peak、residual margin 和 `tbsi_fuse_search` 前后排序，或改为候选模板条件的保守更新；不得仅增加门控复杂度。

