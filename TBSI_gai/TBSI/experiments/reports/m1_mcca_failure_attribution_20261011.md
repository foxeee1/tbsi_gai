# M1 MCCA 逻辑与失效归因（2026-10-11）

## 目的与结论

本报告只读核对 MCCA 实现、训练图和已有结果，不启动训练或推理。结论先分三层：

1. **实现接通：通过。** MCCA 插在 TBSI-E 模板联合 MHSA 消息路径；关闭开关和零输出初始化均与原模型精确 parity。
2. **优化发生：有证据。** P0 中输出层首步梯度有限且参数更新，第二步瓶颈层出现梯度；3-epoch G1 中六个 adapter 输出投影均离开零初始化。
3. **跟踪效用：未证明稳定。** test60 的正向差值没有在 fresh96 重现；full244 的点估计小幅为正但配对区间跨零。应称为“当前 MCCA 结构与训练合同未通过稳定收益验证”，不能称为代码没接上，也不能断言 MCCA 原理无效。

## 实际公式

在第 \(l\in\{3,6,9\}\) 个 TBSI-E 模板交互位置，将 RGB/TIR 模板 token 拼为 \(X=[X_r;X_t]\)，先归一化并计算 joint MHSA：

\[
Z=\operatorname{LN}(X),\quad [Q,K,V]=ZW_{qkv},\quad
A=\operatorname{softmax}(QK^\top/\sqrt{d_h}),\quad \bar A=\operatorname{Dropout}_{attn}(A).
\]

记跨模态消息和原始 joint attention 输出为：

\[
C_{r\leftarrow t}=W_o\operatorname{Concat}_h(\bar A_{r,t}V_t),\qquad
C_{t\leftarrow r}=W_o\operatorname{Concat}_h(\bar A_{t,r}V_r),
\]
\[
M=W_o\operatorname{Concat}_h(\bar A V),\qquad M=[M_r;M_t].
\]

每个接收模态、每个交互层有独立瓶颈 adapter：

\[
\phi(q,c)=W_2\,\operatorname{GELU}\!\left(W_1[q;c;|q-c|]+b_1\right)+b_2,
\]

其中 \(q_r=Z_r\)、\(q_t=Z_t\)。MCCA 输出是：

\[
\widetilde M_r=M_r+\phi_r(Z_r,C_{r\leftarrow t}),\quad
\widetilde M_t=M_t+\phi_t(Z_t,C_{t\leftarrow r}),
\]
\[
X^+=X+\operatorname{Dropout}([\widetilde M_r;\widetilde M_t]).
\]

\(W_2,b_2\) 零初始化，因此初始 \(\phi=0\)，G1 起始前向等于关闭 MCCA 的路径。模块不重算或替换 attention，也不改变 CENTER head；它是在原 joint MHSA 输出上，依据接收端模板 token 与单独提取的跨模态消息增加一个可学习校正项。代码：[`mcca_message_adapter.py`](../lib/models/layers/mcca_message_adapter.py#L5)、[`tbsi_layer.py`](../lib/models/layers/tbsi_layer.py#L294)。

训练仍使用原跟踪目标：

\[
\mathcal L=2\mathcal L_{GIoU}+5\mathcal L_{L1}+\mathcal L_{focal},
\]

没有新增 MCCA 专属的模态可靠性、跨模态一致性或定位排序损失。梯度可以经后续网络和 CENTER 输出反传到模板 adapter，但这是间接的任务监督，不等于显式教会模块“何时该信 RGB/TIR”。

## 证据与数据

- P0：disabled parity `0`，enabled zero-output parity `0`；输出层首步梯度范数非零，第二步 fc1 梯度非零，连续两步参数更新范数约 `8.97e-4`。证据：`output/experiments/M1_MCCA_G0_G1_20261010/p0.json`。
- 3 epoch matched run：G1 六个 adapter 输出投影均被更新；通用 trainer 记录 G1 `2/252` 参数零梯度，G0 `2/228`。这不报告 MCCA 的逐损失梯度、adapter 残差比例或难例条件下更新量。
- 学习率分组源码按参数名中的 `tbsi` 归入基础 `TRAIN.LR` 组；MCCA 位于 TBSI layer 内，因此不是落入其余 backbone 的 `0.1×` LR 组。既有 G0/G1 配置 `TRAIN.LR=1e-4`；现有记录不支持“adapter 被错误分到 backbone 低 LR”这一解释。分组代码见 `lib/train/base_functions.py`，actor 的 loss 权重见 `lib/train/train_script.py`。
- test60：G1-G0 AUC `+1.3995`，39 胜/21 负；去掉最大获益序列后仍 `+0.5001`，但 bootstrap 95% CI `[-0.162, 3.664]`，跨零。
- fresh96：G1-G0 AUC `-0.8453`，41 胜/55 负，95% CI `[-2.3608, 0.2783]`；PR/NPR 也负向点估计，区间跨零。
- full244：G1-G0 AUC `+0.4488`，中位差 `+0.0776`，135 胜/109 负，95% CI `[-0.5310, 1.4521]`；PR/NPR 点估计小幅正向，区间同样跨零。G1 对原始 M1 AUC 仅 `+0.0494`，PR `-0.1006`。
- full244 的 G0/G1 都在 M1 epoch15 checkpoint 上继续训练 3 epoch；所以 G1-G0 才是控制了续训的 MCCA 对照。M1 vs G1 不能单独归因于 MCCA。

## 失效原因排序

| 优先级 | 解释 | 依据 | 证据等级 |
|---|---|---|---|
| 1 | **作用点和目标问题不完全对齐。** MCCA 调整的是 3 个模板阶段的 MHSA 消息；M1 已诊断的困难还包括搜索候选排序和 CENTER 的 offset/size 读出。它没有直接修正当前帧搜索 token、搜索融合输出或解码框，作用必须穿过后续 backbone 才能影响定位。 | 插入路径代码；D1/D2 诊断显示候选排序与回归读出两者均有限。 | 强结构事实；作为失败主因仍是推断 |
| 2 | **它不是可靠性门控。** 输入只包含接收端 token、跨模态消息及两者绝对差；没有质量/遮挡/错位估计，也没有空间稀疏选择或显式抑制不可靠消息。因而双向 adapter 可能修正消息，但不能保证只在源模态可靠、位置对应时注入。 | 公式及代码；fresh96 的负向、full244 的小正向提示跨序列异质性。 | 结构事实；与指标冲突的因果关系未证实 |
| 3 | **监督目标偏间接。** 只有标准跟踪 focal/L1/GIoU loss；这些目标能训练 adapter，却不直接约束跨模态贡献的可靠性或改善困难负峰排序。模块可能学到提升训练损失的通用偏置，而不是稳定的融合决策。 | actor loss 路径；无 MCCA 专属损失。 | 结构事实；是否导致泛化不足未证实 |
| 4 | **输入统计可能不匹配。** query 经 LayerNorm，cross message 是注意力加权的 V 再经输出投影；二者幅度、稀疏性与方差可能不同，拼接后未做专门的 message normalization/gating。 | 代码无输入/残差分布记录。 | 待测假设，不能据此断言“接不上” |
| 5 | **零输出初始化可能让短训早期学习偏慢。** 首步 \(W_2=0\) 时 fc1 梯度为零，先更新输出投影，之后瓶颈层才获得梯度。 | P0 的首、次步梯度观测。 | 机制已证实；对 3 epoch 结果影响大小未知。不是断路 bug |
| 6 | **双向/多层校正可能冗余或相互干扰。** 六个 adapter 同时训练，方向开关能改变内部输出，但 fixed-weight 单方向干预没有在 fresh96 上超过 G0；它不是单方向独立训练对照。 | fresh96 固定权重方向消融。 | 低置信假设；不能断言两方向抵消 |

## 判断

当前最稳妥的归因是**有效性/泛化不稳定，且机制与被期待的搜索期可靠性选择并不等价**，不是实现断开或完全没有学习。现有数据不足以在“模板消息校正无效”“输入尺度不适配”“监督间接”之间唯一归因。另一次被用户停止的训练是 G0（MCCA 关闭）从 SOT 起训，运行至 epoch1 batch `750/3744`，出现一次非有限梯度范数警告；它不能用来判断 MCCA 的训练稳定性，已按中断记录保存。

不建议现在继续训练或调参。若以后重新开启，应先做无训练统计：在预注册的 train-side 失败前驱帧记录每层/每方向 \(\|\phi\|/\|M\|\)、query/message 均值方差与 cosine、GT 区域和困难负峰位置的 adapter 残差差异，并将这些量与 CENTER Hann 峰选择和解码误差配对。只有确认 adapter 产生了有方向的有效改变量，且改变量命中目标/干扰竞争区域，才值得立一个新实验合同；test-244 不再用于调参。

## 来源与复现身份

- Parent：受保护 M1 epoch15，checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`。
- Matched MCCA full244 evidence: `M1-MCCA-FULL244-PAIRED-20261011`; report `m1_mcca_full244_paired_20261011.md`。
- Source hashes: `mcca_message_adapter.py` `986519c1b675acb2c6f3efcf5a6e69ae743f2c7b6d61a678fb8984a5c5a03f7a`; `tbsi_layer.py` `a87283c5518c1a355a2659762b905150d3316a16a150f923461c333fdfb5ae73`; optimizer grouping `base_functions.py` `5c85dde62b1587c15a5d35eda8cf1b45c12b64a51ca32e91c29fcd0d0a11cf22`; MCCA G1 config `5256d1bac0573dd7a7e4da5ad13a4d6c091b777219a5cd81d2f884290400467d`.
- No training, evaluation, checkpoint write, or new GPU work in this attribution.
