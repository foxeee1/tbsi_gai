# M1 MHSA 候选峰重排机制验证（2026-10-10）

## 目标与实验边界

沿用已冻结的 96 条 train-side cohort、126 个固定状态帧，检验“正确模板不变时，MHSA 模板身份线索能否只在原 CENTER 候选峰内帮助选峰”。这是无训练反事实，不新增模型参数、不改 M1 配置/权重、不使用 test-244。

## 预注册解码组

- `B0`: M1 原始 Hann score argmax。
- `B1`: 离线 GT cell oracle，使用原模型同一位置 size/offset 解码；不在线部署。
- `B2`: 只对原 Hann top-8 重排。身份分数是 fused M1 search feature 与 TBSI 后 RGB/TIR 模板前景池化向量的平均 cosine；不拟合投影或温度。CENTER 与身份分数在 top-8 内分别 z-score。主权重 `lambda=0.5`，另报告 `0.25/1.0` 敏感性。
- `B3`: 同 B2 主权重，但只有 top-8 identity top1-top2 间隔超过候选内 1 个标准差才启用，置信度 `sigmoid((gap_z-1)/0.5)` 且 `c>0.5` 才重排。阈值预先固定，未按标签挑选。

报告的 top-1 是实际选中格是否落入 GT 支持区；top-5/rank 和目标对困难负峰 margin 均在该组候选排序内计算。B1 只有 oracle 格一个候选，因此不定义 margin。候选集 GT top-8 oracle IoU 用同一 size/offset 图枚举，是离线上界。

## 结果

下表给出每类固定状态的 target-cell 选择率、实际定位中位数和 B2 的正常帧副作用。`B2 λ=0.5` 为预先指定主结果。

| 状态 | B0 选中 GT / 中心误差 / IoU | B1 GT-cell 中心误差 / IoU / size L1 | B2 λ=0.5 选中 GT / 中心误差 / IoU | B2 λ=1.0 选中 GT / 中心误差 / IoU |
|---|---|---|---|---|
| matched-good (42) | 81.0% / 0.354 / 0.717 | 0.285 / 0.737 / 0.471 | 76.2% / 0.354 / 0.718 | 78.6% / 0.350 / 0.720 |
| pre-onset (40) | 70.0% / 0.444 / 0.657 | 0.396 / 0.670 / 0.639 | 70.0% / 0.434 / 0.654 | 67.5% / 0.490 / 0.652 |
| severe-onset (44) | 29.5% / 1.333 / 0.371 | 0.822 / 0.484 / 0.856 | 38.6% / 1.191 / 0.406 | 43.2% / 1.160 / 0.406 |

单位：中心误差为 16×16 token；size L1 为 token。严重起点的 B0 原始候选 top-5 含 GT 支持区 `36/44`，top-8 含 GT `40/44`；top-8 的 oracle-best IoU 中位数 `0.508`。所以不少事件存在候选恢复空间，但身份排序只修正了其中少数。

- `B2 λ=0.25`: severe-onset 选中 GT `16/44`，比 B0 多 3 个事件（+6.8pp），未达预设 +8–10pp；matched-good 选中率无损。
- `B2 λ=0.5`: severe-onset `17/44`（+4，+9.1pp），IoU 中位数 `+0.035`、中心误差中位数 `-0.142 token`；但 matched-good 由 `34/42` 降至 `32/42`（-4.8pp），超过 2pp 稳定性预算。pre-onset 框 IoU 均值差 `-0.0058`。
- `B2 λ=1.0`: severe-onset `19/44`（+6，+13.6pp），在 6 个不同序列纠正 target top-1；IoU 中位数 `+0.035`、中心误差中位数 `-0.173 token`。但 matched-good 少 1/42（-2.4pp），pre-onset 少 1/40（-2.5pp），仍越过 2pp 门槛；严重起点框 IoU 7 序列升、2 序列降、35 不变，存在异质性。
- 候选内 target-hard-negative margin 中位数从 B0 severe 的 `0.0195` 增至 B2 λ=0.5 的 `0.0555`、λ=1.0 的 `0.0875`；它证明身份分数确实影响了 top-K 内排序，但不代表多数帧最终选中目标。
- `B3`: 全体约 15.1% 帧通过置信门；44 个 severe-onset 中有 6 个触发，但仅 1 帧换候选 cell，target top-1 仍为 `13/44`，中心误差与 IoU 中位数均未优于 B0。该置信定义当前偏保守，不能证明可靠门控可行。
- `B1`: severe-onset GT-cell oracle 把 IoU 中位数从 `0.371` 提到 `0.484`、中心误差从 `1.333` 降至 `0.822 token`，但 size L1 从 `0.805` 升至 `0.856 token`。选格改善有上界收益，仍残留回归误差。

## 决策

**身份分数存在少量候选重排信号，但本轮不通过晋级门槛，不开始训练。** 强度加大后，严重起点 top-1 提升可达到 13.6pp，但 matched-good/pre-onset 各退化约 2.4–2.5pp；较温和的 `lambda=0.25` 保持正常帧，却只提升 6.8pp。B3 没有兑现安全门控的增益。增大 lambda 或调置信阈值继续在同一 96 序列上寻找折中，会构成事后调参，因此本 cohort 到此冻结。

现有证据支持：CENTER 的 top-K 候选中常有可恢复位置；raw cosine 身份排序能改变少量严重起点的选择，但不是稳定、正常帧安全的改进。它可能受限于模板原型/融合 search 不在同一判别空间，以及用未校准 cosine 排序。尚不能据此提出训练式身份路由，也不能称为 M1 整体涨点。下一步若继续，应在新的 train-side 独立 cohort 上先验证一个固定的身份相似度校准/匹配方式；或转查 CENTER 的 score 与 offset/size 联合读出，不再在本 96 条上搜索超参数。

## 复现信息

- 脚本：`tools/diagnose_m1_mhsa_candidate_rerank_20261010.py`，SHA256 `f07f36265cc0fb935eb07047649f0df8fe45315cf0166a2834608d7e21b0df7b`。
- 正式命令：`python tools/diagnose_m1_mhsa_candidate_rerank_20261010.py`。
- 正式结果目录：`output/experiments/M1_MHSA_CANDIDATE_RERANK_V3_20261010/`；含 `summary.json`、`per_arm.csv`、`paired_deltas.csv`。初始 V1 的 top-rank 字段按整张 score map 统计、V2 的 hard-negative margin 也按整张图统计，均与候选受限决策口径不一致；两个目录都保留但标为 superseded，报告和结论只引用 V3。
- 最终脚本 SHA256 `704e8c94574593c776cc4b33523325db8702a0e4562a70e361cbea72fb725aea`。
- M1 checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`；config SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1`，运行前后均一致。
- 1× NVIDIA GeForce RTX 4090 D 24 GiB；Python 3.8.20、PyTorch 2.0.0+cu118、CUDA 11.8、FP32；GPU 固定状态前向约 9.9 秒（不含加载）。B0 trace 最大框差 `0.0000687 px`。当前工作树 replay 非归档逐帧等价；该队列是 M1 训练数据，虽对既有诊断新鲜，也不是 backbone 未见序列。
- 无 AUC/PR/NPR/OP50/OP75/FPS：这是固定状态解码诊断，不是闭环跟踪 benchmark。
