# M1 AMCM 固定失败帧结构诊断（2026-10-19）

## 目的与协议

检验 E6 directed-residual AMCM 是否在 M1 低 AUC 测试序列的 crop 内，将 RGB/TIR 表示及 `tbsi_fuse_search` 输出朝目标、远离强背景峰修正；并检查该变化是否与 CENTER 解码 IoU 同向。

从 M1 full-244 逐序列表中筛选 AUC 46.50--55.08、`IoU<0.2` 帧比例不高于 0.30、无持续失锁到序列末尾的 10 条序列。每条选取一帧：M1 trace 中 GT 有效且位于 crop 内，M1 闭环 IoU 最接近 0.30。冻结该帧 M1 `tracker_state_before`，M1 与 E6 分别重置同一首帧模板，再处理固定帧；不把任一臂预测状态反馈到后续帧。

目标区域由 GT 框投影到 16x16 CENTER grid 得到。分数、GT rank 与 margin 使用 tracker 实际解码用的 `CENTER score_map * Hann window`。对每帧先用 M1 分数图选 GT 区域外最高响应 cell，并固定该 cell 作为两臂比较的 hard-background 特征位置；各臂 margin 仍各自用其最强背景响应计算。特征快照包含 AMCM 输入/输出 RGB/TIR、residual、`tbsi_fuse_search` 输入/输出、分数图与 GT mask。

## 配置与来源

- Parent：受保护 M1 `TBSI_E_MHSA_FULL_4090D_seed42`，epoch 15；checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`。
- 候选：`M1_AMCM_DR_LOSS_E6_REPR_QUICK15_3EP_20261019`，epoch 3；AMCM 为 pre-fusion directed residual，corrected RGB/TIR 实际进入 `tbsi_fuse_search`；checkpoint SHA256 `15ad1b72d543aadf3bb1a30857a37ca35091992a863c7c5eb5e5028a01111f58`。
- 权重核对：M1 state dict 的 267 个共有 tensor 与 E6 完全相同；E6 新增的 16 个 tensor 都属于 `amcm_reliability.*`。因此本诊断中的网络权重差异限定在 AMCM 模块。
- 代码版本：Git HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`；诊断脚本 SHA256 `b775ffb89d96f20ebbd523e4b31705b942fa5abd6ee5d491a572e89732aef3bf`。
- 配置：E6 YAML SHA256 `f2c225fec9e0c6f8eb64f78a3c207f235f30f1a5424684fed1947365c9e7e84f`；M1 使用受保护配置及固定 checkpoint。
- 环境：1x NVIDIA GeForce RTX 4090 D；Python 3.8 系列环境，PyTorch `2.0.0+cu118`，CUDA `11.8`；FP32；GPU 时长与 FPS 未单独计量，记 `NA`。
- 数据及范围：LasHeR 官方 `testingset`，来源于已完成 M1 full-244 的固定闭环 trace；本诊断只跑其中 10 个单帧事件，不是完整测试或闭环跟踪指标。
- 命令：`python tools/run_m1_amcm_test_failure_structure_diagnostic_20261019.py`。

## 逐序列结果

`res T/B` 是 E6 residual 在 GT 区域均值与固定 M1 hard-background cell 的范数比。`cos RGB/TIR` 比较该模态 GT 区域均值与同一个 hard-background 特征的 cosine；`cos fuse` 比较 `tbsi_fuse_search` 输出。余弦降低代表这两个特征方向更分离，不直接等同于 CENTER 排序变好。

| 序列 | 帧 | AUC | M1→E6 fixed IoU | ΔIoU | GT rank | margin M1→E6 | top-1 M1→E6 | res T/B RGB/TIR | cos RGB pre→post | cos TIR pre→post | cos fuse M1→E6 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 7rightorangegirl | 343 | 52.91 | .325→.326 | +.001 | 1→1 | +.2130→+.2442 | GT→GT | .97/ .98 | .468→.467 | .639→.643 | .713→.700 |
| ab_bolstershaking | 2952 | 53.87 | .300→.309 | +.009 | 1→1 | +.0931→+.1029 | GT→GT | .91/.89 | .564→.566 | .562→.565 | .764→.760 |
| blkboydown | 49 | 49.77 | .277→.303 | +.026 | 6→7 | -.0771→-.0913 | BG→BG | 1.18/1.19 | .675→.676 | .694→.695 | .643→.626 |
| boyss | 218 | 54.10 | .306→.299 | -.007 | 5→5 | -.1656→-.1773 | BG→BG | 1.17/1.17 | .470→.471 | .552→.556 | .698→.679 |
| foamatgirl&#96;srighthand | 67 | 51.84 | .297→.303 | +.006 | 2→3 | -.0202→-.0388 | BG→BG | 1.01/.98 | .504→.507 | .816→.817 | .728→.720 |
| girldownstairfromlight | 708 | 46.50 | .301→.322 | +.021 | 1→1 | +.0601→+.0796 | GT→GT | .95/.94 | .719→.721 | .530→.532 | .753→.744 |
| leftopenexersicebook | 120 | 48.12 | .300→.302 | +.002 | 7→7 | -.7118→-.7427 | BG→BG | 1.19/1.17 | .396→.398 | .354→.357 | .639→.633 |
| rightbluewhite | 236 | 53.12 | .298→.292 | -.006 | 1→1 | +.3830→+.4010 | GT→GT | 1.10/1.10 | .440→.443 | .546→.552 | .660→.651 |
| rightdarksingleman | 883 | 55.08 | .307→.327 | +.020 | 1→1 | +.2540→+.2961 | GT→GT | 1.03/1.03 | .635→.636 | .554→.557 | .629→.616 |
| umbrella | 197 | 53.19 | .303→.342 | +.039 | 3→3 | -.0380→-.0434 | BG→BG | 1.03/1.05 | .436→.437 | .628→.634 | .659→.646 |

## 结果解释

- 固定状态解码 IoU：均值 `0.3014 -> 0.3126`，平均 `+0.0112`；8/10 上升、2/10 下降。该结果只表示固定帧 decode 对这 10 个事件的小样本响应。
- 峰值排序：GT rank `0/10` 改善，`2/10` 变差，其余不变；top-1 的目标区/背景区归属 `0/10` 翻转。margin 5 条变好、5 条变差。ΔIoU 与 Δmargin Pearson `r=0.047`，在这组事件里没有同向关系。
- residual 空间分布：RGB/TIR GT residual 与 M1 hard-background peak residual 范数比均值分别为 `1.053/1.051`，范围约 `0.89--1.19`；空间 CV 均值 `0.403/0.414`。因此 residual 不是空间常数，但在目标与最强背景位置的幅值没有明显选择性，支持“位置选择信号弱”的怀疑。
- 模态表示：RGB/TIR target-to-hard-background cosine 均值分别从 `0.5307 -> 0.5320`、`0.5875 -> 0.5907`，轻微向更相似方向变化；二者并未在模态特征层清晰地把目标从背景拉开。
- 融合输出：`tbsi_fuse_search` 输出 target-to-hard-background cosine 均值从 `0.6886 -> 0.6775`，有轻微分离；但 CENTER rank/top-1 没随之改善。这表明融合表征变化确实进入 head，却没有稳定转成峰值排序变化。IoU 的少量提升更可能来自局部 box decode 的联合变化，当前证据不能证明是错误背景峰被纠正。

## 归因与下一步

结论：**支持下一步单变量检验**“E6 residual 缺少目标位置选择性，因此不足以稳定改变背景峰竞争”；同时对“AMCM 完全没有改特征”这一解释已排除。结果暂不支持用这 10 帧宣称已改善定位峰或解决 M1 失败链条。

下一步不应直接放大 alpha。优先在同一固定帧协议下拆开 residual 的方向与空间位置：保留幅值，比较 GT residual 与 M1 hard-background residual 的方向 cosine、逐位置有符号投影，以及 residual 对 `tbsi_fuse_search` 输出和 CENTER Hann margin 的逐位置贡献；并加入一组 GT/background hard-pair 特征诊断。这样可以区分“残差幅度相近但方向有效”与“残差没有目标条件方向”两种解释。只有目标区相对背景的改变量与 margin/GT rank 呈可重复同向关系，才值得考虑新的空间监督或融合注入设计。

限制：样本固定为 10 帧，存在序列内抽样依赖；GT 区域来自 GT 框在 16x16 网格上的投影；hard-background cell 固定由 M1 Hann map 选择，以保证两臂空间配对；单帧 IoU 不等同于闭环成功率。V1 使用了未加 Hann 的峰排序及特征范数背景点，仅留作审计；本报告与 V2 CSV 使用 Hann 响应和固定 M1 hard-background cell。`per_event.csv` 中补充的 max-feature-norm background 列仅作审计，正式配对特征指标以 `paired_per_sequence.csv` 为准。

## 证据文件

- `output/experiments/M1_AMCM_TEST_FAILURE_STRUCTURE_DIAG_V2_20261019/fixed_event_manifest.json`
- `output/experiments/M1_AMCM_TEST_FAILURE_STRUCTURE_DIAG_V2_20261019/per_event.csv`
- `output/experiments/M1_AMCM_TEST_FAILURE_STRUCTURE_DIAG_V2_20261019/paired_per_sequence.csv`
- `output/experiments/M1_AMCM_TEST_FAILURE_STRUCTURE_DIAG_V2_20261019/summary.json`
- `output/experiments/M1_AMCM_TEST_FAILURE_STRUCTURE_DIAG_V2_20261019/tensors/*.npz`
- `tools/run_m1_amcm_test_failure_structure_diagnostic_20261019.py`
