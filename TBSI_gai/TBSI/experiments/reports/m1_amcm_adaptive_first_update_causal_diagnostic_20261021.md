# Adaptive AMCM 首更新因果诊断

> 2026-10-21 更新：本报告中的 12/32 条 FP32 固定状态诊断仍有效，但开头引用的旧 full-244 `-12.56` AUC 差已被半精度 NaN/运行口径审计判为不可直接比较。新增独立 50 条 train-side 首更新结果与 FP16 根因见 `m1_amcm_adaptive_precision_contract_audit_20261021.md`。因此下文关于 test catastrophic 的假设已被数值合同证据更新，不能再用旧结果推断融合方法本身退化。

## 问题与实验合同

full-244 配对分析发现 adaptive 组相对 M1 下降 12.56 AUC 点，且 48 条严重退化序列常在初始化后的首次更新就接近零 IoU。此诊断只回答：**训练后的 adaptive AMCM residual 在固定输入、固定 template、固定 crop 和固定 state 下，是否改变 CENTER 的目标/背景排序与框解码？**不训练、不更新状态、不使用 testingset 选帧或调参。

样本来自已冻结的 32 条 LasHeR train-side manifest：`output/experiments/M1_NEED_AWARE_ACTION_SWEEP_20261012/trace_run/trace/trace_sequences.txt`。先做 12 个机制状态（6 个 M1 trace 中 crop-contained 的低 IoU 状态，6 个高 IoU 控制），再做 32 条序列全部可用的首次 crop-contained tracker update。全程在相同状态下比较 5 个 arm：M1、adaptive、adaptive residual 置零、residual 符号翻转、residual 空间平移 4×4 token。hard-background cell 由每个 arm 固定输入的 CENTER Hann response 在 GT mask 外取最大值；各 arm 使用相同 GT mask 和状态。

受保护 M1 配置哈希为 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`，checkpoint 哈希为 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`。adaptive config/checkpoint 哈希分别为 `9c360c75f44c23762ae7a1b31306e168e5c39e16cd468b042b2d5936f8dbe967` 与 `15e0bf9b2a489cecd24f1d4994a8b44a535aae2c689eca2ba3e2be237981a1b2`。环境为 1× NVIDIA GeForce RTX 4090 D、PyTorch `2.0.0+cu118`、CUDA `11.8`、FP32。

## 结果

### 12 个分层固定状态

| Cohort | Arm | n | 固定状态 IoU 均值 | GT rank=1 | 平均 GT-hard-BG margin |
|---|---|---:|---:|---:|---:|
| M1 低 IoU 状态 | M1 | 6 | 0.130514 | 2/6 | 0.012416 |
| M1 低 IoU 状态 | Adaptive | 6 | 0.130507 | 2/6 | 0.012319 |
| M1 低 IoU 状态 | Residual=0 | 6 | 0.130511 | 2/6 | 0.012414 |
| 稳定控制 | M1 | 6 | 0.900816 | 6/6 | 0.772546 |
| 稳定控制 | Adaptive | 6 | 0.900798 | 6/6 | 0.772543 |
| 稳定控制 | Residual=0 | 6 | 0.900824 | 6/6 | 0.772546 |

12 个状态中，adaptive 与 residual=0 的 GT rank 和 top-1 cell **0/12 发生改变**。IoU 平均差为 `-0.0000153`，最大绝对差为 `0.000162`；GT-hard-background margin 平均绝对差 `0.0000925`。符号翻转和 4×4 空间平移也都没有改变 rank 或 top-1 cell。

### 32 条序列首次更新

32/32 首次更新的 M1 GT rank 均为 1，top-1 均落在 GT 区；M1 trace IoU 最小/中位/最大为 `0.642/0.900/0.972`。这批冻结 train-side 首帧没有复现 full-test 的 catastrophic collapse，因此不能用它估计该故障在数据总体中的比例。

但它能检验常规首帧是否被模块扰动：adaptive 与 residual=0 在 32 个状态里 rank 变化 `0/32`、top-1 cell 翻转 `0/32`；平均绝对 IoU 差 `0.0000644`，最大绝对差 `0.000229`；平均绝对 GT-hard-background margin 差 `0.0000793`。符号翻转和空间平移同样没有改变 CENTER rank 或 top-1 cell。

### Residual 幅度与 checkpoint 合同

- 32 个首次更新上 adaptive residual/input L2 比值均值：RGB `1.65e-4`，TIR `3.68e-4`；12 状态相应为 `1.57e-4` 和 `3.35e-4`。
- candidate checkpoint 与 M1 有 267 个共同 state keys；所有共同键逐张量比较均相同。candidate 的 283 个键中只有额外 16 个 `amcm_reliability.*` 参数；没有 backbone、TBSI 或 CENTER 参数漂移。
- M1 checkpoint 与 AMCM 开启但 residual 置零的 arm 接近数值 parity：32 状态最大绝对 IoU 差 `2.96e-5`，rank/top-1 均未改变。

因此，当前 AMCM 在这些 train-side 固定状态上的实际注入接近 no-op。它没有把现有 target/background 排序修好，也没有在这些状态制造新的峰值错误。测试集的灾难性下降只能收窄到“AMCM 启用后的前向交互”，不能据本实验断言是匹配公式、residual 方向、CENTER 或解码的某一个首因；训练 cohort 上没有复现测试集故障是重要的未区分项。

## 失败尝试与 provenance

实验 runner 在初次执行时暴露了三项合同问题，失败输出目录均保留：

1. `M1_AMCM_ADAPTIVE_FIRST_UPDATE_CAUSAL_DIAG_20261021_RETRY1`：M1 arm 的相似度 map 维度读取不兼容；修正为按实际 tensor rank 读取。
2. `...RETRY2`：连续重建 tracker 时触发配置/权重 key 不匹配；修正后每个模型只构造一次，并在相同 tracker 上依次切换可逆 intervention hook。
3. `...RETRY3`：固定状态切片结果先完成，但后续 32 状态运行的 CSV 路径误指向默认目录；该 32 状态 CSV 已复制到自己的独立目录，12 状态实验用 `RETRY4` 重新执行并恢复。不得使用 `RETRY3/per_arm.csv` 作证据；它被 32 状态 CSV 覆盖，摘要与逐臂表不匹配。

有效结果位于：

- `output/experiments/M1_AMCM_ADAPTIVE_FIRST_UPDATE_CAUSAL_DIAG_20261021_RETRY4/{per_arm.csv,selected_events.json,summary.json}`
- `output/experiments/M1_AMCM_ADAPTIVE_INITIAL_UPDATE_32SEQ_20261021/{per_arm.csv,selected_events.json,sequence_list.txt,summary.json,causal_analysis.json}`
- runner：`tools/run_m1_adaptive_first_update_causal_diagnostic_20261021.py`
- aggregator：`tools/analyze_m1_adaptive_first_update_causal_diagnostic_20261021.py`

## 路线判断

**已排除（在本 train-side 样本范围内）**：adaptive residual 对常规首更新/所选低 IoU 状态造成可测的 CENTER peak 翻转；AMCM 以外的 checkpoint 参数漂移。

**暂不支持**：当前 3 epoch 无监督 AMCM 已学到足够强的目标条件修正；简单放大现有 residual 就能成为稳定贡献。

**尚未区分**：测试集 catastrophic 首更新为何只在某些分布下出现。由于候选 checkpoint 与 M1 除新增 AMCM 参数外完全相同，完整系统回归确实由 AMCM 启用路径触发；但 residual 很小且 train-side 没有复现崩溃，提示其影响可能集中在测试分布中的低 margin/近似平局峰值，随后经 CENTER 尺寸解码与 crop/state 闭环放大。这是待检验解释，不是结论。

## 下一步方向

暂停训练和幅值搜索。先构造一个 outcome-blind train-side 首更新诊断 cohort：从独立序列按目标面积、search-crop 边缘距离和 M1 CENTER top1-vs-top2 margin 分层抽样，锁定至少 50 个状态；其中低 margin 状态必须在选择时不看 adaptive 输出。对 M1/adaptive/置零三臂记录未经 Hann 的 score、Hann 后 GT/最强背景 margin、top-1/top-2 margin、top cell、该 cell 的 size/offset、解码框及裁剪前后状态。仅在低 margin 层额外记录 residual=1 与 4 倍的有限差分响应，用于判断有效变化是否发生在 CENTER argmax 翻转或 size/offset 解码；该倍率只作固定状态敏感度诊断，不能作为性能候选或据此挑阈值。

若有限差分显示仅少数低 margin 状态发生 top-1 翻转且尺寸解码异常，模块贡献应转向“定位质量校准 + 可信接纳/拒绝”，保留 global/foreground 双证据作为提议，避免熵值直接决定 residual。若增幅只能改变 feature margin、不能得到可靠且可校准的定位质量信号，停止 pre-fusion residual 路线，改查 CENTER score 与 size/offset 接口及状态提交策略。进入训练前先证明质量信号在固定训练样本上可拟合，并以独立序列验证 no-op 控制和低 margin 覆盖；当前证据不支持直接长训或 full test。

## 限制

- train-side trace 来自既有 manifest；M1 checkpoint 是否见过这些序列无法确认。
- 12 个失败/稳定状态是机制样本，不是总体估计；32 个首次更新全部稳定，不能代表测试集困难分布。
- 固定状态不等于闭环轨迹，不报告 AUC/Success 增益。
- 3 epoch candidate 与 M1 epoch15 训练进度不匹配；checkpoint 参数审计排除了主干权重漂移，但不补足多 seed 或完整训练比较。
