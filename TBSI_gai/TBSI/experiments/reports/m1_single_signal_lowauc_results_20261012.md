# M1 单信号 search pre-fusion 闭环对比（2026-10-12）

## 结论

在按归档 M1 AUC<40 预先锁定的 LasHeR test 62 序列上，三个冻结单信号算子相对**同一 M1 epoch-15 权重、当前源码重跑**的匹配 M1，对序列宏平均 AUC 分别为 +1.54、+2.53、+2.35 点。不过三个配对 AUC bootstrap 95% 区间均跨零；train-side 24 条留出序列的同 crop 框 IoU 准入也全部失败。当前结果是探索性闭环变化，**不支持信号已具备稳定任务判别性，也不能证明融合是 M1 失效首因**。按预定停止规则，不继续叠加信号或搜索更复杂的融合算子。

## 四行实测表

各指标为 62 序列宏平均、百分制；括号是相对当前源码匹配 M1 的配对差值。AUC 的区间按序列重采样 5,000 次，不能解释为独立帧区间。

| 条件 | AUC / Success | PR | NPR | OP50 | OP75 | AUC 胜/负/平 | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 匹配 M1 | 22.65 | 28.12 | 25.29 | 25.90 | 14.28 | 基准 | 30.56 |
| S1 模板条件响应 | 24.19 (+1.54; CI [-0.28, 3.86]) | 30.25 (+2.13) | 27.20 (+1.92) | 27.65 (+1.75) | 15.08 (+0.81) | 40/20/2 | 29.45 |
| S2 预测 CENTER 框效用 | 25.18 (+2.53; CI [-0.15, 5.85]) | 31.30 (+3.18) | 28.45 (+3.16) | 29.11 (+3.21) | 16.48 (+2.20) | 33/29/0 | 29.90 |
| S3 退化/几何量 | 25.00 (+2.35; CI [-0.10, 5.43]) | 31.04 (+2.92) | 27.94 (+2.65) | 28.70 (+2.79) | 16.12 (+1.84) | 35/26/1 | 28.62 |

归档 M1 在相同**归档定义**子集上的 AUC 约 22.18，仅供历史参照；当前源码逐帧轨迹并不与归档一致，不把它与上述候选直接相减，也不把该表称为 full-244 或严格归档复现。

## 信息流与序列审计

每臂均完成 62,496 帧，信息流在首帧及每 32 帧采样，共 1,974 点；下表均为先按序列再平均。`GT top-1` 是采样状态的原 CENTER Hann 选格落在 GT 格的比例；`rank` 越小越好。各臂在线状态和 crop 已可因先前预测分叉，**这些信息流量不是同 crop 因果对照**。

| 条件 | GT top-1 | GT Hann rank | 峰距 GT (token) | 采样框 IoU | RGB/TIR/fused 范数 | RGB 增强占比 |
|---|---:|---:|---:|---:|---|---:|
| 匹配 M1 | 9.24% | 111.31 | 9.95 | 0.259 | 169.91 / 153.67 / 289.72 | NA |
| S1 | 9.92% | 104.77 | 9.24 | 0.275 | 152.93 / 168.62 / 288.70 | 17.48% |
| S2 | 11.02% | 103.74 | 9.00 | 0.287 | 165.85 / 156.73 / 291.99 | 46.40% |
| S3 | 11.37% | 106.09 | 9.29 | 0.283 | 174.47 / 149.32 / 294.14 | 64.89% |

三个信号的路由方向分布差异很大，却都给出正的 AUC 均值；这降低了“某一信号准确选择可靠模态”的解释优先级。范数变化是算子施加缩放的直接结果，不是单模态信息增加的证据。S1/S2/S3 的 10% 截尾 AUC 增益仅 +0.91/+0.31/+0.63；中位差 +0.20/+0.05/+0.11。S1 的 `3bike1` 一条 +56.29，移除后其余 61 条平均 +0.64；S2 的 `3bike1` +55.47、`3rdfatboy` +53.29，同时 `basketballathand` -20.01；S3 的 `blktribikecome` +60.53、`3bike1` +55.47，同时 `shinycarcoming` -17.34。完整逐序列原始值在各臂 `per_sequence.csv`，配对统计在 `paired_analysis/summary.json`。

归档场景属性的重叠切片上，AUC 配对差如下；属性只描述序列，不代表每帧失败原因：

| 属性（序列数） | S1 | S2 | S3 |
|---|---:|---:|---:|
| TC (50) | +0.93 | +1.20 | +1.20 |
| SA (38) | +1.42 | +1.66 | +0.08 |
| LR (26) | +0.19 | -0.63 | +3.32 |
| LI (17) | -0.47 | -1.67 | +0.04 |

S2 并未在其预期的 LR 切片受益，S3 在 LI 切片几乎持平；不能借全组均值确认原信号假设。严重失配帧的 train-side 同 crop 筛选在 72 序列、144 状态上完成，48 序列拟合、24 留出；留出序列宏平均 IoU 相对不动作：S1 -0.0039、S2 -0.0086、S3 -0.0039，三动作事后 oracle 仅 +0.0135。三者均未达到预设 +0.03 且区间下界大于零的准入门。因用户要求，仍固定信号/±15% 算子各做一次上述 test 子集闭环探索，未根据 test 结果调参。

## 合同、归因与下一步

- 唯一被替换的候选变量是最终 RGB/TIR search token 拼接前的标量路由信号；共同算子为 `RGB*(1+0.15*sign(g))`、`TIR*(1-0.15*sign(g))`。不改 MHSA、`tbsi_fuse_search`、CENTER 或 M1 权重；默认关闭路径在 3,377 帧预检中与改动前预测文件 SHA256 完全相同。
- **已排除**：当前三个冻结信号在预设独立 train-side 固定状态合同下通过准入；同一候选在这 62 序列上的稳定、均匀收益。**暂不支持**：单信号引导的 ±15% pre-fusion 算子已经解决 M1 低 AUC 的主要原因。**尚未区分**：一般性 RGB/TIR 常数缩放、信号方向、离散闭环轨迹分叉和场景分布差异各自的贡献；本次没有完整 test-side 常数缩放臂，也没有 full-244 正常序列哨兵集。
- 下一步应先在独立 train-side 状态上做同 crop、相同 ±15% 幅度的固定 RGB/TIR 两个常数动作和信号动作对照，并为大幅正负的序列审计首次轨迹分叉、GT 是否仍在 crop、Hann 峰、offset/size 解码。若不能超过简单常数或损伤正常帧，停止此位置的信号门控；必要时转查 CENTER 读出和闭环状态。不能用本 62 序列选择新阈值后再报告同一子集为验证集。

## 复现资产

- Parent：`TBSI_E_MHSA_FULL_4090D_seed42` epoch15；受保护 YAML SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`；checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`，二者结束时复核未变。Git HEAD `09f3d57c`；候选完整源码快照 `experiments/snapshots/m1_single_signal_20261012_candidate/source_candidate_v2.tar.gz`，SHA256 `f0f7d78790218e96a7ded2c0b27208a66a3fca527f5943090b0ac20914c5ddc9`。
- 数据：LasHeR testingset，`/root/autodl-tmp/TBSI_gai/TBSI/data/lasher/testingset`；由归档 M1 AUC<40 锁定的 62 序列，成员清单见 `output/experiments/M1_SINGLE_SIGNAL_LOW_AUC_20261012/matched_m1_full62/manifest.json`，SHA256 `3e71584650ff481ef0e968c54886ca7acb486b2ad724a508c58d52ef393ec708`。train-side 校准见 `train72_screen_v1/summary.json`，SHA256 `c37df223d44fc90e36c20e15f5dfbdf35bdddb0bb4ebfcd18d083e1bd7e5e291`；M1 训练暴露不能据现存资产完全确定。
- 环境：1×NVIDIA GeForce RTX 4090 D 24 GiB，Python 3.8.20，PyTorch 1.9.0+cu111，CUDA 11.1，FP32；每次只运行 1 个跟踪进程。S1/S2/S3 用时 2163.9/2130.9/2232.3 秒，约 0.6011/0.5919/0.6201 GPU 小时。62 序列的 FPS 是序列宏平均；没有并行评测。
- 完整调用分别为 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/autodl-tmp/conda_envs/tbsi/bin/python -u tools/run_m1_single_signal_lowauc_20261012.py --output-dir output/experiments/M1_SINGLE_SIGNAL_LOW_AUC_20261012/<arm>_full62 --max-sequences 62 --signal <s1|s2|s3> --artifact-dir output/experiments/M1_SINGLE_SIGNAL_LOW_AUC_20261012/train72_screen_v1`；工作目录为 `TBSI_gai/TBSI`，三个 arm 各运行一次。分析命令为 `/root/autodl-tmp/conda_envs/tbsi/bin/python tools/analyze_m1_single_signal_lowauc_20261012.py --root-dir output/experiments/M1_SINGLE_SIGNAL_LOW_AUC_20261012 --output-dir output/experiments/M1_SINGLE_SIGNAL_LOW_AUC_20261012/paired_analysis`。原始预测、逐序列指标、采样信息流、完整汇总和四行 CSV 均在上述独立输出目录；分析汇总 SHA256 `215df113b4c061aea9326b170ecfc3f78aba64bd7af5afe890877c966d2997b9`。
