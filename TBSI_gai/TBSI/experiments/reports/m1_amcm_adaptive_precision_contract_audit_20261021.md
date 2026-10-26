# Adaptive AMCM 半精度合同与 full-244 复核

## 归因问题

此前 adaptive full-244 保存轨迹的 AUC 为 43.65，48 条严重退化序列常在首次更新输出边界小框。固定状态 FP32 诊断却显示 AMCM residual/input L2 比约 `1e-4`，CENTER 排序几乎不变。两种证据矛盾，本次先核对推理精度与轨迹可复算性。旧轨迹及报告均保留，不覆盖。

## 50 条独立 train-side 首更新

从 LasHeR trainingset 977 条候选序列中排除原冻结 32 条，以固定 SHA256 顺序抽取 50 条，选择时不查看 adaptive 结果。`per_arm.csv` 为 50 状态 × M1/adaptive/zero/reverse/shift = 250 行，清单及规则见 `sample_manifest.json`。M1 首更新 IoU 均值 `0.862849`，最小 `0.566011`；50/50 GT rank=1，M1 GT-hard-background margin 最小/中位/最大 `0.432081/0.739883/0.947005`。adaptive 对 zero 的 rank 改变 `0/50`、top-1 cell 翻转 `0/50`，平均绝对 IoU 差 `9.21e-5`、最大 `3.62e-4`；RGB/TIR residual/input 比均值 `1.78e-4/3.71e-4`。该随机首更新样本没有低 margin 状态，不能验证困难帧的目标选择性。

证据：`output/experiments/M1_AMCM_ADAPTIVE_INITIAL_UPDATE_50SEQ_20261021/{per_arm.csv,selected_events.json,sequence_list.txt,sample_manifest.json,summary.json}`。

## testingset 首更新精度复算

选 3 条旧轨迹首次更新崩溃序列 `bikeboyturntimes`、`blackboy`、`10runone`，另选旧轨迹正常的 `runningcameragirl` 作控制。只审计保存轨迹 provenance，不用 testingset 选择模型参数。同一初始化、帧、state、checkpoint 下比较 M1 FP32、adaptive FP32、adaptive residual=0、M1 FP16 和 adaptive FP16。

| 序列 | 旧 adaptive 首更新 IoU | adaptive FP32 复算 IoU | 修复前 adaptive FP16 IoU | 修复后 adaptive FP16 IoU |
|---|---:|---:|---:|---:|
| bikeboyturntimes | 0 | 0.9079 | 0 | 0.9079 |
| blackboy | 0 | 0.9778 | 0 | 0.9785 |
| 10runone | 0 | 0.8739 | 0 | 0.8737 |
| runningcameragirl | 0.9535 | 0.9541 | 0 | 0.9542 |

修复前，四条序列的 adaptive FP16 熵/相似度路径出现 NaN，框均为 `0,0,10,10`。前三条与旧官方保存首框完全一致（最大 bbox 差 `0`），第四条旧保存框却正常，说明旧 full-244 产物的精度/代码口径无法视为单一、已验证条件。FP32 adaptive 复算四条均正常；M1 FP16 也正常。此处 AMCM 的启用路径是 NaN 来源，但旧 full-244 的精确逐序列运行口径尚未从日志完全重建。

算子首因在 `AMCMLiteReliability._template_token_match` 的 entropy：FP16 中 `clamp_min(1e-8)` 的阈值下溢为 0，softmax 的零概率进入 `0 * log(0)`，产生 NaN。NaN 进入 confidence、template matching、融合特征及 CENTER size/offset；GT rank 的 NaN 输出不能解释为“正确选峰”。最小修复只把 entropy 的乘法、log 与归一化放到 FP32，再转回原 dtype；没有改变模型参数或其他分支。修复前后四条 FP32 adaptive 的 IoU 差均为 0；修复后四条 FP16 输出有限，最大相对 M1 FP16 首框 IoU 差 `0.000225`。

证据：`output/experiments/M1_AMCM_ADAPTIVE_TEST_FIRST_UPDATE_PRECISION_REPLAY_RETRY2_20261021/{per_arm.csv,summary.json}` 与 `...PRECISION_REPLAY_ENTROPYFIX_20261021/{per_arm.csv,summary.json}`。第一次 precision replay 因 runner 重复构造 M1 时全局配置残留而在预测前失败，目录 `...PRECISION_REPLAY_20261021/` 保留；RETRY2 按 M1、M1、adaptive、adaptive 顺序构建完成。修复前模块快照为 `experiments/snapshots/amcm_reliability_pre_fp16_entropy_fix_20261021.py`，SHA256 `c07f640823722d25a8394ee65b2f4f416b961ab4534ae7476e320a18ee6ff572`；修复后模块 SHA256 `38f46bd8514a2bd9068328a67b8c7ee34bd7da7cc2df389908933059a3b922ad`。

## 决策

**已排除**：旧 full-244 的 `43.65` 可作为当前 adaptive 算子在匹配 FP32 口径下的有效负结果；这四条首更新崩溃由 GT/hard-background 排序失效造成。

**支持**：AMCM adaptive 熵的 FP16 数值合同是可复现的首更新 NaN 机制；局部 FP32 熵修复消除该机制。50 条 train-side 常规首更新上当前 residual 接近 no-op，尚无目标判别性收益证据。

**尚未区分**：修复后完整闭环性能、困难低 margin 状态的收益与 harm、不同 seed 的可迁移性。修复前的 4 条 test 首帧只用于 provenance 审计，不用于挑选阈值。

独立 FP32 串行 full-244 复核使用 `M1_AMCM_ADAPTIVE_FP32_REAUDIT_FULL244_20261021.yaml`，与原 candidate YAML 的解析结构完全相同；显式 `TBSI_INFER_FP16=0`、原 epoch3 checkpoint SHA256 `15e0bf9b2a489cecd24f1d4994a8b44a535aae2c689eca2ba3e2be237981a1b2`、LasHeR `testingsetList.txt` SHA256 `16eef326975aacccb94a7e133e8ef0aceb734ece4264caf6f82bc94442c20e46`、串行 threads=0。新 config SHA256 `0b0e53b615314e71edb2432f775b7c86186d47cafed0f0c43dd7cd955ed6dadf`；Git HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9` 加上述模块快照/哈希。1× NVIDIA GeForce RTX 4090 D，Python 3.8.20，PyTorch 2.0.0+cu118，CUDA 11.8。该全量复核运行中，结果完成后补写；当前不宣称性能提升。

执行目录为 `TBSI_gai/TBSI`，完整跟踪命令：

```bash
TBSI_INFER_FP16=0 TBSI_CHECKPOINT_OVERRIDE=/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/M1_AMCM_DR_ADAPTIVE_NOSUP_QUICK15_3EP_20261021/checkpoints/TBSITrack_ep0003.pth.tar python tracking/test.py tbsi_track M1_AMCM_ADAPTIVE_FP32_REAUDIT_FULL244_20261021 --dataset_name lasher_test --threads 0 --num_gpus 1
```

跟踪标准输出保存为 `output/experiments/M1_AMCM_ADAPTIVE_FP32_REAUDIT_FULL244_20261021/full244_fp32.log`。结束后用 `python tools/evaluate_m1_amcm_adaptive_fp32_reaudit_20261021.py`，独立 evaluator cache 名为 `M1_AMCM_ADAPTIVE_FP32_REAUDIT_FULL244_20261021`。

### 线程合同更正

上述首次 FP32 full-244 尝试在 57/244 时中断。虽然 `--threads 0`，启动命令遗漏了 M1 系列全量脚本使用的 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`；实际进程约150线程、RSS约41GB，217帧序列耗时异常。57条的114个 bbox/time 文件和完整日志留在旧目录，不进入最终评测。该中断是运行合同修正，不是因为负指标或运行时间而停止。

独立重跑配置 `M1_AMCM_ADAPTIVE_FP32_OMP1_FULL244_20261021.yaml` 与原 candidate YAML 的解析结果完全相同，SHA256 `28b6f5802f6b11586e9176c93ebeeec92c9da28690746b021f9449dd3be7c03d`。从 0/244 开始，命令为：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 TBSI_INFER_FP16=0 TBSI_CHECKPOINT_OVERRIDE=/root/autodl-tmp/TBSI_gai/TBSI/output/experiments/M1_AMCM_DR_ADAPTIVE_NOSUP_QUICK15_3EP_20261021/checkpoints/TBSITrack_ep0003.pth.tar python tracking/test.py tbsi_track M1_AMCM_ADAPTIVE_FP32_OMP1_FULL244_20261021 --dataset_name lasher_test --threads 0 --num_gpus 1
```

新日志为 `output/experiments/M1_AMCM_ADAPTIVE_FP32_OMP1_FULL244_20261021/full244_fp32_omp1.log`；结束后使用同一个分析脚本，独立 evaluator cache 为 `M1_AMCM_ADAPTIVE_FP32_OMP1_FULL244_20261021`。新运行进行中。

### 用户要求停止后状态（2026-10-21）

用户要求先停止实验、分析自适应混合路线。向本次进程发送 SIGINT 后，已核实 PID 31528 不存在，未见跟踪或训练实验进程。本次独立 FP32/OMP1 运行保存 8/244 条序列、16 个 bbox/time 文件，日志以 KeyboardInterrupt 结束；没有完整指标，不能与此前 57/244 的独立运行拼接。上述两次中断均保留为运行证据，权威资产主表与 CSV 状态同步更新为 INTERRUPTED。
