---
name: tbsi-pipeline
description: "TBSI 两阶段训练→测试流水线。Phase A: 训练（前置检查→清理→训练→验证checkpoint）。Phase B: 测试（清理残差进程→设置线程→测试→验证结果→分析）。Use when user says \"跑流水线\", \"train and test\", \"两阶段\", \"tbsi pipeline\", \"训练测试\", \"训完了测一下\", or wants to run the full TBSI train+eval cycle."
argument-hint: [config-name]
allowed-tools: Bash(*), Read, Write, Grep, Glob, Agent, Skill
---

# TBSI 两阶段流水线: $ARGUMENTS

运行 TBSI 训练→测试完整流水线，配置文件: **$ARGUMENTS**

## 核心原则

1. **一次只跑一个测试** — 串行执行，绝不并行启动多个 test.py 实例
2. **单进程模式** — 用 `--threads 0`，不用 multiprocessing，避免孤儿进程
3. **训练/测试分离** — 训完再测，测完再训下一个，绝不交叉
4. **环境隔离** — 每个 step 开始前检查 GPU 空闲、无残留进程

## Constants

- **TBSI_DIR = `/root/autodl-tmp/TBSI_gai/TBSI`** — TBSI 代码根目录
- **CONDA_ENV = `/root/autodl-tmp/conda_envs/tbsi`** — Conda 环境
- **DATA_DIR = `/root/autodl-tmp/TBSI_gai/data/lasher/`** — LasHeR 数据集路径
- **PRETRAINED_DIR = `/root/autodl-tmp/TBSI_gai/pretrained/`** — 预训练模型路径
- **OUTPUT_DIR = `output/checkpoints/train/tbsi_track/<config>/`** — 训练输出相对路径
- **TEST_OUTPUT_DIR = `output/test/tracking_results/tbsi_track/<config>/`** — 测试输出相对路径
- **ANALYSIS_SCRIPT = `tracking/analysis_results.py`** — 结果分析脚本
- **LOG_DIR = `output/logs/`** — 训练/测试日志输出目录（所有记录均写到此目录）
- **TEST_SCRIPT = `tracking/test.py`** — 测试脚本
- **TRAIN_SCRIPT = `tracking/train.py`** — 训练脚本
- **DATASET = `lasher_test`** — 测试集名称
- **CHECKPOINT_EPOCH = 4** — 验证 checkpoint 的 epoch 编号
- **THREADS = 4** — OMP/MKL 线程数（测试时）
- **AUTO_PROCEED = true** — 设置为 `false` 时，每阶段结束暂停询问是否继续
- **COMPACT = true** — 设为 `false` 时输出更详细的日志
- **TOTAL_EXPECTED_FILES = 488** — 测试应输出文件数（244 序列 × 2）
- **BASELINE_TRAIN_TIME = "~18 min"** — 4 epoch 训练参考时长
- **BASELINE_TEST_TIME = "~35 min"** — 244 序列测试参考时长
- **BASELINE_FPS = 31** — 参考 FPS

> 用法: `/tbsi-pipeline "vitb_256_tbsi_sprint_da_ch"` (不带 `—` 参数则使用默认值)

## Workflow

### Phase 0: Pre-flight (环境检查)

```bash
cd {TBSI_DIR}

# 1. 检查 GPU 状态
nvidia-smi

# 2. 检查残留 Python 进程
ps aux | grep python | grep -v grep | grep -v jupyter | grep -v tensorboard

# 3. 确认 Conda 环境可用
{BASE_CONDA}/conda shell.bash hook && conda activate {CONDA_ENV}
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

**检查结果:**
- 如果 GPU memory-used > 500 MiB → 需确认无其他训练占用
- 如果有残留 Python 进程 → 提示用户清理或自动执行 Phase 0.5
- 如果 Conda 环境不可用 → 报错退出

### Phase 0.5: 进程清理 (按需)

当有残留进程时：

```bash
# 1. 杀掉 test.py 进程
pkill -f "test.py tbsi_track" 2>/dev/null || true

# 2. 杀掉 multiprocessing.spawn 残留
ps aux | grep "multiprocessing.spawn" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true

# 3. 确认 GPU 空闲
nvidia-smi
```

**确认标准:** GPU memory-used 回落到 ~1 MiB 或 ~5000 MiB（显存常驻程序），`ps aux | grep python | grep -v grep | grep -v jupyter | grep -v tensorboard` 无输出。

---

### Phase A: 训练 (Training)

#### A1: 清理历史训练输出 (按需)

```bash
rm -rf output/checkpoints/train/tbsi_track/{config}/
rm -rf output/tensorboard/tbsi_track/{config}/
```

> 📌 如需从零开始训练则执行，否则跳过。当 `AUTO_PROCEED = true` 时默认清理。

#### A2: 启动训练（单 GPU 模式）

```bash
unset OMP_NUM_THREADS

python {TRAIN_SCRIPT} \
    --script tbsi_track \
    --config {config} \
    --mode single 2>&1 | tee -a {LOG_DIR}/train_{config}.log
```

使用 `run_in_background: true` 启动。训练脚本自动将详细 log 写入 `{LOG_DIR}/tbsi_track-{config}.log`，此处同时将 stdout/stderr 写入 `{LOG_DIR}/train_{config}.log` 以便回溯终端日志。训练过程中：
- 提供进度估算（参考: baseline ~18 min for 4 epoch）
- 训练完成后自动进入 Phase A3

**🚦 Checkpoint (when AUTO_PROCEED = false):**
```
训练完成。Checkpoint 验证通过。
进入 Phase B（测试）？ (Y/N)
```

#### A3: 验证 Checkpoint 完整性

```bash
ls -lh {OUTPUT_DIR}/TBSITrack_ep{CHECKPOINT_EPOCH}.pth.tar
```

验证文件存在且大小合理（> 100 MB）。如果 checkpoint 不存在或为空 → 报错，不进入测试阶段。

---

### Phase B: 测试 (Testing)

#### B0: 前置清理

```bash
# 最关键的一步：检查 multiprocessing.spawn 残留
ps aux | grep python | grep -v grep | grep -v jupyter | grep -v tensorboard
# 如果看到以下进程，手动 kill：
#   - "from multiprocessing.spawn"  ← 测试子进程
#   - "tracking/test.py"            ← 旧测试主进程
# 自动清理：
ps aux | grep "multiprocessing.spawn" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
pkill -f "test.py tbsi_track" 2>/dev/null || true
```

#### B1: 设置线程数（防止 CPU 过竞争）

```bash
export OMP_NUM_THREADS={THREADS}
export MKL_NUM_THREADS={THREADS}
```

#### B2: 启动测试（单进程，单 GPU，无 multiprocessing）

```bash
python {TEST_SCRIPT} \
    tbsi_track {config} \
    --dataset_name {DATASET} \
    --threads 0 \
    --num_gpus 1 2>&1 | tee -a {LOG_DIR}/test_{config}.log
```

使用 `run_in_background: true`。测试时长参考：~35 min（244 序列）。测试输出同时写入 `{LOG_DIR}/test_{config}.log`。

#### B3: 验证结果完整性

```bash
ls {TEST_OUTPUT_DIR}/*.txt | wc -l
```

**验证标准:**
- 应输出 `{TOTAL_EXPECTED_FILES}` 个文件（244 序列 × 2 = 结果 + 时间文件）
- 如果文件数不匹配 → 输出警告，但仍尝试分析
- 常见原因：测试中途中断、磁盘 I/O 问题

#### B4: 分析结果

```bash
python {ANALYSIS_SCRIPT} \
    --tracker_name tbsi_track \
    --tracker_param {config} \
    --dataset_name {DATASET} 2>&1 | tee -a {LOG_DIR}/analysis_{config}.log
```

分析结果同时写入 `{LOG_DIR}/analysis_{config}.log`。

---

### Phase C: 结果汇总

分析完成后，输出结果摘要：

```
📊 TBSI 流水线完成 — {config}

## Phase A: 训练
- Status: ✅ (或 ❌)
- Checkpoint: {OUTPUT_DIR}/TBSITrack_ep{CHECKPOINT_EPOCH}.pth.tar
- Duration: ~{实际时长 min}

## Phase B: 测试
- Status: ✅ (或 ❌)
- 结果文件: {TOTAL_EXPECTED_FILES} / {TOTAL_EXPECTED_FILES} 预期
- Duration: ~{实际时长 min}

## 关键指标
| Metric | Value |
|--------|-------|
| SR (Success Rate) | ... |
| PR (Precision Rate) | ... |
| NPR (Normalized Precision) | ... |

## 性能
- Test FPS: ... (baseline: {BASELINE_FPS})
```

---

## 进程清理 SOP

当测试异常中断后，必须执行以下清理步骤：

```bash
# 1. 杀掉所有 test.py 进程
pkill -f "test.py tbsi_track"

# 2. 杀掉所有 multiprocessing.spawn 残留（最重要！）
#    这些进程会持续占用 CPU 并拖慢后续测试
ps aux | grep "multiprocessing.spawn" | grep -v grep | awk '{print $2}' | xargs kill -9

# 3. 确认 GPU 彻底空闲
nvidia-smi
# 确认 memory-used 回落到 ~1 MiB 或 ~5000 MiB（显存常驻程序）

# 4. 最后检查
ps aux | grep python | grep -v grep | grep -v jupyter | grep -v tensorboard
# 应无输出
```

---

## 性能基线参考

| 实验 | 训练时长 (4 epoch) | 测试时长 (244 seq) | 测试 FPS |
|:----|:-----------------:|:------------------:|:--------:|
| channel 基线 | ~18 min | ~35 min | ~31 |
| MADC | ~18 min | ~35 min | ~31 |
| CSR | ~18 min | ~35 min | ~33 |

> 注：测试时长存在波动（取决于磁盘 I/O，磁盘占用 >90% 时速度下降约 30%）

## 实验配置速查

| 实验名 | 配置 yaml | 说明 |
|:------|:---------|:-----|
| `vitb_256_tbsi_sprint_da_ch` | channel baseline | 空间+通道融合 |
| `vitb_256_tbsi_sprint_da_ch_madc` | channel + MADCV1 | 幅度归一化校准 |
| `vitb_256_tbsi_sprint_da_ch_csr` | channel + CSR | 互补调制 |
| `vitb_256_tbsi_sprint_da_ch_full` | channel + MADCV1 + CSR | 联合方案 |

## Key Rules

- **所有记录必须写入 `output/logs/`** — 训练（`train_{config}.log`）、测试（`test_{config}.log`）、分析（`analysis_{config}.log`）的 stdout/stderr 均需重定向到 `{LOG_DIR}`。训练脚本自身的 `{script}-{config}.log` 也在此目录，保存完整训练过程
- **一次只跑一个测试** — 绝不并行启动多个 test.py 实例
- **单进程模式** — 始终用 `--threads 0`，避免 `multiprocessing.spawn` 残留
- **训练/测试严格分离** — 必须在训练完全结束后才能开始测试
- **测试前必须清理残留** — multiprocessing.spawn 进程会占用 CPU 并拖慢测试
- **磁盘 I/O 感知** — 磁盘占用 >90% 时测试速度下降约 30%，需提前提示
- **如果某个测试异常中断** — 调用 SOP 清理后再重试，不要直接重跑
- **checkpoint 验证** — 测试前确认 checkpoint 文件存在且大小合理

## Composing with Other Skills

```
/tbsi-pipeline "vitb_256_tbsi_sprint_da_ch"    ← 两阶段训练→测试流水线
/analyze-results                                ← 深入分析实验结果
/auto-review-loop "TBSI CASTBlock"              ← Workflow 2: review + iterate
/paper-writing "NARRATIVE_REPORT.md"            ← Workflow 3: write the paper

Or use /research-pipeline for the full end-to-end flow.
```
