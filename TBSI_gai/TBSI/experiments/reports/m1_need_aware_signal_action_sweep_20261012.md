# M1 需求感知动作效用与信号筛选（2026-10-12）

运行于 2026-10-12 启动、2026-10-13 完成；实验 ID 与预注册协议日期保持冻结。

## 结论

**判定：当前输入信号未通过“need-aware”准入；不训练、也不接入后融合校正算子。** 本实验在相同固定 M1 crop/state 上直接比较 RGB/TIR 小剂量缩放与 no-op 的真实 CENTER 解码框 IoU，并用留一序列 ridge 选择动作。原始信号模型的序列宏平均 utility 为 `+0.0017`，95% sequence-bootstrap CI `[-0.0028,+0.0093]`，没有可靠正收益；其覆盖率为 100%，但 57 个 oracle-no-op 状态中拒绝干预数为 0，11.5% 的干预状态造成超过 0.01 IoU 的损失。

加入 CENTER 风险特征并使用训练折残差 RMSE 构造保守 LCB 后，gate 拒绝了 56/57 个 oracle-no-op 状态（specificity `98.2%`），但只干预 `1/104` 状态；它在 4 个 severe-onset 状态全部 abstain，漏掉了两个真实高收益动作。risk-coverage 曲线在目标 5%、10%、20% 时都只能覆盖 `1/104`，没有显示可用效用。因此当前证据支持“保守拒绝能压低干预伤害，但尚不能同时找到需要救援的状态”，不支持上线 gate 或新融合算子。

## 实验合同与复现

- Parent：受保护 `TBSI_E_MHSA_FULL_4090D_seed42` epoch 15；M1 YAML 与 checkpoint 实验前后 SHA256 仍分别为 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1`、`1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`。
- 数据：LasHeR trainingset，32 条 outcome-blind 40–149 帧短序列，清单 3,224 帧；状态采样共 104 个、32 条序列。分层计数：64 probe-reference、28 good-only、4 matched-good、4 pre-onset、4 severe-onset。此 cohort 独立于记录到的先前信号诊断名单，但 M1 训练序列暴露无法从现有日志恢复，不能称为独立于 M1 训练集。
- 每个状态固定同一 `tracker_state_before`、图像 crop、模板、保存特征、M1 fusion convolution、CENTER score/offset/size、Hann 与 decoder；动作是 no-op、RGB/TIR 互反缩放 rho `0.05/0.10/0.15` 和 seed 3407 的随机方向 rho `0.10`。utility 标签为动作 decoded bbox IoU 减同状态 no-op decoded bbox IoU。没有训练权重、改 MHSA 或执行闭环 rollout。
- 信号模型使用固定 alpha=10 的 ridge；按 sequence leave-one-out；`prefusion_signals` 用亮度/对比、模板响应与模态分歧、局部能量、尺度/运动等部署可见量；`prefusion_plus_center_risk` 再加入 CENTER entropy/top margin。区间以序列为 bootstrap 单位，10,000 次、seed 3407。残差 RMSE×1.645 仅是未校准 uncertainty proxy。
- 环境：1× NVIDIA GeForce RTX 4090 D 24 GiB，Python 3.8.20、PyTorch 2.0.0+cu118、CUDA 11.8，FP32。trace 102.75 s、feature capture 17.23 s、action replay 9.88 s，GPU 总耗时约 0.036 小时；分析 CPU-only。
- Git HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`。本轮最终 source snapshot `experiments/snapshots/m1_need_aware_action_sweep_20261012_sources_v5.tar.gz` SHA256 `0a6010e2c52180ceba84e756e3fc39a71cdd5e8b4fe00adbe142c92a097842df`。cohort manifest SHA256 `96dc80aae880b308060010a5069608afbb163ddbd69ec9953826f434a0379680`；trace manifest `622bbc829c50dd5353aa40b0c6c8ab37cda0312ad89b6e401c966c49e4bd994f`；selected features `c5bed04d14eaecbd20fcdd031deca72253f1547183086b5412659120dc4b37f0`；signals/actions CSV SHA256 分别 `6cafe7be3cb96d0cf23b77daba37ffec1ffc2b7e6c84da8457510386f802603c`、`e22da3970d0b5b00d5140d0c573fc1935273e30392267083102d9ab5ced239b1`；final summary SHA256 `d35b6532513616c7773e921a05897c33e6d1d25ac4f852bd3814e58a085643ad`。
- 最终复现命令：

```bash
TBSI_INFER_FP16=0 python3.8 tools/run_m1_search_stage_train_trace_20261009.py \
  --output-dir output/experiments/M1_NEED_AWARE_ACTION_SWEEP_20261012/trace_run \
  --sequences 32 \
  --sequence-list experiments/diagnostics/m1_need_aware_action_sweep_20261012/sequence_list.txt \
  --allow-short-sequences
TBSI_INFER_FP16=0 python3.8 tools/diagnose_m1_search_stage_20261009.py \
  --trace-dir output/experiments/M1_NEED_AWARE_ACTION_SWEEP_20261012/trace_run/trace \
  --output-dir output/experiments/M1_NEED_AWARE_ACTION_SWEEP_20261012/feature_capture \
  --no-perturb --probe-reference-frames 2 --selection-rule onset-3-iou-0.5 \
  --diagnostic-id M1-NEED-AWARE-ACTION-SWEEP-20261012 --capture-template-tokens
TBSI_INFER_FP16=0 python3.8 tools/replay_m1_need_aware_action_sweep_20261012.py \
  --features output/experiments/M1_NEED_AWARE_ACTION_SWEEP_20261012/feature_capture/selected_features.npz \
  --center-csv output/experiments/M1_NEED_AWARE_ACTION_SWEEP_20261012/feature_capture/per_arm.csv \
  --trace-dir output/experiments/M1_NEED_AWARE_ACTION_SWEEP_20261012/trace_run/trace \
  --output-dir output/experiments/M1_NEED_AWARE_ACTION_SWEEP_20261012/action_replay
python3.8 tools/analyze_m1_need_aware_action_sweep_20261012.py \
  --signals output/experiments/M1_NEED_AWARE_ACTION_SWEEP_20261012/action_replay/per_state_signals.csv \
  --actions output/experiments/M1_NEED_AWARE_ACTION_SWEEP_20261012/action_replay/per_state_actions.csv \
  --output-dir output/experiments/M1_NEED_AWARE_ACTION_SWEEP_20261012/analysis_final
```

The first trace launch exposed a 150–800 frame runner restriction inconsistent with the frozen 40–149 frame cohort. An opt-in flag was added; default runner behavior remains unchanged. Two analyzer startup/serialization failures and a protocol-output omission were also fixed and retained in `output/experiments/M1_NEED_AWARE_ACTION_SWEEP_20261012/startup_attempts.md`. Partial outputs under `analysis/` and `analysis_v2/` are audit artifacts; `analysis_final/` is authoritative.

## 固定动作对照

下表为逐序列先平均、再对 32 序列宏平均的 decoded-IoU 差；区间为配对 sequence bootstrap 95%。动作臂的比较是同 crop/state 单步效用，不是跟踪 AUC。

| 动作 | Δ IoU | 95% CI | 序列胜/负/平 |
|---|---:|---:|---:|
| no-op | 0.0000 | 基准 | 0/0/32 |
| RGB 5% | -0.0011 | [-0.0020,-0.0002] | 11/21/0 |
| RGB 10% | +0.0017 | [-0.0031,+0.0098] | 12/20/0 |
| RGB 15% | +0.0006 | [-0.0046,+0.0089] | 11/21/0 |
| TIR 5% | +0.0006 | [-0.0000,+0.0012] | 21/11/0 |
| TIR 10% | +0.0029 | [-0.0001,+0.0076] | 20/12/0 |
| TIR 15% | +0.0029 | [-0.0006,+0.0077] | 20/12/0 |
| 随机方向 10% | -0.0004 | [-0.0022,+0.0013] | 15/17/0 |

RGB 5% 是唯一总体区间低于零的固定臂；TIR 10/15% 均值略正但区间跨零。固定 TIR 倾向与此前低 AUC 闭环筛查一致，但这里仍只是固定状态单步结果。

## 信号选择器结果

| 选择器 | 序列宏平均 Δ IoU [95% CI] | no-op specificity | harmful intervention | 覆盖率 | 平均 regret |
|---|---:|---:|---:|---:|---:|
| pre-fusion 信号 raw argmax | +0.0017 [-0.0028,+0.0093] | 0/57 (0%) | 11.5% | 100% | 0.0127 |
| pre-fusion + CENTER-risk raw | -0.0017 [-0.0035,+0.0002] | 0/57 (0%) | 11.5% | 100% | 0.0183 |
| pre-fusion 信号 LCB | +0.00002 [0.00000,+0.00005] | 56/57 (98.2%) | 0% | 1/104 (0.96%) | 0.0161 |
| pre-fusion + CENTER-risk LCB | -0.00002 [-0.00007,0.00000] | 57/57 (100%) | 0% | 1/104 (0.96%) | 0.0162 |

Risk-coverage 在目标 5%、10%、20% 时两种模型都只接受 1/104 状态，实际 coverage 均为 0.96%。pre-fusion LCB 选中的单状态 utility `+0.00175`，加入 CENTER-risk 后选中状态 utility `-0.00374`。LCB 所有 severe-onset 样本都选择 no-op，因此“安全”来自几乎不干预，而不是准确识别需求。

## 场景与机制诊断

按预注册标签的分层 CSV 在 `analysis_final/by_label.csv`。28 个 good-only 状态中 RGB 5/10/15% 均值为 `-0.0011/-0.0022/-0.0032`，TIR 5/10/15% 为 `+0.0009/+0.0018/+0.0025`；该方向分层是描述性的，不能据此直接固定部署 TIR 增益。

4 个 severe-onset 状态显示强异质性，不足以支持一条通用方向规则：

- `motofromdark` frame 52：RGB 10/15% 分别 `+0.6048/+0.6081`，raw pre-fusion 信号命中 RGB 15%。
- `guardman` frame 75：TIR 10/15% 为 `+0.3399/+0.3370`，raw 信号却选 RGB 5%，得到 `-0.0469`。
- 另两个 severe-onset 状态 oracle 最佳动作 utility 不超过 `0.005`，应选择 no-op。

这说明潜在可救援动作确实存在，但当前运行时信号未可靠判别“是否需要动作”以及“哪一路、哪个剂量”。severe 样本只有 4 个，不能把单个大 IoU 个例当成可推广收益或闭环恢复证据。

## 归因边界与下一步

**已排除/降级：** 当前线性信号选择器不是已验证的 no-op-aware controller；把 pre-fusion 信号与 CENTER risk 简单拼接没有改善，保守 LCB 的低伤害来自极低覆盖；当前小剂量对所有序列统一施加也无稳健总体收益。

**仍未区分：** 固定状态单步 utility 能否转为闭环恢复；其他非缩放型融合残差是否可被同类信号预测；信号特征交互、非线性模型和小样本 severe 状态能否泛化；当前 CENTER/crop 的高损失究竟来自定位读出还是反馈漂移。

本轮建议决策是**停止当前缩放动作和这批信号上的 gate 调参**。若继续，先在新的 sequence-held-out cohort 上做更高功效的需求标签审计，要求同时检验 no-op specificity 与 severe positive-action recall；不要在这 32 条上换回归器、改阈值后再报效能。之后是否进入 residual operator，应由新 cohort 的预注册准入结果决定。
