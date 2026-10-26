# M1 Native MSHA Mechanism Analysis (2026-10-23)

本分析承接固定方向缩放实验，回答一个更具体的问题：干预后的 IoU 变化来自 CENTER 峰选择，还是来自 size/offset 回归变化。

输入是五个已完成的 46 状态 `per_event.csv`，沿用相同 crop、template、frame 和 tracker state。分析只做离线 CSV 运算，没有新增推理、训练或 holdout 评测。捕获运行的 manifest 环境为 1x RTX 4090D、PyTorch 2.0.0+cu118、CUDA 11.8、FP32、torch threads 1。

对每个干预计算：fixed replay IoU；raw CENTER margin 和 CENTER*Hann margin；CENTER GT rank；CENTER top-1 到 GT cell 的距离；post-fusion query 的 GT peak、GT-hard-background margin 和 rank；以及最终 predicted bbox 的中心移动和尺寸移动。两个极端状态和它们所在序列的另一个 phase 单独汇总。

## 结果

| cohort | arm | IoU delta | raw margin delta | Hann margin delta | CENTER rank changes | top-1 changes | bbox center shift | bbox size shift |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all 46 | RGB->TIR 0.75 | -0.01025 | -0.000337 | -0.000276 | 1 | 1 | 0.59 | 0.28 |
| all 46 | RGB->TIR 1.25 | -0.00870 | -0.000224 | +0.000139 | 3 | 2 | 0.81 | 1.02 |
| all 46 | TIR->RGB 0.75 | -0.00030 | +0.001306 | +0.000663 | 2 | 2 | 1.35 | 0.38 |
| all 46 | TIR->RGB 1.25 | +0.00030 | -0.001857 | -0.000846 | 0 | 0 | 0.09 | 0.13 |
| `bike150` 2 states | TIR->RGB 0.75 | +0.23684 | +0.001411 | +0.001401 | 1 | 1 | 18.11 | 2.58 |
| `theleftestrunningboy` 2 states | RGB->TIR 0.75 | -0.23405 | +0.001035 | +0.000747 | 1 | 1 | 10.74 | 3.34 |

### `bike150`

The mean gain is almost entirely from `bike150:severe_onset:144`; its IoU delta is `+0.47267`, while the paired pre-onset state changes by only about `+0.00101`. TIR->RGB 0.75 moves the CENTER top-1, improves the signed CENTER rank, and moves the decoded box center by about 36 pixels on that state. The post-fusion query GT peak and GT-hard-background margin increase only by about `0.001`, so the large IoU gain is a discrete readout transition rather than a broad score-map improvement.

### `theleftestrunningboy`

The adverse effect is similarly concentrated in `theleftestrunningboy:pre_onset:17`; RGB->TIR 0.75, RGB->TIR 1.25, and TIR->RGB 0.75 all move the CENTER top-1 away from the parent and produce about `-0.47` IoU on that state. The other phase is near zero. Across the two states, the direction controls move the decoded center by about 10.7--10.8 pixels for the first three arms, while TIR->RGB 1.25 keeps the parent-like center and IoU.

### Ordinary states

Across all 46 states, CENTER top-1 changes in only 1--2 states for each arm, and the average CENTER margin changes are on the order of `1e-4` to `1e-3`. This means most direction scaling does not repair the CENTER decision. The nonzero IoU response is usually a continuous change in predicted box center or size. For TIR->RGB 1.25, the decoded box is especially stable and its mean IoU is near zero, which explains why a small positive aggregate does not imply localization improvement.

## Interpretation

The fixed direction control has two regimes. Near a decision boundary, a small feature change can switch the CENTER peak and cause a large box displacement. Away from that boundary, the CENTER choice is stable and the same intervention only perturbs regression outputs. The large oracle gain therefore measures sensitivity of the discrete readout, not a stable target-versus-background improvement.

This supports continuing MSHA as a causal insertion point, while changing the next hypothesis. A single direction-level scalar is too coarse to make CENTER selection reliable. The next MSHA experiment should preserve the native message path and add a traceable candidate relation that can be evaluated at the search-position or query level, then measure peak-switch rate and regression displacement separately. A learned router should wait until those pre-action features predict which states are near a peak boundary under sequence-held-out analysis.

The evidence is diagnostic for this M1 implementation and fixed-state cohort. It does not establish that MSHA is the sole cause of the tracking failure, and it does not justify a full-training claim.

## Artifacts

- `output/experiments/M1_NATIVE_MSHA_MECHANISM_ANALYSIS_20261023/per_event_mechanism.jsonl`
- `output/experiments/M1_NATIVE_MSHA_MECHANISM_ANALYSIS_20261023/summary.json`
- `tools/analyze_m1_native_msha_mechanism_20261023.py`
- Input fixed-state arms under `output/experiments/M1_T2S_CONDITIONED_PARENT_CALIB46_20261023/` and `output/experiments/M1_NATIVE_MSHA_FIXED_*_CALIB46_20261023/`
