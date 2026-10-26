# M1 Native MSHA Directional Utility Analysis (2026-10-23)

## Question

Do the four fixed native MSHA direction-amplitude interventions provide a repeatable localization benefit on the existing M1 calibration states, and how large is the hindsight action-selection ceiling?

## Protocol and provenance

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15; the protected YAML and checkpoint hashes were rechecked and match the registered values.
- Input runs: `M1_T2S_CONDITIONED_PARENT_CALIB46_20261023` and `M1_NATIVE_MSHA_FIXED_{R2T075,R2T125,T2R075,T2R125}_CALIB46_20261023`.
- Cohort: 23 LasHeR training-set calibration events, each with pre-onset and severe-onset fixed states (46 paired states total). All arms use the same crop, frame, template, and state for each paired event. This is fixed-state replay, not closed-loop tracking.
- Intervention: scale one native cross-value direction by 0.75 or 1.25; the other direction and all other inputs remain as in the parent.
- Measures: fixed replay IoU, CENTER GT-versus-hard-background margin, GT CENTER rank, and distance from CENTER top-1 to GT center. CENTER rank delta is signed so positive means a better rank; top-1 distance decreases are favorable.
- Analysis command: `python tools/analyze_m1_native_msha_direction_controls_20261023.py`. The analysis is Python standard-library offline processing of the five captured `per_event.csv` files; it performs no inference or training and consumes no GPU time.
- Code anchor: repository commit `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`; analysis script SHA256 `58bcd183f24de92e5e9d8e0ef44318c37df7309e3b2cb7837bfd232f96202e59`. The experiment inputs were captured under the working-tree source recorded by their run manifests; this analysis does not claim those manifests are a clean-commit reproduction.
- Environment of captured runs: 1x NVIDIA GeForce RTX 4090D; Python 3.8.20, PyTorch 1.9.0+cu111, CUDA 11.1, FP32, torch threads 1. Offline aggregation used Python 3.8.20 on CPU.
- Locked holdout was not opened. No full tracking metric is applicable.

## Paired results

| Direction and scale | Mean IoU delta | Median IoU delta | CENTER rank improved | Mean margin delta |
|---|---:|---:|---:|---:|
| RGB -> TIR, 0.75 | -0.01025 | +0.00002 | 0/46 | -0.000276 |
| RGB -> TIR, 1.25 | -0.00870 | 0.00000 | 1/46 | +0.000139 |
| TIR -> RGB, 0.75 | -0.00030 | 0.00000 | 1/46 | +0.000663 |
| TIR -> RGB, 1.25 | +0.00030 | 0.00000 | 0/46 | -0.000846 |

For all four arms, the CENTER top-1 location was unchanged in at least 44/46 states. The small mean margin changes do not consistently track IoU or CENTER rank. Thus the amplitude control can change the decoded box, but it does not reliably improve the center decision on this cohort.

The arm means are sensitive to individual states. On `bike150:severe_onset:144`, TIR -> RGB at 0.75 changes IoU by +0.47267 and accounts for most of the hindsight-oracle sum. Excluding both `bike150` states, the four arm mean IoU deltas are -0.01071, -0.00910, -0.01108, and +0.00030 respectively. On `theleftestrunningboy:pre_onset:17`, the first three interventions each reduce IoU by about 0.47, while TIR -> RGB at 1.25 changes it by +0.00075. These states dominate the apparent tails; medians and center-rank changes better reflect how rarely the actual center decision changed.

## Hindsight oracle ceiling

Selecting the best of the four interventions separately for every state using that same state's GT replay IoU gives mean delta +0.01368, median +0.00139; 42 states are positive and 4 are tied. The largest state supplies 75.1% of the summed gain; excluding it reduces the mean to +0.00348. The oracle improves CENTER GT rank in only 2/46 states. A per-sequence retrospective fixed action gives positive IoU on 22/23 sequences, but the same action is best in both phases for only 7/23 sequences.

This is an optimistic, non-deployable ceiling: it uses the ground-truth replay outcome to choose the intervention. It shows heterogeneous controllability, not that an online policy can identify the useful direction. Its apparent gain is concentrated in outliers and rarely changes CENTER's selected location.

## Decision

**暂不支持** fixed direction and amplitude as a general CENTER localization improvement. **尚未区分** whether pre-intervention, observable features can predict which direction will help, especially around onset. The result does not reject MSHA as an insertion point and does not identify MSHA as the cause of M1's system-level failure chain.

The next permitted experiment is a frozen, offline sequence-held-out predictability analysis: use only pre-action measurements available at inference time to predict per-event direction utility, compare against a fixed-action baseline, and report sequence-level uncertainty and outlier sensitivity. Ground-truth IoU may define the utility label but must not be used as a predictor. Do not train a router or run full tracking until this predictor shows held-out discrimination beyond the fixed-action baseline. Keep the CENTER/Hann peak-selection mechanism as a separate branch; do not conflate the two hypotheses.

## Artifacts

- `output/experiments/M1_NATIVE_MSHA_DIRECTIONAL_ORACLE_ANALYSIS_20261023/per_event_paired_oracle.csv`
- `output/experiments/M1_NATIVE_MSHA_DIRECTIONAL_ORACLE_ANALYSIS_20261023/summary.json`
- `output/experiments/M1_NATIVE_MSHA_DIRECTIONAL_ORACLE_ANALYSIS_20261023/analysis_stdout.json`
- `output/experiments/M1_T2S_CONDITIONED_PARENT_CALIB46_20261023/`
- `output/experiments/M1_NATIVE_MSHA_FIXED_R2T075_CALIB46_20261023/`
- `output/experiments/M1_NATIVE_MSHA_FIXED_R2T125_CALIB46_20261023/`
- `output/experiments/M1_NATIVE_MSHA_FIXED_T2R075_CALIB46_20261023/`
- `output/experiments/M1_NATIVE_MSHA_FIXED_T2R125_CALIB46_20261023/`
