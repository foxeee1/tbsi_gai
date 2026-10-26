# M1 joint local signal decision (2026-10-13)

## Question

Can deployment-visible S1/S2 signals predict a useful local RGB/TIR post-fusion action, with no-op available, on sequence-held-out M1 fixed states?

## Contract

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, explicit epoch-15 checkpoint.
- Source: 72 train-side sequences and 144 fixed states; 48 calibration sequences / 24 held-out sequences, 100 / 44 states.
- Same saved crop, tracker state, template, feature tensors, CENTER head and decode for every arm.
- Local action: 3x3 support around the S1 signal peak, residual magnitude `rho=0.05`; actions are no-op, local RGB, local TIR, and utility-predicted three-way choice.
- Models: ridge predictors using S1+S2 features, or S1+S2+S3 features. This is fixed-state decoded-IoU utility, not closed-loop tracking AUC.

## Held-out result

| Model | States | No-op / RGB / TIR | Mean utility | Sequence-macro utility | Fixed RGB | Fixed TIR | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1+S2 | 44 | 26 / 13 / 5 | +0.000013 | +0.000121 | +0.000293 | -0.000761 | +0.001083 |
| S1+S2+S3 | 44 | 25 / 10 / 9 | -0.000095 | +0.000018 | +0.000293 | -0.000761 | +0.001083 |

## Decision

**STOP / SIGNAL-NOT-ADMITTED.** The held-out utility predictor is near zero and recovers only a small fraction of oracle utility. Adding S3 lowers state-level utility. This does not justify training an adaptive post-fusion residual or running the planned 82-sequence closed-loop comparison. Do not tune on the already observed 82-sequence cohort.

The valid follow-up is a new outcome-blind train-side cohort only if it is independent of the reused diagnostics; otherwise pause this signal-to-action route and return to the frame-level conflict audit.

## Evidence and limits

For the full estimand, equations, nested comparison, signal-to-utility noise analysis, and attribution boundaries, see `experiments/reports/m1_joint_local_signal_detailed_analysis_20261013.md`.

- Summary: `output/experiments/M1_JOINT_LOCAL_SIGNAL_DECISION_20261013/model_artifacts_retry2/summary.json`.
- Per-state utility: `output/experiments/M1_JOINT_LOCAL_SIGNAL_DECISION_20261013/model_artifacts_retry2/local_action_utility.csv`.
- Utility models: `output/experiments/M1_JOINT_LOCAL_SIGNAL_DECISION_20261013/model_artifacts_retry2/local_utility_models.json`.
- The 72-state source cohort was previously used in signal diagnostics; M1 pretraining exposure is unknown. This is diagnostic fixed-state evidence, not a generalization or closed-loop AUC claim.
- Protected M1 YAML and checkpoint were read-only; checkpoint SHA256 remains `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
