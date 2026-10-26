# Template Expert P1 RPP-v1 Serial Test

Date: 2026-09-28

## Contract

- Configs: `TEMPLATE_EXPERT_C_P1_seed42`, `TEMPLATE_EXPERT_G_P1_seed42`, `TEMPLATE_EXPERT_T_P1_seed42`
- Dataset: `rpp_lasher_test`, frozen 60-sequence RPP-v1 test manifest
- Manifest SHA256: `2ce184bc7e240fef68c9b72124a89be808e0428622fab8dbd9c7156eb65f825b`
- Evaluation: serial, `threads=0`, one GPU, `TBSI_INFER_FP16=0`
- Checkpoints: each experiment's `TBSITrack_ep0003.pth.tar`
- Environment: Python 3.8.20, PyTorch 1.9.0+cu111, RTX4090D 24 GiB

## Results

| Config | AUC | OP50 | OP75 | Precision | Norm Precision | Mean sequence FPS | Result files |
|---|---:|---:|---:|---:|---:|---:|---:|
| C control | 52.02 | 63.60 | 40.72 | 66.82 | 61.32 | 72.45 | 60 |
| G global template expert | 52.01 | 63.11 | 41.87 | 66.10 | 61.11 | 69.95 | 60 |
| T target-weighted template expert | 50.96 | 61.71 | 40.99 | 64.52 | 59.71 | 69.55 | 60 |

Relative to C on this proxy split:

- G: AUC -0.01, OP50 -0.49, OP75 +1.15, Precision -0.72, Norm Precision -0.21.
- T: AUC -1.06, OP50 -1.89, OP75 +0.27, Precision -2.30, Norm Precision -1.61.

## Failure and repair

The first T run completed the 60-sequence loop but produced no prediction files. The analysis failed on missing `bike2left.txt`. The cause was an inference interface mismatch: the target expert requires an RGB/TIR mask tuple, while the tracker passed one `[1,64]` mask. The inference call was corrected to pass `(self.template_fg_mask, self.template_fg_mask)`; only T was rerun under the same protocol. The repaired T run completed 60/60 and analysis completed successfully.

Evidence:

- `output/experiments/template_expert_p1_20260928/logs/TEMPLATE_EXPERT_C_P1_seed42_rpp_v1_{test,analysis}.log`
- `output/experiments/template_expert_p1_20260928/logs/TEMPLATE_EXPERT_G_P1_seed42_rpp_v1_{test,analysis}.log`
- `output/experiments/template_expert_p1_20260928/logs/TEMPLATE_EXPERT_T_P1_seed42_rpp_v1_test_retry.log`
- `output/experiments/template_expert_p1_20260928/logs/TEMPLATE_EXPERT_T_P1_seed42_rpp_v1_analysis_retry.log`
- `output/test/tracking_results/tbsi_track/TEMPLATE_EXPERT_{C,G,T}_P1_seed42/`

## Interpretation

These are single-seed proxy-60 screening results, not full LasHeR-244 benchmark results and not paper claims. On this fixed proxy split, G is effectively tied with C in AUC while costing about 3.5% mean sequence FPS; T is lower than C on AUC and precision. Neither candidate should be promoted to full confirmation from this single proxy run without the required multi-seed/RPP decision gate.
