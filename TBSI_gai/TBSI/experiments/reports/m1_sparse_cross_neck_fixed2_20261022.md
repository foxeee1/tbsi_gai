# M1 Sparse Cross Neck Fixed-State Smoke (2026-10-22)

## Scope

This is the first mechanism smoke for the sparse candidate-conditioned cross-modal route. It is a fixed-state diagnostic only; no training, optimizer, loss, dataset split, or protected M1 artifact was changed.

The implementation follows the actual M1 `tbsi_e` graph. M1's MSHA updates template tokens, while search RGB/TIR tokens are mixed at `tbsi_fuse_search`. Therefore the candidate mask is generated from block02 template/search tokens and applied at the search fusion neck. Candidate locations use the union of RGB and TIR top-8 template-foreground responses followed by a one-cell dilation. Candidate cells use the mixed fusion output; other cells use the mean of RGB-only and TIR-only fusion outputs. The route is controlled by `TBSI_SPARSE_CROSS_MODE=neck_union` and is off by default.

## Contract checks

- Python compilation passed for `vit_tbsi_care.py` and `tbsi_track.py`.
- Existing random-input M1 preflight passed with the sparse route enabled in the environment.
- Score map max difference between parent and zero-initialized utility candidate: `3.24e-5`; box max difference: `3.70e-6`.
- Output contracts remained `[1,1,16,16]` score map and `[1,1,4]` box.
- The protected M1 config and epoch-15 checkpoint were loaded without modification.

## Fixed-state smoke

The existing calibration event manifest was used only for a two-state smoke (`1handsth`, pre-onset frame 284 and severe-onset frame 285). Parent and candidate used the same template, crop and tracker state. The test was run in FP32 with one CPU thread.

| event | parent IoU | sparse IoU | delta | parent GT margin | sparse GT margin | rank delta |
|---|---:|---:|---:|---:|---:|---:|
| `1handsth:pre_onset:284` | 0.7741 | 0.7006 | -0.0736 | 0.4266 | 0.4365 | 0 |
| `1handsth:severe_onset:285` | 0.4987 | 0.5450 | +0.0463 | 0.3721 | 0.3780 | 0 |

The two states show a small positive CENTER margin change in both states, but IoU changes in opposite directions. The candidate has therefore reached the intended fusion/readout path, but this smoke does not establish target-selective improvement.

## Decision

`CALIBRATION-DIAGNOSTIC-COMPLETE / NOT-TRAINING-ADMITTED`. The route is technically connected and reversible. On all 46 calibration fixed states, the candidate fraction averaged `0.2001` (RGB `0.1428`, TIR `0.1386`). Relative to the matched parent, CENTER margin changed by mean `+0.0111` (29 up, 17 down), GT rank improved on 4 states and worsened on 3, while IoU changed by mean `+0.0326` but median `+0.00027` (23 wins, 20 losses). The effect is too small and inconsistent for quick training admission. The candidate mask is active, but this result does not show that it concentrates correction on the GT rather than suppressing or replacing background responses.

The next valid experiment is a frozen non-target spatial control with the same candidate density and a shifted mask. If the target-conditioned arm does not beat that control on GT rank/margin, stop this route. Do not tune `topk`, dilation or train the candidate before that control.

## Artifacts

- Parent: `output/experiments/M1_SPARSE_PARENT_FIXED2_20261022/`
- Candidate: `output/experiments/M1_SPARSE_NECK_UNION_FIXED2_20261022/`
- Calibration candidate: `output/experiments/M1_SPARSE_NECK_UNION_CALIB46_RETRY_20261022/`
- Calibration parent: `output/experiments/M1_SPARSE_PARENT_CALIB46_20261022/`
- Runner: `tools/run_m1_screening_fixed_state_flow_20261022.py`
- Source changes: `lib/models/tbsi_track/vit_tbsi_care.py`, `lib/models/tbsi_track/tbsi_track.py`
- Data: `experiments/protocols/m1_screening_v1_20261022/calibration_sequences.txt` and `output/experiments/M1_SCREENING_EVENT_MINING_CALIBRATION_20261022/first_persistent_events.jsonl`
