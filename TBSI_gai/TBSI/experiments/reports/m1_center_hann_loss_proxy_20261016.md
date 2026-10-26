# M1 center Hann-aware loss proxy

## Question and intervention

The tracker trains focal loss on `score_map`, but inference selects the
center from `Hann * score_map`. This run tests the single train-side change of
applying the same Hann window before the focal loss. Fusion, offset/size
regression, state submission, and inference code are unchanged.

## Contract

- Parent: protected M1 epoch-15 checkpoint `TBSI_E_MHSA_FULL_4090D_seed42`.
- Config: `M1_CENTER_HANN_LOSS_PROXY_20261016.yaml`.
- Training: 3 epochs, 256 samples/epoch, batch 2, FP32, one worker, initialized
  from the protected M1 checkpoint.
- Evaluation: the fixed difficult5 manifest (`manoncall`, `boyss`,
  `girldownstairfromlight`, `bike2left`, `girlinrain`), serial FP32 tracking.
- This is a proxy screen and is not a full-244 result.

## Aggregate result

| Method | AUC / Success | Precision | Norm Precision | Mean IoU |
|---|---:|---:|---:|---:|
| M1 frozen checkpoint | 66.7115 | 89.3621 | 78.0039 | 0.67690 |
| Hann-aware focal loss, epoch 3 | 64.60 | 84.11 | 74.79 | approximately 0.646 |

Relative to M1, the candidate loses `-2.11` AUC, `-5.25` Precision and
`-3.21` Norm Precision on this proxy.

## Sequence-level result

| Sequence | M1 AUC | Candidate AUC | Delta | Mean-IoU delta |
|---|---:|---:|---:|---:|
| manoncall | 70.06 | 62.90 | -7.16 | -0.0749 |
| boyss | 54.56 | 51.04 | -3.52 | -0.0354 |
| girldownstairfromlight | 47.10 | 48.67 | +1.57 | +0.0154 |
| bike2left | 81.36 | 80.72 | -0.64 | -0.0054 |
| girlinrain | 86.13 | 85.62 | -0.51 | -0.0048 |

## Decision

`DIAGNOSTIC-COMPLETE / HANN-CONTRACT-NOT-SUPPORTED`. Aligning the focal loss
with the inference Hann response did not repair the difficult sequences and
caused broad proxy regression. The center-selection mismatch remains a valid
diagnosis, but this direct loss alignment is not an effective fix under the
current training schedule. Do not run full training or full-244 evaluation.

The next useful direction is a target-aware candidate ranking loss that keeps
the original score-map loss and supervises relative utility among nearby
cells. It must be tested with a matched no-change control before any larger
run.

## Provenance

- Config: `TBSI_gai/TBSI/experiments/tbsi_track/M1_CENTER_HANN_LOSS_PROXY_20261016.yaml`
- Code change: `TBSI_gai/TBSI/lib/train/actors/tbsi_track.py`, gated by
  `TRAIN.HANN_AWARE_CENTER_LOSS` (default false)
- Training output: `TBSI_gai/TBSI/output/experiments/M1_CENTER_HANN_LOSS_PROXY_20261016/`
- Tracking output: `TBSI_gai/TBSI/output/test/tracking_results/tbsi_track/M1_CENTER_HANN_LOSS_PROXY_20261016/`
- Git commit at run start: `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`
