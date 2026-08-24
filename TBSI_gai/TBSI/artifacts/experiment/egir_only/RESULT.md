# EGIR-only Minimal Validation

## Setup

- Baseline checkpoint: `output/experiments/v1.0.0-完美基线-全量/checkpoints/TBSITrack_ep0015.pth.tar`
- Training config: `experiments/tbsi_track/v7.0.0-egir-only-smoke.yaml`
- Frozen TBSI backbone and head; only the 14,944-parameter EGIR router was trainable.
- Training: 3 epochs, 12,000 samples per epoch, seed 42.
- Evaluation: full LasHeR test, 244/244 sequences.

## Results

| Method | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| Pure baseline | 55.46 | 67.28 | 46.73 | 68.98 | 65.23 |
| EGIR-only | 54.71 | 66.50 | 45.87 | 68.56 | 64.62 |
| Delta | -0.75 | -0.78 | -0.86 | -0.42 | -0.61 |

## Interpretation

The minimal EGIR router does not reach the baseline and should not be promoted as a complete method. The result is also consistent with a conservative-router failure mode: the final keep/exchange logits remain close to the identity initialization (approximately `+4/-4`), while the exchange branch is scaled by `0.10`. Thus the experiment validates implementation and end-to-end trainability, but not useful quality-aware fusion under the current supervision, scale, and three-epoch budget.

The current evidence is insufficient to justify a full EGIR training run. Any follow-up should first verify route activation and add an explicit quality/consistency objective; otherwise this branch should be stopped.
