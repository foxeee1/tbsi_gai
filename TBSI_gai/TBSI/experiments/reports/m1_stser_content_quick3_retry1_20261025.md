# M1 STSER Content Quick3 Retry1

## Contract

- Parent: shared SOT initialized state in `experiments/snapshots/m1_stser_content_pair_quick3_20261025/`.
- Dataset: LasHeR RPP train manifest (176 sequences), RPP-v2 test manifest (60 sequences, 97,908 frames).
- Seed 42; 3 epochs; batch 16; accumulation 8; fresh AdamW/StepLR; AMP training; FP32 tracker; `threads=0`; template update off.
- Control and candidate differ only in `SEARCH_CONDITIONED_T2S_CONTENT_STRENGTH` (0 versus 1).
- Retry1 configs add the missing sixth `DATA.STD` entry required by the six-channel RGB-T input.

## Data contract incident

The original quick3 configs had five standard-deviation values while the loader produces six channels. Every processing attempt raised a channel mismatch and the sampler swallowed the exception, causing an apparent pre-first-batch hang. The original interrupted runs remain evidence of this configuration failure and have no checkpoint or metric claim. Retry1 passed a real first-batch check and completed both trainings.

## Training evidence

- Control checkpoint: `output/experiments/M1_STSER_CONTENT_QUICK3_CONTROL_20261025_RETRY1/checkpoints/TBSITrack_ep0003.pth.tar`, SHA256 `4731342a179cdc68e117d74e50cc84c76f0eb6bcd0b364e85104a5c5dbc79356`.
- Candidate checkpoint: `output/experiments/M1_STSER_CONTENT_QUICK3_CANDIDATE_20261025_RETRY1/checkpoints/TBSITrack_ep0003.pth.tar`, SHA256 `9a8ea1deac914f67dd9a786f157e2fe81761d4880ab904c5f28c158767e1efda`.
- Candidate content diagnostics had nonzero source shift/message norms and `content_lambda_has_grad=1` at every recorded probe; this supports learnability, not tracking benefit.

## Test results

| Metric | Control | Candidate | Candidate - Control |
|---|---:|---:|---:|
| AUC | 51.748 | 51.671 | -0.077 |
| OP50 | 61.882 | 61.802 | -0.080 |
| OP75 | 43.447 | 43.660 | +0.213 |
| Precision | 63.090 | 62.962 | -0.128 |
| Norm Precision | 60.171 | 60.142 | -0.029 |
| FPS | 56.496 | 47.982 | -8.514 |

The sequence bootstrap CI for ΔAUC is `[-1.068, +0.929]`; for ΔOP75 it is `[-0.633, +1.095]`. The paired result is therefore **inconclusive for benefit and does not support a performance claim**. The content branch is learnable, but its current message redistribution does not transmit a reliable aggregate tracking gain and adds about 15% inference cost.

Evidence: `output/experiments/M1_STSER_CONTENT_QUICK3_CANDIDATE_20261025_RETRY1/paired_rpp60_summary.json`, `paired_rpp60_per_sequence.csv`, both retry1 run directories, and the two 60-sequence result directories.
