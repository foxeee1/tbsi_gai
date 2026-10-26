# AMCM token-match fixed-state diagnosis (2026-10-19)

## Identity and protocol

- Parent: protected M1 `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15.
- Candidate: `M1_AMCM_TMATCH_TRACKING_QUICK15_3EP_20261019`, trained from M1 for 3 epochs, 512 samples/epoch, seed 42, FP32, batch 2, DataLoader worker 4. Only `amcm_reliability.*` was trainable.
- Change under test: directed-residual AMCM uses search-to-all-template-token soft matching (temperature 0.07) instead of search-to-template-global-mean cosine. No loss, injection, head, or tracker change.
- Config SHA256: `a8b9a86df2d1e3441770e8bd0de654b8c08b74ec71e023131e2f416dda0bbc9e`.
- AMCM source SHA256: `b164e791e05e2a27d1e7d0a6281c3f98d06343ab400044e04313e136983b0a11`.
- Pre-change source snapshot SHA256: `8c5b94305e6c6761db12e23f26a9e15a61b73d24e4b10bde43142ff4f73d2e40`.
- Epoch 3 checkpoint SHA256: `b95a58cb9b39d25d67348b268ad869aaef414b947f4a071a94cd9bdbbbed8c8d`.
- Diagnostic: same 10 M1 low-AUC test events, identical crop, template, and `state_before`; fixed-state decode, not closed-loop sequence performance. GPU: 1x NVIDIA GeForce RTX 4090 D. Evaluation software: Python 3.8 / PyTorch 2.0.0+cu118 / CUDA 11.8. GPU-hours and FPS: NA.

## Results

Mean fixed-state IoU changed `0.30137 -> 0.30346` (`+0.00209`); 7 events increased and 3 decreased. The mechanism did not produce the required peak-ordering evidence: GT rank improved on `0/10`, top-1 attribution flipped to or away from GT on `0/10`. Mean target-to-hard-background residual magnitude ratios were RGB `1.0215` and TIR `0.9910`; the residual was not selectively larger at target positions. Training logs show nonzero AMCM gradients and updates, but mean residual/message norms stayed small relative to input features (input norm about 200; message norm about 0.04; gate probabilities about 0.49/0.51). Zero initialization preserves the disabled identity path.

| Sequence | Frame | Δ fixed IoU | GT rank M1→candidate | Δ margin | Residual target/background RGB | Residual target/background TIR |
|---|---:|---:|---:|---:|---:|---:|
| `7rightorangegirl` | 343 | -0.0004 | 1→1 | +0.00493 | 0.877 | 0.902 |
| `ab_bolstershaking` | 2952 | +0.0039 | 1→1 | +0.00097 | 0.905 | 0.918 |
| `blkboydown` | 49 | -0.0008 | 6→6 | -0.00105 | 1.017 | 1.026 |
| `boyss` | 218 | +0.0007 | 5→5 | -0.00029 | 1.043 | 1.045 |
| `foamatgirl\`srighthand` | 67 | +0.0029 | 2→2 | -0.00144 | 1.136 | 0.992 |
| `girldownstairfromlight` | 708 | +0.0066 | 1→1 | +0.00652 | 1.018 | 0.963 |
| `leftopenexersicebook` | 120 | +0.0008 | 7→7 | -0.00049 | 1.146 | 1.047 |
| `rightbluewhite` | 236 | -0.0034 | 1→1 | +0.00204 | 0.923 | 0.915 |
| `rightdarksingleman` | 883 | +0.0064 | 1→1 | +0.00529 | 1.020 | 0.989 |
| `umbrella` | 197 | +0.0043 | 3→4 | -0.00145 | 1.132 | 1.113 |

## Attribution

Decision: `NOT-PROMOTED / PEAK-ORDERING-NOT-SUPPORTED`. The tracking loss can update AMCM parameters, but in this run the learned messages remain near identity and do not alter GT rank. The small mean IoU rise is not aligned consistently with margin or rank and is not evidence that token matching solved the target/background confusion. The result does not distinguish whether the template matching map itself is weak at these locations or whether the learned spatial path suppresses its useful contrast. The next allowed test is one AMCM-internal change: use the opposite modality's template-match map directly as the spatial weight on each routed residual, without changing message content or training objective.

Evidence: `../output/experiments/M1_AMCM_TMATCH_FIXED10_DIAG_20261019/{summary.json,paired_per_sequence.csv,per_event.csv,tensors/}` and `../output/experiments/M1_AMCM_TMATCH_TRACKING_QUICK15_3EP_20261019/logs/`.
