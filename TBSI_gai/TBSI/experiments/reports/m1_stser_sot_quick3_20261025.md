# STSER SOT Initialization Quick3 Paired Validation

Date: 2026-10-25  
Parent initialization: `TBSITrack_SOT_Pretrained.pth.tar`, SHA256 `1f6ebf5905ed0a404a11beac944ef5d544c9e81605120c2052a2518467d2651b`  
Code base: git HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`, with source/config hashes in `experiments/snapshots/m1_stser_sot_quick_20261025/provenance.txt`.

## Question and intervention

The SOT-initialized STSER plain control routes each search token among shared, private, and null experts using the query, its T2S message, and cross-modal agreement. The candidate adds one input to that router: `query - competitor`. The competitor is a top-4, same-modality search feature pool selected by cosine similarity outside the query's 3x3 neighborhood. T2S attention, experts, outer residual, CENTER head, loss, data, and training schedule are otherwise matched.

Hypothesis: explicit evidence about a competing search location helps the router avoid suppressing useful cross-modal information when visually similar alternatives exist. This screen tests whether that operator has a measurable effect from the proper SOT initialization. It does not establish that competition is the sole source of the observed M1 failures.

## Training and test contract

- Both arms loaded the same SOT checkpoint; neither had `INIT_CHECKPOINT`, loaded M1 epoch15, nor restored an optimizer/scheduler state. Logs say `Load pretrained model from: TBSITrack_SOT_Pretrained.pth.tar` and `No matching checkpoint file found`.
- Each arm used a fresh AdamW and StepLR, seed 42, 3 epochs, 10,112 samples/epoch, 632 batches/epoch, accumulation 8, and 237 optimizer updates. RPP-LasHeR training used the locked 176-sequence manifest (SHA256 `410f7197a6d6a9d6e1886c0511e93d5a4ee1b4729fe77541e9d911aeaa82d8c3`); AMP on, batch16, worker4, prefetch2, persistent workers off. Existing config policy freezes layer 0; all remaining trainable backbone, TBSI, head, and STSER parameters were jointly optimized.
- Test: locked RPP-v2 60-sequence manifest (SHA256 `ab587f801525c2e7e0fc7a8465fb90f3a2d99b67764eb2bc19cac2b422a938a8`), 97,908 frames, FP32, serial `threads=0`, template update off. Both arms produced 60/60 prediction files.
- Hardware/software: 1x RTX 4090 D; Python 3.8.20, PyTorch 1.9.0+cu111, CUDA 11.1. Training and testing ran serially.

## Paired result

| Metric | Plain STSER | Competition | Paired delta | 95% sequence bootstrap CI | Wins / losses / ties |
|---|---:|---:|---:|---:|---:|
| AUC / Success | 51.364 | 52.034 | +0.670 | [+0.020, +1.491] | 34 / 25 / 1 |
| Precision | 62.296 | 63.057 | +0.761 | [-0.046, +1.755] | 27 / 25 / 8 |
| Norm Precision | 59.592 | 60.335 | +0.743 | [-0.031, +1.729] | 33 / 23 / 4 |
| OP50 | 61.229 | 61.916 | +0.687 | [-0.135, +1.726] | 28 / 25 / 7 |
| OP75 | 43.166 | 43.625 | +0.459 | [-0.157, +1.267] | 27 / 27 / 6 |
| Mean IoU | 0.51869 | 0.52560 | +0.00691 | [+0.00032, +0.01527] | 33 / 27 / 0 |

The sequence-macro AUC signal clears zero narrowly, while OP75 does not. The mean AUC delta remains +0.373 after removing its best sequence and +0.231 after trimming the six largest gains and six largest losses. However, the top three AUC gains contribute about 79% of the total delta: `right2ndflagformath` +18.169, `pingpingpad3` +7.331, and `ab_whiteboywithbluebag` +6.233. Largest AUC declines are `leftmirrorside` -3.852, `boyaroundtrees` -3.688, and `rainycarcome_ab` -2.578. These names are sequence observations, not causal scene labels.

Measured aggregate timing from the per-frame timing files is 63.285 FPS for plain and 60.089 FPS for competition, a 5.05% slowdown. The frame-weighted AUC delta is +0.606, so the macro result is not solely a consequence of short sequences, although sequence heterogeneity remains substantial.

## Information-flow evidence

All sampled STSER gradient groups were nonzero in both arms at all nine diagnostic checkpoints: router 8/8 tensors, experts 18/18, T2S attention 8/8, outer parameters 3/3. This excludes a disconnected branch for this run.

Across the nine sampled batches, plain routed TIR messages to null/private/shared with mean probabilities 0.862/0.113/0.025. Competition changed these to 0.071/0.729/0.200. RGB null mass changed from 0.702 to 0.475. Thus the new contrast input changed router decisions, most strongly for TIR, from null suppression toward the private expert.

The downstream write remained small: mean relative search-feature delta was `5.6e-7` RGB and `1.2e-6` TIR for plain, versus `2.8e-7` RGB and `1.65e-6` TIR for competition. Message norms were similar between arms. The evidence therefore supports a learned routing change with a small feature write; it does not yet show how that change propagates through later blocks into CENTER scores or boxes. The large sequence gains may include closed-loop amplification and require fixed-state replay to attribute.

## Judgment and next step

Conclusion: `SUPPORTS NEXT MATCHED FULL-TRAIN VALIDATION`. Correcting the initialization confound yields a positive AUC proxy signal for the competition operator over an SOT-initialized plain STSER control. The one-seed, 3-epoch, 60-sequence result is provisional; the AUC interval only narrowly clears zero, OP75 remains uncertain, and runtime is 5.05% slower. It is not a full-data or multi-seed performance claim.

Next experiment should be a matched 15-epoch pair from the same SOT checkpoint: plain STSER and competition STSER, each with fresh optimizer/scheduler and the same full LasHeR training and test contracts. Preserve the present 60-sequence checkpoint results. Add fixed-state comparisons for the three largest gain sequences and representative losses, recording layer-9 features, downstream CENTER response and box changes. Do not compare this pair as a direct replacement for protected M1 or reuse the M1 epoch15 continuation result as a causal baseline.

## Evidence

- Configs: `experiments/tbsi_track/M1_STSER_SOT_QUICK3_PLAIN_20261025.yaml` and `experiments/tbsi_track/M1_STSER_SOT_QUICK3_COMPETITION_20261025.yaml`.
- Runner/analyzer: `tools/run_m1_stser_sot_quick3_20261025.sh`, `tools/analyze_m1_stser_sot_quick3_20261025.py`.
- Summary/per-sequence results: `output/experiments/M1_STSER_SOT_QUICK3_COMPETITION_20261025/paired_rpp60_summary.json`, `paired_rpp60_per_sequence.csv`.
- Training diagnostics: each run's `logs/stser_gradient_flow.csv` and `logs/stser_flow.csv`; full commands, software/hardware, manifests, config/checkpoint hashes and timestamps: `experiments/snapshots/m1_stser_sot_quick_20261025/provenance.txt`.
- Checkpoints: plain SHA256 `897a15ed3978186e9ea92a7a54dcfb25f332dd86ccb61944bc89a6b91e8e47f6`; competition SHA256 `d641c601742d32d74ca2c4c4563ef2a9588e52efd0179f5c57831ea3921e12a5`.
