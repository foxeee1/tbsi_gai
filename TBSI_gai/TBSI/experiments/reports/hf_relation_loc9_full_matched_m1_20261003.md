# HF-Relation-MHSA location 9 full matched M1 analysis

## Result

`HF_RELATION_LOC9_seed42_retry1` completed the planned 15-epoch training and serial FP32 tracking on all 244 LasHeR test sequences. The candidate adds a zero-initialized, bounded frequency relation bias only to the cross-modal attention-logit blocks of the joint MHSA at TBSI location 9. Q/K/V, Value, template features, output message residuals, loss, optimizer schedule, and the TBSI-E topology remain unchanged.

| Metric | Candidate | TBSI-E M1 | Delta |
|---|---:|---:|---:|
| AUC | 56.47 | 56.21 | +0.26 |
| OP50 | 68.56 | 68.33 | +0.23 |
| OP75 | 47.49 | 47.13 | +0.36 |
| Precision | 70.51 | 70.21 | +0.30 |
| Norm Precision | 66.62 | 66.20 | +0.42 |

The headline metrics are positive on this single seed, but they do not establish a stable gain.

## Paired sequence analysis

The candidate and parent contain the same 244 sequence names and were evaluated with the same LasHeR metric implementation. The candidate prediction files were re-evaluated to create the durable candidate curve artifact before pairing.

- Mean sequence AUC delta: `+0.2634` percentage points.
- Median sequence AUC delta: `+0.0128` percentage points.
- 5% trimmed mean: `-0.1353` percentage points.
- 10% trimmed mean: `-0.0720` percentage points.
- 20% trimmed mean: `-0.0117` percentage points.
- Win/tie/loss: `122 / 1 / 121`; strict win rate `50.00%`.
- Bootstrap 95% interval for mean delta: `[-0.8991, +1.4704]` percentage points.
- AUC delta `<= -5`: `21` sequences; delta `>= +5`: `21` sequences.

The untrimmed mean is positive, but its interval crosses zero, the median and trimmed means are near zero or negative, and the tail-risk counts are symmetric and material. The result is therefore best classified as a provisional positive single-seed result, not a confirmed method gain.

Attribute summaries are descriptive only. For example, DEF is positive (`+3.22` AUC points, n=46), while HI is negative (`-3.61`, n=34); these are multi-label subgroups without preregistered subgroup intervals and must not be used as mechanism claims.

## Decision

Mark the full run as `VALID-FULL-MATCHED-PROVISIONAL`. Do not claim stable improvement and do not launch additional frequency variants from this result. The next method-level action is a decision about whether this small, unstable single-seed signal justifies the preregistered spatial/ordinary modulation control or a second seed. Until that decision, the evidence boundary is: the cross-modal MHSA-logit interface is trainable and produced a small positive aggregate result, but the per-sequence evidence is inconclusive.

## Provenance

- Candidate config: `experiments/tbsi_track/HF_RELATION_LOC9_seed42_retry1.yaml`
- Candidate checkpoint: `output/experiments/HF_RELATION_LOC9_seed42_retry1/checkpoints/TBSITrack_ep0015.pth.tar`
- Candidate checkpoint SHA256: `8936b1b69e27d56fac92bf7da829177aa22336d38163a055d95da3545f270a0b`
- Candidate curve artifact: `output/experiments/hf_relation_loc9_full_matched_m1/candidate_eval_data.pkl`
- Paired analysis: `output/experiments/hf_relation_loc9_full_matched_m1/hf_relation_vs_m1_analysis.json`
- Parent curve artifact: `output/experiments/cfvi_p1_full_matched_m1_20261001/M1_reference/analysis/eval_data.pkl`
- Full pipeline log: `output/experiments/HF_RELATION_LOC9_seed42_retry1/logs/full_pipeline.log`
- Tracking results: `output/test/tracking_results/tbsi_track/HF_RELATION_LOC9_seed42_retry1/`
- Hardware: 1 x RTX4090D 24GB; Python 3.8.20; PyTorch 1.9.0+cu111; CUDA 11.1; FP32 tracker inference
- Canonical baseline YAML was unchanged.
