# M1 CENTER ranking loss proxy20 full-244 follow-up (2026-10-17)

## Contract

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15 checkpoint.
- Candidate: `M1_CENTER_RANK_LOSS_PROXY20_20261016`, epoch-3 proxy checkpoint; only the gated CENTER pairwise ranking loss was added during proxy training.
- Test: official LasHeR test split, 244/244 sequences, FP32 serial tracker, read-only official metric audit.
- This is a proxy-trained checkpoint evaluated on full test; it is not a full-training method result.

## Results

| Metric | M1 | ranking-loss proxy20 | Delta |
|---|---:|---:|---:|
| AUC / Success | 56.2077 | 54.8910 | -1.3167 |
| OP50 | 68.3339 | 66.7206 | -1.6133 |
| OP75 | 47.1349 | 45.9218 | -1.2132 |
| sequence-equal mean valid IoU | 0.5713 | 0.5576 | -0.0136 |

The candidate had no missing prediction files. Persistent IoU<0.5 segments increased from 430 to 474, and the number of sequences containing such segments increased from 136 to 145. The longest segment was shorter for the candidate (1133 vs 1494 frames), so the change is not a general recovery effect.

## Decision

`NO-GO / PROXY-FULL244-NEGATIVE`. The proxy20 center ranking loss did not preserve M1 on the official split and is not promoted to full training or further tuning. This supports the diagnosis that adding a training-time score ordering term alone does not solve the crop-internal center-selection and closed-loop recovery problem.

The full-training rank-loss run remains separately `INCONCLUSIVE / USER-REQUESTED-INTERRUPTION`; this proxy checkpoint result must not be used to label that interrupted run negative.

## Evidence

- Predictions: `output/test/tracking_results/tbsi_track/M1_CENTER_RANK_LOSS_PROXY20_20261016_20261017/`
- Audit: `output/experiments/M1_CENTER_RANK_LOSS_PROXY20_FULL244_20261017/official_audit/official_metric_and_persistence.json`
- Per sequence: `output/experiments/M1_CENTER_RANK_LOSS_PROXY20_FULL244_20261017/official_audit/per_sequence_official.csv`
- Config/checkpoint: `experiments/tbsi_track/M1_CENTER_RANK_LOSS_PROXY20_20261016.yaml`, `output/experiments/M1_CENTER_RANK_LOSS_PROXY20_20261016/checkpoints/TBSITrack_ep0003.pth.tar`

