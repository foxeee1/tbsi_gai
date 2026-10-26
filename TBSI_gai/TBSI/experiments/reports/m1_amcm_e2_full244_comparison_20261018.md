# M1 AMCM E2 full-244 comparison (2026-10-18)

## Protocol

The E2 supervised post-fusion AMCM model was evaluated on the complete LasHeR `testingset` (244/244). The model used the independent 10-epoch checkpoint trained from protected M1 epoch15; only `amcm_reliability.*` was trained. Evaluation was FP32 (`TBSI_INFER_FP16=0`), with the same tracker search/template parameters and serial-equivalent per-sequence state handling as the M1 reference. Two workers were used only to schedule independent sequences; no sequence was shared between workers.

## Results

| Method | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| M1 protected reference | 56.21 | 68.33 | 47.13 | 70.21 | 66.20 |
| E2 AMCM 10 epoch | 56.68 | 69.03 | 47.60 | 70.83 | 66.88 |
| E2 - M1 | +0.47 | +0.70 | +0.47 | +0.62 | +0.68 |

All E2 metrics were computed over 244/244 sequences. The aggregate result is a small positive relative to the fixed M1 reference, but it is a single seed and one checkpoint. It should be treated as a preliminary full-test observation, not a robust multi-seed claim.

## Decision

`FULL244-COMPLETE / PRELIMINARY-POSITIVE / NOT-YET-PROMOTED`. The 10-epoch supervised post-fusion route passes the full-test screening comparison, unlike the earlier 3-epoch difficult5 and 5-epoch first30 screens. Before treating this as a paper-level improvement, repeat the matched full-244 comparison with at least one additional seed or an explicitly preregistered confirmation run, and inspect paired per-sequence gains/退化.

## Evidence

- Config: `experiments/tbsi_track/M1_AMCM_E2_10EP_FULL244_20261017.yaml`
- Checkpoint: `output/experiments/M1_AMCM_E2_10EP_50SEQ_20261017/checkpoints/TBSITrack_ep0010.pth.tar`
- Tracking results: `output/test/tracking_results/tbsi_track/M1_AMCM_E2_10EP_FULL244_20261017/` (244/244)
- Checkpoint SHA256: `b1317972f99a1ec2c7c44429f9e5297fdcf853fa0304d4cb6f0dc11a05e9bc41`
- M1 reference: `experiments/reports/tbsi_e_mhsa_full_20260927.md`
