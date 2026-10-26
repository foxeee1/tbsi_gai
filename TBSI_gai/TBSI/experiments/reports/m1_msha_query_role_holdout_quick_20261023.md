# M1 MSHA Query Role Holdout Quick Screen

## Contract

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15.
- Cohort: locked LasHeR trainingset holdout for candidate training, 24 sequences, 0 overlap with the candidate training manifest; first 128 frames per sequence (or sequence end). The protected parent was trained on the full 977-sequence LasHeR train population, so this cohort is not unseen by M1.
- Arms: M1, `foreground_queries`, and matched-mass `shifted_queries`.
- Precision and scheduling: FP32, one Torch thread, serial closed-loop tracking, seed 42.
- No training, optimizer update, checkpoint write, or M1 artifact modification.

## Result

Sequence macro mean IoU:

| Arm | Macro mean IoU | Paired result |
|---|---:|---|
| M1 | 0.807367 | reference |
| foreground query | 0.806284 | vs M1: -0.001083; 8 wins / 16 losses |
| shifted query | 0.806877 | foreground vs shifted: -0.000593; 15 wins / 9 losses |

## Decision

This candidate-training holdout screen does not support a closed-loop utility gain for foreground-query restriction. The structural effect seen in fixed-state replay does not transfer to a positive tracking result in this quick screen. Because M1 was trained on all LasHeR train sequences, this is not an unseen-parent generalization test. Do not run full-sequence evaluation or short training for this operator; retain the mechanism evidence as an inconclusive P3 asset and switch to another route.

## Provenance

- Runner: `tools/run_m1_msha_query_role_holdout_quick_20261023.py`, SHA256 `2e16f4d2bc1883df04e8ed40cc4368dcec142cdde41e6ead94115735a59a6837`.
- Sequence list SHA256 `a80f4b5f7779ee0a2dcbb2469708e74569d2732c68b21f8a021a2778af7b1267`.
- Outputs: `output/experiments/M1_MSHA_QUERY_ROLE_HOLDOUT_QUICK_20261023/{per_frame.csv,per_sequence.csv}`.
- Output hashes: `per_frame.csv` `65d107ebb3a5323793e7db94b7a514a063350a3191a12f9fdf1f249c135a4296`; `per_sequence.csv` `17a9a1dbbe5347ec09457d90329a580351949349f90bf7479b235050533ac7dc`.
