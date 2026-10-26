# E2 AMCM per-sequence diagnosis (2026-10-18)

## Verdict

The stitched AMCM-lite module does not solve the intended problem of modality-conflict correction in the current implementation. The seed42 full-test gain is not reproduced by seed3407: seed3407 is `-0.08` AUC below M1 and `-0.55` AUC below seed42 E2. The result is `SECOND-SEED-CONTRADICTORY / NOT-PROMOTED`.

## What the code actually does

In `tbsi_track.py`, the E2 configuration sets `POSITION: post_fusion`, but the module receives the separate RGB/TIR search streams before `tbsi_fuse_search`. It predicts two spatial deltas, `delta_rgb` and `delta_tir`, and then computes:

`amcm_post_gate = 0.5 * (delta_rgb + delta_tir)`

The base fused feature is then multiplied by `1 + amcm_post_gate`. Therefore:

1. The module does not change the relative RGB/TIR contribution after fusion.
2. Opposite modality gates cancel in the arithmetic mean.
3. The supervision target trains `delta_tir - delta_rgb` from frozen template-search similarity, but the post-fusion forward path applies the mean gate. The supervised quantity is not the quantity that controls the fused representation.
4. The gate is bounded by `MAX_DELTA=0.1` and only the GT cell receives explicit reliability supervision; the rest of the map is driven indirectly by tracking loss.

This is an interface/objective mismatch, rather than evidence that reliability estimation is impossible.

## Full-test metrics

| Method | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| M1 | 56.21 | 68.33 | 47.13 | 70.21 | 66.20 |
| E2 seed42 | 56.68 | 69.03 | 47.60 | 70.83 | 66.88 |
| E2 seed3407 | 56.13 | 68.26 | 47.15 | 70.08 | 66.24 |

## Per-sequence evidence

The complete paired table is `output/experiments/M1_AMCM_E2_SEED3407_10EP_FULL244_20261018/analysis/per_sequence_comparison.csv`.

For AUC relative to M1:

| Comparison | Gain | Loss | Tie | Mean delta |
|---|---:|---:|---:|---:|
| seed42 E2 - M1 | 125 | 117 | 2 | +0.47 |
| seed3407 E2 - M1 | 134 | 110 | 0 | -0.08 |

The sign agreement is weak at the sequence level: 90 sequences improve under both seeds, 74 degrade under both, 35 switch from seed42 gain to seed3407 loss, and 43 switch from seed42 loss to seed3407 gain. The Pearson correlation of the two per-sequence AUC deltas is `0.56`, which indicates some shared sequence difficulty but not stable module benefit.

Seed3407's largest AUC gains over M1 are:

- `3rdfatboy`: `+53.68` (seed42 `+54.11`)
- `leftfarboycomingpicktheball`: `+47.66` (seed42 `+47.10`)
- `leftuphand`: `+19.88` (seed42 `+16.32`)
- `whiteboyrightcoccergoal`: `+8.70` (seed42 `+0.39`)
- `girlrightthewautress`: `+7.47` (seed42 `+25.31`)

The shared gains on `3rdfatboy`, `leftfarboycomingpicktheball`, `leftuphand`, and `umbrellawillopen` are the strongest evidence that the module can help a small subset of scenes. They do not establish that it is correcting modality conflict, because no modality-specific gate trace was recorded for test sequences.

Seed3407's largest AUC losses are:

- `blackdown`: `-57.76` (seed42 `-0.08`)
- `blkboydown`: `-38.97` (seed42 `+0.17`)
- `carcomingfromlight`: `-14.81` (seed42 `-15.02`)
- `ab_whiteboywithbluebag`: `-14.39` (seed42 `-14.41`)
- `pinkwithblktopcup`: `-13.98` (seed42 `-11.26`)

The two extreme seed3407 failures (`blackdown`, `blkboydown`) dominate the negative stability signal. They are not explained by the seed42 run, so the current evidence is consistent with seed-sensitive gate behavior or an interaction with the closed-loop tracker state.

## What is supported and what is not

Supported: the adapter is trainable, its parameters move away from initialization, and it can produce very large positive or negative sequence changes. A small set of sequences benefits under both seeds.

Not supported: a robust E2 improvement, reliable modality selection, recovery from target loss, or a claim that the module fixes M1's crop-center feedback failure. The training logs contain only CSV headers for feature, gradient, signal, and loss diagnostics, so there is no recorded test-time gate magnitude or correlation with failures.

## Next discriminating experiment

Do not add another reliability module yet. First make one minimal contract correction: preserve separate modality weights through fusion, e.g. apply `delta_rgb` and `delta_tir` to the two inputs to `tbsi_fuse_search`, or add a learned post-fusion residual that uses the signed difference `delta_tir - delta_rgb` rather than the mean. Run zero-gate parity, one-batch gradient, fixed-state gate trace, and a small paired sequence set before another full test. This directly tests whether the present negative result comes from the injection/interface mismatch.
