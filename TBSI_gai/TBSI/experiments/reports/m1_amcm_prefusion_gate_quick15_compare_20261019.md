# M1 AMCM gate vs pre-fusion residual quick comparison (2026-10-19)

## Scope

Two independent 3-epoch AMCM-only runs were initialized from the protected M1 epoch-15 checkpoint. They use the same LasHeR training proxy (`512` samples/epoch, validation `128`, batch `2`, FP32, seed 42, `NUM_WORKER=4`) and the same 15-sequence RPP manifest. The comparison changes only the AMCM path:

- `M1_AMCM_GATE_QUICK15_3EP_20261019`: legacy modality scaling followed by post-fusion average gate.
- `M1_AMCM_PREFUSION_RESIDUAL_QUICK15_3EP_20261019`: zero-initialized directed residual messages; corrected RGB/TIR maps are concatenated and passed into `tbsi_fuse_search`.

This is a mechanism screen, not a full-data or robust performance claim.

## Aggregate result

| run | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| post-fusion average gate | 54.36 | 65.98 | 44.38 | 70.02 | 64.07 |
| pre-fusion directed residual | 51.16 | 61.75 | 42.36 | 65.77 | 60.34 |
| residual minus gate | -3.20 | -4.23 | -2.02 | -4.25 | -3.73 |

The residual run wins 5/15 sequences and loses 10/15. Mean per-sequence AO is `0.5521` for gate and `0.5196` for residual (`-0.0325`).

## Per-sequence AO

| group | sequence | gate | residual | delta |
|---|---|---:|---:|---:|
| low AUC | boyleftblkrunning2crowd | 0.5516 | 0.5454 | -0.0063 |
| low AUC | bluegirlbiketurn | 0.4953 | 0.2667 | -0.2286 |
| low AUC | bluebuscoming | 0.7452 | 0.7455 | +0.0002 |
| low AUC | leftexcersicebookyellow | 0.5382 | 0.4579 | -0.0803 |
| low AUC | ab_whiteboywithbluebag | 0.2839 | 0.2838 | -0.0001 |
| difficult | manoncall | 0.6710 | 0.7023 | +0.0313 |
| difficult | boyss | 0.5499 | 0.4680 | -0.0819 |
| difficult | girldownstairfromlight | 0.4550 | 0.4696 | +0.0146 |
| difficult | bike2left | 0.8128 | 0.8111 | -0.0018 |
| difficult | girlinrain | 0.8630 | 0.8632 | +0.0002 |
| normal | blktribikecome | 0.2427 | 0.2403 | -0.0024 |
| normal | large | 0.6361 | 0.6465 | +0.0104 |
| normal | whiteboyrightcoccergoal | 0.3988 | 0.3986 | -0.0002 |
| normal | leftboyoutofthetroop | 0.5840 | 0.4510 | -0.1330 |
| normal | 3rdgrouplastboy | 0.4541 | 0.4439 | -0.0102 |

Group mean AO deltas are `-0.0630` for low-AUC, `-0.0075` for difficult, and `-0.0271` for normal. The result is therefore driven by a few large regressions (`bluegirlbiketurn`, `leftboyoutofthetroop`) rather than a uniform failure.

## Information and gradient checks

The implementation path is active: in directed-residual mode `amcm_stats['injection_path']` is `prefusion_corrected_modalities`, and the corrected feature maps replace the original maps before the `tbsi_fuse_search` call. The residual log has `16/16` nonzero AMCM parameter gradients at sampled steps; the gate control has `8/8`.

The residual message is not dead, but remains small. Across logged steps, mean input norms are RGB/TIR `221.5/204.0`, mean message norms `0.653/0.353`, and mean residual norms `0.364/0.157`, roughly `0.16%/0.08%` of the input norms. The learned modality probability is only mildly asymmetric (`RGB 0.455`, `TIR 0.545`). The gate control has no directed message path and its legacy gate starts at zero.

The residual run has higher and less stable training loss (mean sampled total `0.750` vs `0.736`; final `0.830` vs `0.239`) and AMCM gradient norm (`0.0853` vs `0.0526`). Its AMCM supervision loss is about `0.005-0.011`, versus about `0.0005` for the gate control. This indicates that the residual branch is receiving optimization signal, but the probability-at-GT-cell distillation target is not well matched to the residual operation or the short AMCM-only training contract. The current test cannot distinguish a poor residual operator from a mismatched auxiliary target.

## Decision

The experiment supports the information-flow claim: corrected RGB/TIR features really reach `tbsi_fuse_search` and receive gradients. It does not support the performance claim: under this short contract, pre-fusion residual is worse than the post-fusion gate, especially on low-AUC sequences. Do not start a full-data residual run yet. The next minimal experiment should keep the injection path and change only the auxiliary target or disable it, with a zero-gate parity and fixed-state peak trace on `bluegirlbiketurn`, `leftboyoutofthetroop`, `boyss`, `manoncall`, and one stable sequence.

Evidence: configs and checkpoints under `output/experiments/M1_AMCM_GATE_QUICK15_3EP_20261019/` and `output/experiments/M1_AMCM_PREFUSION_RESIDUAL_QUICK15_3EP_20261019/`; 15-sequence manifest `experiments/manifests/m1_amcm_prefusion_quick15_20261019.txt`; tracking results under `output/test/tracking_results/tbsi_track/`.
