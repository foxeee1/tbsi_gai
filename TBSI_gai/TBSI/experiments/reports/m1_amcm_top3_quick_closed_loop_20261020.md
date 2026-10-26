# M1 AMCM 三候选快速训练与闭环测试（2026-10-20）

## Protocol

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch15 checkpoint.
- Only AMCM internal mode changed: `cbam`, `focal_modulation`, or `swinfusion`.
- Each candidate: 3 epochs, 512 training samples/epoch, batch 2, `NUM_WORKER=4`, FP32, AMCM-only training; CENTER head, backbone, loss weights, crop and optimizer settings fixed.
- Evaluation: the same 20 M1 failure events and 5 high-AUC controls from `M1_PRECISION_SCREEN_MANIFEST_20261020`. Each sequence was initialized from frame 0 and tracked frame by frame only through the selected event frame. This is a closed-loop event test, not a full-244 result.
- AMCM output is injected before `tbsi_fuse_search` as corrected RGB/TIR feature maps. No post-fusion average scaling or tracker-state policy was changed.

## Contract evidence

All three modes pass the zero-initialized identity check: maximum RGB/TIR output difference is `0.0` before training. Their message projections receive nonzero tracking-loss gradients in a synthetic backward check. During training, the diagnostic reports `4/20` AMCM parameters with zero gradient at every logged step. These are the spatial router final-layer parameters: because the residual message projection is zero initialized, the router is initially multiplied by zero and is not directly optimized. Thus the trained candidates learn message content first; the route map is not yet a fully supervised reliability estimator.

## Tracking comparison

| Candidate | Failure M1 event IoU | Failure candidate IoU | Failure delta | Wins/Losses | Control delta | All-event delta |
|---|---:|---:|---:|---:|---:|---:|
| CBAM | 0.2465 | 0.2641 | **+0.0176** | 8/7 | +0.0056 | **+0.0152** |
| Focal Modulation | 0.2465 | 0.2276 | -0.0189 | 7/8 | +0.0163 | -0.0118 |
| SwinFusion | 0.2465 | 0.2113 | -0.0352 | 7/8 | -0.0009 | -0.0284 |

The failure-group CBAM increase is exploratory and driven by a few large changes. It is not a uniform repair: several difficult events are unchanged because the parent has already lost the target by the selected frame, while two events show severe regressions. Focal and SwinFusion do not support the hypothesis that multi-scale context alone repairs M1's crop-local peak competition.

## Per-sequence event delta (candidate minus M1)

| Sequence | CBAM | Focal | SwinFusion |
|---|---:|---:|---:|
| 1stcol4thboy | +0.000 | +0.000 | +0.000 |
| 1strowleftboyturning | -0.026 | -0.027 | -0.027 |
| 2girlup | -0.007 | -0.047 | -0.047 |
| ab_bikeoccluded | +0.044 | +0.074 | +0.074 |
| ab_pingpongball2 | +0.013 | -0.008 | -0.008 |
| basketball849 | +0.002 | -0.521 | -0.521 |
| biketurnright | +0.015 | +0.023 | +0.023 |
| bottlebetweenboy`sfeet | -0.019 | -0.028 | -0.028 |
| boy`sheadingreycol | **-0.422** | +0.006 | +0.017 |
| boyruninsnow | -0.001 | -0.004 | -0.004 |
| boyunder2baskets | -0.028 | -0.010 | -0.047 |
| darkcarturn | +0.017 | -0.023 | -0.023 |
| girl`sblkbag | -0.029 | +0.017 | +0.017 |
| girlafterglassdoor | -0.035 | +0.007 | -0.028 |
| lastleftgirl | +0.000 | +0.000 | +0.000 |
| leftboy2jointhe4 | +0.000 | +0.000 | +0.000 |
| leftgirl'swhitebag | +0.012 | +0.092 | +0.092 |
| leftpingpongball | +0.000 | +0.000 | +0.000 |
| leftunderbasket | **+0.624** | -0.008 | -0.008 |
| middrillmaster | +0.192 | +0.165 | +0.165 |
| rightcar-chongT | **-0.334** | **-0.415** | **-0.415** |
| rightgirltakingcup | **+0.314** | **+0.310** | +0.008 |
| silvercarturn | +0.059 | +0.060 | +0.060 |
| whitecarturnleft | -0.011 | +0.041 | -0.010 |
| whitesuvturn | +0.000 | +0.000 | +0.000 |

Exact rows are stored in:

- `output/experiments/M1_AMCM_CBAM_QUICK15_3EP_20261020/closed_loop_25/per_sequence.csv`
- `output/experiments/M1_AMCM_FOCAL_QUICK15_3EP_20261020/closed_loop_25/per_sequence.csv`
- `output/experiments/M1_AMCM_SWINFUSION_QUICK15_3EP_20261020/closed_loop_25/per_sequence.csv`

## Diagnosis

1. **CBAM is the only candidate admitted to a second diagnostic round, not to full training.** It has the best failure-group mean and improves `leftunderbasket`, `middrillmaster`, and `rightgirltakingcup`, which is consistent with a spatial modulation effect. The severe regressions show that its generic spatial evidence can also select the wrong peak.
2. **Focal Modulation is rejected under this loss and interface for now.** Its control-group gain does not transfer to failures, and `basketball849` and `rightcar-chongT` collapse. This indicates that multi-scale context is not aligned with the target-specific objective by itself.
3. **SwinFusion is rejected under this contract.** It is worse than Focal on failures and controls, so adding broader context does not solve the missing target representation.
4. The common `4/20` zero-gradient pattern means the current residual/router parameterization is only partly trained. Increasing `MAX_DELTA` or adding another operator before fixing this optimization contract would confound the result.

## Decision

Keep CBAM as the next single-variable AMCM candidate. Before any longer run, change only its initialization/gradient contract so the spatial router receives a usable gradient while retaining near-identity output, then repeat the same 25-event closed-loop test. Do not combine CBAM with Focal/SwinFusion or change the loss yet. The present evidence supports “CBAM has a promising but unstable spatial signal,” not “AMCM solves the M1 failure chain.”

## Provenance

- Configs: `experiments/tbsi_track/M1_AMCM_{CBAM,FOCAL,SWINFUSION}_QUICK15_3EP_20261020.yaml`.
- Checkpoints: each candidate's `output/experiments/.../checkpoints/TBSITrack_ep0003.pth.tar`.
- Closed-loop script: `tools/run_m1_candidate_closed_loop_20261020.py`.
- Full 244 test was intentionally stopped after 3 sequences because its measured throughput projected over 50 minutes; no full-test metric is claimed.
