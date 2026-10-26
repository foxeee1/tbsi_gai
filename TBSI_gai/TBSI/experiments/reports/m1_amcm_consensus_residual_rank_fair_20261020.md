# M1-AMCM-CONSENSUS-RESIDUAL-RANK-FAIR-20261020

## Why this comparison

The preceding consensus experiment compared a 3-epoch candidate against the protected M1 epoch-15 checkpoint. That comparison is not a balanced parent comparison. This run uses a matched 3-epoch control and candidate, both initialized from `TBSITrack_SOT_Pretrained.pth.tar`, with identical data, optimizer, batch, worker count and schedule.

## Single changed variable

The candidate keeps the AMCM consensus residual route and adds an AMCM-specific residual-energy ranking loss:

```text
E(x) = 0.5 * (mean_c |Delta_rgb(x,c)| + mean_c |Delta_tir(x,c)|)
L_res = lambda * max(0, margin - max_GT E + max_hard_background E)
```

The control disables AMCM and the residual loss. Backbone, TBSI, `tbsi_fuse_search`, CENTER head, tracker state and ordinary box losses are unchanged.

## Matched training

- LasHeR train: 512 samples/epoch, 3 epochs.
- Batch 2, FP32, `NUM_WORKER=4`.
- Published SOT pretrained initialization only; no M1 checkpoint.
- Candidate ranking weight `0.50`, margin `0.05`, one hardest background cell.
- Candidate AMCM gradient audit: 14/22 tracked parameters nonzero at every sampled step; `nlformer_alpha` reached about `0.0168`.
- The logged `Loss/amcm_residual_rank` was `0.00000` at sampled train and validation prints. AMCM parameters were trained by the ordinary tracking path, but activation of the new ranking term itself was not demonstrated by this run.

## Fixed 10-event results

| metric | fair control | AMCM + residual rank | candidate minus control |
|---|---:|---:|---:|
| mean fixed-state IoU | 0.29641 | 0.25275 | -0.04366 |
| mean GT rank | 3.8 | 4.5 | +0.7 (worse) |
| mean target/background margin | -0.06195 | -0.07363 | -0.01168 |
| RGB residual target/background | 0 | 0.26854 | background-dominant |
| TIR residual target/background | 0 | 0.26852 | background-dominant |

Per-event paired values are in `output/experiments/M1_AMCM_CONSENSUS_RESIDUAL_RANK_FIXED10_DIAG_20261020/per_event.csv` and the control counterpart is in `output/experiments/M1_FAIR_CONTROL_FIXED10_DIAG_20261020/per_event.csv`.

| sequence | control IoU | candidate IoU | delta |
|---|---:|---:|---:|
| 7rightorangegirl | 0.0925 | 0.2619 | +0.1694 |
| ab_bolstershaking | 0.3335 | 0.0938 | -0.2397 |
| blkboydown | 0.3547 | 0.0037 | -0.3510 |
| boyss | 0.4144 | 0.4632 | +0.0488 |
| foamatgirl`srighthand | 0.3580 | 0.3262 | -0.0318 |
| girldownstairfromlight | 0.4104 | 0.3820 | -0.0284 |
| leftopenexersicebook | 0.3003 | 0.3002 | -0.0000 |
| rightbluewhite | 0.3317 | 0.3515 | +0.0198 |
| rightdarksingleman | 0.2319 | 0.2794 | +0.0475 |
| umbrella | 0.1367 | 0.0655 | -0.0712 |

## Interpretation

The fair control removes the major training-progress confound. The AMCM path changes its output, but the ranking term itself was logged as zero and cannot yet be credited as the cause of that change. The candidate does not generalize to the selected M1 failures: all candidate residual target/background ratios remain below one. In the difficult cases `blkboydown`, `ab_bolstershaking` and `umbrella`, the candidate amplifies the wrong spatial support and loses the center ranking.

The ranking target is not yet validated as active. Even if activated, it ranks only the magnitude of the AMCM residual, not its **direction** in feature space or the post-`tbsi_fuse_search` representation. A residual can satisfy a local energy ranking while still moving the fused feature toward a background mode.

## Decision

`FAIR-FIXED10-COMPLETE / NOT-PROMOTED`.

The earlier M1-versus-3-epoch comparison is downgraded to exploratory evidence. The matched result does not support the current candidate, but it also does not cleanly falsify the ranking loss because its active value was zero. Before another training run, add an explicit raw-loss/active-pair diagnostic and verify a nonzero one-batch gradient. Only then test a signed target/background margin after `tbsi_fuse_search`. No 50-sequence run is justified for this candidate.
