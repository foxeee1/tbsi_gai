# AMCM spatial residual centering diagnosis (2026-10-19)

## Protocol

Direct parent was the routed token-match AMCM. The single added AMCM-internal operation subtracted each channel's spatial mean from each predicted message before applying the existing cross-modal match route. All training settings and the ten fixed M1 test events were held constant. Training completed for 3 epochs, 512 samples/epoch, seed 42, FP32, batch 2, worker 4; 1x NVIDIA GeForce RTX 4090 D. This is fixed-state analysis, not closed-loop sequence performance.

- Config SHA256: `0cbf8e014fb1abba4c4dfe05ea9d1464f7b9be18f93b219e89ed1b28a931fdd3`.
- AMCM source SHA256: `a0f4ddc1197f2277267a544551ec630c51d711d84c610df25463225bcbed0550`.
- Routed-parent source snapshot: `experiments/snapshots/amcm_reliability_tmatch_residual_center_20261019.py` with the two spatial-centering lines removed; source hash of the trained routed parent was `3bf9e9b318930c219018d0297aad27ecc0efd03f6efda9ef831dd7ccfa777dc6`.
- Epoch 3 checkpoint SHA256: `3601738bf2ffe0ba382ebb1084a35f0dddabbb43b613937e205d0bad958f70df`.
- The M1 disabled-AMCM arm matched its fixed-state reference IoU values; zero-initialized message output starts at the parent representation. Training logs report finite nonzero AMCM gradients for all 16/16 parameters.

## Results

Mean fixed-state IoU was M1 `0.301368` and candidate `0.301410`, Δ`+0.000042`; 8 increased and 2 decreased by numerical amounts. GT rank improved `0/10`, top-1 attribution flipped `0/10`, and peak margin improved in 6/10 but by at most `0.00017`, too small to change ranking. Average target/background residual magnitude was RGB `0.00058/0.00149` and TIR `0.00149/0.00249`; the correction was larger on average at the hard-background positions. Mean residual/input ratios were RGB `0.015%` and TIR `0.028%`.

| Sequence | Frame | Δ fixed IoU | GT rank M1→candidate | Δ margin | RGB residual T/B | TIR residual T/B |
|---|---:|---:|---:|---:|---:|---:|
| `7rightorangegirl` | 343 | +0.00005 | 1→1 | -0.00002 | 0.93 | 1.24 |
| `ab_bolstershaking` | 2952 | +0.00013 | 1→1 | +0.00006 | 3.56 | 0.17 |
| `blkboydown` | 49 | -0.00013 | 6→6 | -0.00004 | 1.80 | 2.05 |
| `boyss` | 218 | -0.00002 | 5→5 | -0.00001 | 2.54 | 2.65 |
| `foamatgirl\`srighthand` | 67 | +0.00006 | 2→2 | +0.00003 | 1.10 | 1.87 |
| `girldownstairfromlight` | 708 | +0.00009 | 1→1 | +0.00003 | 0.92 | 0.53 |
| `leftopenexersicebook` | 120 | +0.00003 | 7→7 | +0.00005 | 1.71 | 16.64 |
| `rightbluewhite` | 236 | +0.00002 | 1→1 | -0.00004 | 0.87 | 0.36 |
| `rightdarksingleman` | 883 | +0.00012 | 1→1 | +0.00017 | 0.50 | 0.52 |
| `umbrella` | 197 | +0.00009 | 3→3 | +0.00010 | 1.05 | 0.88 |

## Attribution

Decision: `NOT-PROMOTED / TARGET-SELECTIVE-CORRECTION-NOT-SUPPORTED`. Spatial mean removal does not address the common bottleneck: the residual remains extremely small and is not larger at target locations in aggregate. The loss gives gradients and updates to AMCM, but the fixed-frame target/background ranking barely moves. Together with the token-match and match-route trials, this is the third distinct AMCM variant without peak-order improvement. Following the repository failure protocol, further operator expansion is paused. The next step is an AMCM-only signal audit that separately measures token-match-map GT/background separability, loss gradient projected onto target-vs-background residual directions, and the effect of the trained residual on `tbsi_fuse_search` output. These determine whether the remaining constraint is input signal quality, optimization alignment, or fusion sensitivity before any fourth structural edit.

Evidence: `../output/experiments/M1_AMCM_TMATCH_RESIDUAL_CENTER_FIXED10_DIAG_20261019/{summary.json,paired_per_sequence.csv,per_event.csv,tensors/}` and `../output/experiments/M1_AMCM_TMATCH_RESIDUAL_CENTER_QUICK15_3EP_20261019/logs/`.
