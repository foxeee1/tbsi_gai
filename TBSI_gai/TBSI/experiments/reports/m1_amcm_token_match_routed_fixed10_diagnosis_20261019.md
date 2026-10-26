# AMCM token-match spatial routing diagnosis (2026-10-19)

## Protocol

Direct parent was the token-match AMCM candidate. The only code change was inside AMCM: RGB residual used the TIR token-match map, and TIR residual used the RGB token-match map. Both used `2 * sigmoid((match - spatial_mean(match)) / 0.07)` as a per-location multiplier. The loss, trainable parameters, backbone, TBSI fusion, head, tracker, and fixed event set were unchanged. Training was 3 epochs, 512 samples/epoch, seed 42, FP32, batch 2, worker 4, AMCM-only.

- Config SHA256: `06ca9481426772ba332f2885158432fcdc2b0229f07761baae8c8590018ad408`.
- AMCM source SHA256: `3bf9e9b318930c219018d0297aad27ecc0efd03f6efda9ef831dd7ccfa777dc6`.
- Epoch 3 checkpoint SHA256: `753a5adaaf483cf5cedae40836d04e92ccf663ee61b91db405e73494b039cf70`.
- Evaluation: same 10 fixed M1 events, same crop/template/`state_before`, Hann-weighted CENTER map, FP32 serial. 1x RTX 4090 D. This is fixed-state mechanism analysis, not sequence-level performance.

## Paired results

Mean fixed-state IoU was M1 `0.30137` and routed candidate `0.30177`, Δ`+0.00040`; 7 increased and 3 decreased. GT rank improved `0/10`; top-1 did not flip in either direction. RGB/TIR residual target-to-fixed-hard-background magnitude ratios had means `1.64/2.74`, but were highly unstable because residual magnitudes were very small (many below `0.01`) and target/background ratios therefore became large from a near-zero denominator. They do not establish useful target selectivity. Peak margins changed by less than `0.003` on every event and did not change rank.

| Sequence | Frame | Δ fixed IoU | GT rank M1→candidate | Δ margin | RGB residual T/B | TIR residual T/B |
|---|---:|---:|---:|---:|---:|---:|
| `7rightorangegirl` | 343 | -0.0001 | 1→1 | +0.00256 | 1.14 | 1.55 |
| `ab_bolstershaking` | 2952 | +0.0016 | 1→1 | +0.00043 | 4.02 | 0.20 |
| `blkboydown` | 49 | -0.0037 | 6→6 | -0.00017 | 1.22 | 2.26 |
| `boyss` | 218 | +0.0005 | 5→5 | +0.00007 | 1.94 | 3.18 |
| `foamatgirl\`srighthand` | 67 | +0.0012 | 2→2 | -0.00007 | 0.97 | 1.91 |
| `girldownstairfromlight` | 708 | +0.0026 | 1→1 | +0.00214 | 0.59 | 0.62 |
| `leftopenexersicebook` | 120 | +0.0003 | 7→7 | -0.00044 | 3.83 | 15.68 |
| `rightbluewhite` | 236 | -0.0009 | 1→1 | +0.00007 | 0.99 | 0.54 |
| `rightdarksingleman` | 883 | +0.0020 | 1→1 | -0.00006 | 0.41 | 0.51 |
| `umbrella` | 197 | +0.0005 | 3→3 | -0.00064 | 1.31 | 0.99 |

Decision: `NOT-PROMOTED / DIRECT-SPATIAL-ROUTING-NOT-SUPPORTED`. Directly multiplying the cross-modal residual by the other stream's match map changed the residual distribution but did not change the CENTER peak ordering. The observed near-identity effect and unstable residual ratios indicate this routing alone is insufficient. Next test is to remove per-channel spatially constant message components inside AMCM, leaving only spatially varying residuals; no other path or objective changes.

Evidence: `../output/experiments/M1_AMCM_TMATCH_ROUTED_FIXED10_DIAG_20261019/{summary.json,paired_per_sequence.csv,per_event.csv,tensors/}` and `../output/experiments/M1_AMCM_TMATCH_ROUTED_QUICK15_3EP_20261019/logs/`.
