# Target-conditioned CENTER-aware fusion proxy (2026-10-16)

## Hypothesis

The previous pre-CENTER residual only used fused features and RGB/TIR disagreement. Adding a pooled template target context to the residual and gate may make the learned fusion correction target-aware and improve CENTER localization.

## Contract

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15.
- One changed variable: `PRE_CENTER_GATED_RESIDUAL.TARGET_CONDITIONED=True`.
- M1 backbone, CENTER head, loss, optimizer and seed otherwise follow the proxy contract.
- M1 is frozen; only the new residual parameters train.
- Target context is the mean of RGB and TIR template tokens, broadcast over the search grid.
- The residual and gate are zero initialized; disabled/default path is unchanged.
- Training proxy: LasHeR train, 256 samples/epoch, 3 epochs, batch 2, one worker, FP32.
- Tracking screen: five fixed difficult sequences: `manoncall`, `boyss`, `girldownstairfromlight`, `bike2left`, `girlinrain`.

## Results

| method | AUC | avg IoU | delta AUC vs M1 | delta IoU vs M1 |
|---|---:|---:|---:|---:|
| M1 | 66.712 | 0.67690 | 0 | 0 |
| target-conditioned residual, 1 epoch | 64.881 | 0.65856 | -1.830 | -0.01834 |
| target-conditioned residual, 3 epochs | 66.015 | 0.67035 | -0.697 | -0.00655 |

3 epoch per-sequence AUC deltas: `manoncall -0.859`, `boyss -0.254`, `girldownstairfromlight -2.320`, `bike2left -0.204`, `girlinrain +0.153`. The module parameters changed from their zero initialized output layers, so this is an optimization-valid but performance-negative result.

## Conclusion

`DIAGNOSTIC-COMPLETE / TARGET-CONDITIONED-POOLED-CONTEXT-REJECTED`. A global pooled template vector is too coarse to correct the CENTER cell. It adds target information but does not provide a spatial target correspondence signal; on the selected difficult screen it reduces both AUC and IoU. Do not launch full training or add more gate complexity for this variant.

The remaining viable training-fusion direction must preserve spatial correspondence, for example a local target-to-search cross-attention or target-conditioned spatial correlation map, with an auxiliary center-cell objective. The next implementation should change the target signal representation only, keeping the residual, zero initialization, and frozen M1 protocol fixed.

Evidence:

- `experiments/tbsi_track/M1_TARGET_CONDITIONED_CENTER_PROXY_3EP_20261016.yaml`
- `output/experiments/M1_TARGET_CONDITIONED_CENTER_PROXY_3EP_20261016/logs/train.log`
- `output/experiments/M1_TARGET_CONDITIONED_CENTER_PROXY_3EP_20261016/logs/test_difficult5.log`
- `output/experiments/M1_TARGET_CONDITIONED_CENTER_PROXY_20261016/logs/metrics_difficult5.txt`
