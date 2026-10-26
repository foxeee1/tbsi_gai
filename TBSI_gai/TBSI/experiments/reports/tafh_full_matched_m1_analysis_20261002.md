# TAFH Full Matched M1 Analysis

## Result

TAFH_FULL_MATCHED_M1_seed42 completed the planned 15-epoch training and serial full LasHeR test: 244/244 sequences.

| Metric | TBSI-E M1 | TAFH | Delta |
|---|---:|---:|---:|
| AUC | 56.21 | 55.19 | -1.02 |
| OP50 | 68.33 | 67.12 | -1.21 |
| OP75 | 47.13 | 46.40 | -0.73 |
| Precision | 70.21 | 68.81 | -1.40 |
| Norm Precision | 66.20 | 65.18 | -1.02 |

The comparison is against the completed TBSI-E M1 full result recorded in the asset registry. TAFH therefore regresses on all five reported metrics. It is not a positive full-confirmation result.

## Training And Utilization

The TAFH diagnostics contain 283 training records and 1,095 test records. Training epoch-15 averages were:

- `rho_rgb = 0.001484`, `rho_tir = 0.000537`.
- Mean 4x4 foreground mask occupancy was about `0.248` for both modalities.
- Mean gamma at epoch 15 was approximately RGB `[-0.04446, -0.05724, 0.00216]` and TIR `[-0.01098, -0.03007, 0.01099]` for LH/HL/HH.
- The checkpoint contains the TAFH module only in `backbone.tbsi_layers.2`, corresponding to TBSI location 9.

The branch is not dead: gamma receives gradients from zero initialization, then address and band projections receive gradients after gamma becomes nonzero. However, the final injected residual is only around 0.05%--0.15% of the receiving template feature norm in the logged training summaries. The branch was learned and updated, but its effective contribution remained small.

The repeated `2/238 params have zero gradient` messages are not evidence that the whole TAFH branch is disconnected. They are consistent with parameters that are unused in this TBSI-E route; the TAFH-specific staged P0 gradient test and checkpoint construction both passed.

## Code Audit

The implementation path is:

1. `TBSITrackActor` builds a foreground token mask from the template annotation when TAFH is enabled.
2. `VisionTransformerTBSI` enables `TargetAnchoredFreqExchange` only when the TBSI layer location is in `FREQ_EXCHANGE_LOC`.
3. At location 9, `V2TemplateFusion` applies TAFH before concatenation, LayerNorm, and QKV generation.
4. TAFH uses LL bands only for local address matching; LH/HL/HH source bands are projected, masked, and reconstructed with zero LL before residual addition.

P0 passed on CPU and CUDA: Haar reconstruction, finite forward, zero-gamma identity, staged gradient flow, zero-mask identity, and local-address restriction. The model-build smoke confirmed locations `(3: off, 6: off, 9: on)`.

One interpretation detail matters: the current implementation uses the same template box-derived mask for RGB and TIR, and `_address` multiplies the post-softmax attention by the source mask without a second renormalization. Thus the effective exchange strength is coupled to foreground-mask occupancy. This is a design property, not a runtime failure, but it can suppress the already small residual and makes the result a test of this exact masked operator.

## Diagnosis

The result supports a coordination/scale limitation rather than a basic implementation failure. The module is connected, finite, and trainable, but the optimizer settles on a weak residual while the full tracker loses performance. The current evidence does not show that target-anchored high-frequency exchange provides useful incremental information over TBSI-E.

This is a single-seed result, but the regression is large relative to the prior P1 full result (`56.20` AUC) and is negative on every headline metric. It should be treated as `VALID-FULL-NEGATIVE`, not as a candidate for a second seed or a larger frequency module.

## Decision

Stop the TAFH frequency-exchange line in its current form. Do not proceed to dynamic gates, additional bands, or full confirmation on another seed. Preserve the code, checkpoints, diagnostics, and report as a negative controlled experiment. The most informative future control, only if the frequency direction is reopened, would be a matched no-frequency continuation or a parameter-matched spatial adapter; it should be chosen only after fixing the mask/normalization contract and would test whether the loss comes from the exchange mechanism or from the added trainable path.

Evidence:

- `output/experiments/TAFH_FULL_MATCHED_M1_seed42/logs/tafh_diagnostics.jsonl`
- `output/experiments/TAFH_FULL_MATCHED_M1_seed42/logs/gradient_flow.csv`
- `output/experiments/TAFH_FULL_MATCHED_M1_seed42/checkpoints/TBSITrack_ep0015.pth.tar`
- `output/experiments/tafh_full_matched_m1_20261001/full_pipeline.log`
- `output/test/tracking_results/tbsi_track/TAFH_FULL_MATCHED_M1_seed42/`
