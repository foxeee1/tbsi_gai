# CFVI frozen frequency attribution

Date: 2026-10-01  
Scope: RPP-v1 frozen proxy-60, serial FP32 test, seed 42  
Checkpoint: existing `CFVI_PARENT_P1_FIXED_LOC9_seed42` epoch 3  
Protocol: no training; only test-time Haar band masks with energy renormalization.

## Metrics

| Condition | Mask | AUC | OP50 | OP75 | Precision | NPR |
|---|---|---:|---:|---:|---:|---:|
| FULL | LL/LH/HL/HH | 51.64 | 62.88 | 40.98 | 66.15 | 61.22 |
| LL_ONLY / NO_HIGH | LL | 51.29 | 62.46 | 40.73 | 65.48 | 60.53 |
| HIGH_ONLY | LH/HL/HH | 51.13 | 62.32 | 40.64 | 65.30 | 60.36 |

The duplicated `NO_HIGH` condition was intentionally retained as the same mask as `LL_ONLY`; it reproduced the same metrics. `HIGH_ONLY` is the complementary no-LL condition.

## Paired sequence analysis

Relative to FULL, sequence-level AUC differences were:

| Comparison | Mean | Median | Bootstrap 95% CI | Win rate |
|---|---:|---:|---|---:|
| LL_ONLY - FULL | -0.003625 | -0.000085 | [-0.012008, 0.004610] | 48.3% |
| HIGH_ONLY - FULL | -0.004979 | +0.000200 | [-0.017806, 0.006651] | 51.7% |

Both intervals include zero. The mean loss is descriptive sensitivity evidence only, not a statistically supported ranking of bands.

## Injection audit

The P1 lambdas were `lambda_rgb=0.111053` and `lambda_tir=0.105071`. Across 62,808 runtime records, the measured `||lambda * delta_V|| / ||V_before||` was:

- RGB receiver: mean `0.1962`, median `0.1965`, 95th percentile `0.2046`; fraction above 0.01: 100%.
- TIR receiver: mean `0.1860`, median `0.1854`, 95th percentile `0.1937`; fraction above 0.01: 100%.

The source-band energy ratios before projection were approximately:

- RGB receiving TIR: LL `0.8641`, LH `0.0531`, HL `0.0616`, HH `0.0212`.
- TIR receiving RGB: LL `0.8581`, LH `0.0538`, HL `0.0667`, HH `0.0214`.

Thus the runtime branch is measurably active, but most raw source energy is in LL. Energy renormalization was enabled, so the ablation does not simply reduce the total source norm. The diagnostic record count is a forward-call count rather than an independently deduplicated event count, and the layer field was not populated; the configured CFVI location remains `[9]`.

## Conclusion

FULL is better than both masked conditions on aggregate metrics, while the sequence-level intervals are inconclusive. The result supports the narrower claim that the trained P1 output is sensitive to retaining the complete band mixture; it does not identify a uniquely useful band, prove frequency complementarity, or justify a new selective-exchange module, a second seed, M2, or full-244.

Evidence: `output/experiments/cfvi_frequency_attribution_20261001/`; raw metrics are in each condition's `analysis/eval_data.pkl` and logs.
