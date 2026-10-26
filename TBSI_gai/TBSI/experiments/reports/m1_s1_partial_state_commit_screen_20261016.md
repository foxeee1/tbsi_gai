# S1 partial state commit screen

## Question

Can reducing the amount of S1 state submitted to the next crop preserve S1 gains while reducing remaining162 degradation?

## Protocol

- Frozen protected M1 epoch15 checkpoint, S1 peak-margin predictor and independent action policy.
- Same five-sequence screen used for the earlier S1/B2 mechanism checks: `manoncall`, `boyss`, `girldownstairfromlight`, `bike2left`, `girlinrain`.
- Only state writeback changed. Reported output remains S1.
- Center-only: `alpha_center=0.5`, `alpha_size=0`, runid 1063.
- Size-only: `alpha_center=0`, `alpha_size=0.25`, runid 1064.

The full-244 center run was started as runid 1061 but stopped after 7 sequences because the selected RPP sequence ordering contained very long sequences and projected runtime exceeded four hours. Its partial output is retained as incomplete evidence and is not used for metrics.

## Results

| Sequence | M1 | S1 | Center 0.5 | Size 0.25 |
|---|---:|---:|---:|---:|
| manoncall | 68.94 | 67.77 | 69.15 | 69.37 |
| boyss | 54.10 | 46.13 | 45.47 | 45.31 |
| girldownstairfromlight | 46.50 | 46.47 | 46.67 | 46.43 |
| bike2left | 79.62 | 79.49 | 79.47 | 79.64 |
| girlinrain | 84.40 | 84.49 | 84.49 | 84.49 |
| **mean** | **66.71** | **64.87** | **65.05** | **65.05** |

Center-only is `+0.18` AUC over S1 on this screen; size-only is `+0.18`. Both remain `-1.66` below M1 because `boyss` remains severely degraded: center-only `-8.62` and size-only `-8.79` versus M1.

## Conclusion

`PARTIAL-COMMIT-NOT-SUPPORTED-AS-SOLUTION`. Reducing either center or size submission independently does not remove the remaining162-like failure represented by `boyss`. The two alpha values are useful mechanism probes, but neither merits full-244 promotion. The next priority is train-side state-aware supervision or a future-harm state target; another alpha sweep is not justified.

Evidence:

- `output/experiments/M1_S1_PARTIAL_STATE_SCREEN_20261016/center_half_difficult5.csv`
- `output/experiments/M1_S1_PARTIAL_STATE_SCREEN_20261016/size_quarter_difficult5.csv`
- `output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42_1063/`
- `output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42_1064/`
