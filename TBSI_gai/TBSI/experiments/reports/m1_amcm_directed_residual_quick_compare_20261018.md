# AMCM directed residual quick comparison (2026-10-18)

## Protocol

- Candidate: `M1_AMCM_DR_DIRECTED_RESIDUAL_PROXY_20261018`, 3-epoch seed42 checkpoint.
- Parent: protected M1 epoch15 checkpoint.
- Dataset: fixed difficult5 RPP LasHeR list: `manoncall`, `boyss`, `girldownstairfromlight`, `bike2left`, `girlinrain`.
- Test: `rpp_lasher_test`, FP32, same search/template settings, two independent workers.
- This is a quick mechanism screen, not a full-test result.

## Aggregate

| Method | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| M1 | 65.09 | 76.05 | 53.54 | 85.65 | 76.43 |
| Directed residual | 65.11 | 76.01 | 53.96 | 86.03 | 76.94 |
| Candidate - M1 | +0.02 | -0.04 | +0.42 | +0.38 | +0.51 |

## Per-sequence results

| Sequence | AUC delta | OP50 delta | OP75 delta | Precision delta | Norm Precision delta |
|---|---:|---:|---:|---:|---:|
| `manoncall` | -0.23 | -0.77 | +0.19 | +0.77 | +0.96 |
| `boyss` | +0.20 | +0.53 | +1.60 | +0.53 | +0.80 |
| `girldownstairfromlight` | +0.14 | +0.41 | +0.08 | +0.58 | +1.15 |
| `bike2left` | -0.06 | -0.39 | -0.39 | +0.00 | +0.00 |
| `girlinrain` | -0.00 | +0.00 | +0.64 | +0.00 | -0.32 |

## Interpretation

The directed residual path is numerically close to M1 on all five sequences. It avoids the large unstable shifts observed with legacy E2 on full-244, and its changes are concentrated in the localization thresholds rather than a large AUC shift. This supports interface and gradient compatibility, but does not yet establish that the reliability signal improves tracking. The next gate is a larger fixed sequence screen with test-time gate traces before any full-244 run.
