# M1 bounded readout predictor (2026-10-14)

## Purpose

Fit low-capacity offset and size correction predictors with a no-op fallback. Only deployment-visible `base_margin` was used; GT rank was excluded. Evaluation used sequence-held-out fixed-state decoded replay.

## Protocol

- 72 train-side sequences and 144 fixed states.
- Action location: current response peak cell.
- Separate ridge predictors for offset and size deltas.
- Correction strength `rho=0.5`; no-op when predicted utility is below threshold.

## Risk-coverage

| Threshold | Coverage | Selected decoded utility | Harmful rate |
|---:|---:|---:|---:|
| 0.00 | 100.0% | +0.07525 | 4.17% |
| 0.04 | 81.9% | +0.07065 | 1.69% |
| 0.08 | 51.4% | +0.05963 | 0.00% |
| 0.12 | 13.9% | +0.02255 | 0.00% |

## Decision

`DIAGNOSTIC-COMPLETE / BOUNDED-PREDICTOR-PROMISING / CLOSED-LOOP-PROXY-REQUIRED`.

Threshold `0.08` provides a zero-harm fixed-state subset, but this is not a tracking gain claim. The next step is a small closed-loop proxy on the prior negative-62 plus normal-20 cohort with the same predictor, `rho=0.5`, and threshold `0.08`. Full training remains unauthorized.

## Invalidated attempts

The first implementation applied offset correction while measuring size utility. The second used diagnostic-only `base_gt_rank`. Both outputs are retained but excluded. The reported result is corrected retry3 using only `base_margin` and separate action utilities.

Evidence: `output/experiments/M1_BOUNDED_READOUT_PREDICTOR_20261014_retry3/`; script `tools/fit_m1_bounded_readout_predictor_20261014.py`.
