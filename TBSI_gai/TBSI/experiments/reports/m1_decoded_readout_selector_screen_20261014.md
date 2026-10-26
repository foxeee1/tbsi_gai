# M1 decoded readout selector screen (2026-10-14)

## Purpose

Test whether deployment-visible fixed-state signals can decide when to apply a bounded readout action. Action utilities come from the real decoded-box IoU replay; no network parameters or checkpoints were changed.

## Protocol

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15.
- Cohort: 72 train-side sequences and 144 fixed states with identical decoded/signal keys.
- Split: leave-one-sequence-out ridge model for each held-out sequence.
- Actions: no-op, readout offset oracle, readout size oracle. Oracle labels measure utility but do not provide deployable parameters.

## Results

| Signal model | Threshold | Coverage | Sequence-macro utility | Harmful rate |
|---|---:|---:|---:|---:|
| base rank/margin | 0.000 | 97.2% | +0.00707 | 0.69% |
| base rank/margin | 0.010 | 33.3% | +0.00388 | 0.00% |
| evidence signals | 0.000 | 93.1% | +0.00599 | 2.08% |
| all available signals | 0.000 | 93.8% | +0.00499 | 3.47% |

The base selector retained 48.3% of offline oracle utility at threshold zero. Adding modality/template signals did not improve risk-adjusted selection.

## Decision

`DIAGNOSTIC-COMPLETE / NO-OP-SIGNAL-SUPPORTED / PARAMETER-PREDICTOR-NOT-YET-VALIDATED`.

The simple base signal identifies a low-risk subset for readout intervention in fixed-state decoded replay. This permits one narrowly scoped follow-up: predict bounded offset/size deltas under the base selector and evaluate decoded IoU on held-out sequences. It does not authorize feature residual, adaptive router, or full training. Current action labels use GT readout values and are not deployable gains.

## Evidence

- Selector output: `output/experiments/M1_DECODED_READOUT_SELECTOR_SCREEN_20261014_retry1/`.
- Script: `tools/screen_m1_decoded_readout_selector_20261014.py`.

## Limitations

Fixed-state offline replay is not closed-loop tracking. The cohort is train-side and exploratory. The next parameter-prediction test must use sequence-held-out targets and explicit no-op fallback.
