# M1 bounded readout predictor closed-loop proxy (2026-10-14)

## Protocol

- Parent: protected M1 checkpoint epoch 15, run `9904`.
- Candidate: same checkpoint with the bounded readout predictor enabled, run `9915`.
- Cohort: fixed union manifest of 62 prior low-AUC sequences and 20 normal sentinel sequences.
- Predictor: peak-margin-only offset/size ridge artifact, `rho=0.5`, threshold `0.08`, explicit no-op fallback.
- Evaluation: same tracker and per-sequence box metrics for all 82 sequences.

## Paired deltas

| Cohort | AUC delta | Precision delta | Norm Precision delta | OP50 delta | OP75 delta |
|---|---:|---:|---:|---:|---:|
| Low-AUC 62 | +3.00 | +3.71 | +3.41 | +3.68 | +2.43 |
| Normal 20 | -0.14 | -0.20 | -0.28 | -0.24 | -0.32 |
| Union 82 | +2.23 | +2.76 | +2.51 | +2.73 | +1.76 |

Sequence AUC wins/losses/ties were 35/27/0 on low-AUC, 11/8/1 on normal, and 46/35/1 overall.

## Decision

`PROXY-COMPLETE / PROMISING-LOW-AUC / NORMAL-REGRESSION-SMALL / NO-FULL-TRAIN-YET`.

The predictor transfers the fixed-state signal into a closed-loop gain on the difficult cohort. Normal-sequence degradation is small and meets the exploratory guard of `>= -0.25` AUC, but the normal NPR/OP75 deltas remain slightly negative and the proxy is single-seed. Permit one stricter repeat on the same cohort with threshold `0.08` and threshold `0.12` as a conservative arm, then decide on full training. Do not claim a general 1% benchmark gain yet.

## Evidence

- Candidate tracking results: `output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42_9915/`.
- Paired analysis: `output/experiments/M1_BOUNDED_READOUT_PROXY_20261014/{summary.json,per_sequence.csv}`.
- Predictor artifact: `experiments/artifacts/m1_bounded_readout_predictor_peakmargin_20261014.json`.
- Code gate: `lib/test/tracker/tbsi_track.py` under `TBSI_BOUNDED_READOUT_PREDICTOR`.

## Limitations

This is an 82-sequence proxy, not full-244 evaluation. The predictor was fit on train-side fixed states and has no second seed. The current trace does not yet persist per-frame gate/action counts, so a follow-up should add that audit before full training.
