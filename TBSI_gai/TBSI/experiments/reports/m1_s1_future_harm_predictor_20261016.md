# M1/S1 future state harm predictor diagnostic

## Research question

Can future closed-loop damage caused by committing the S1 state be predicted
from signals available at the current frame? This is the minimum test before
adding another state gate.

## Contract

- Parent: frozen M1 epoch-15 checkpoint and frozen S1 readout predictor/policy.
- Source: `M1_S1_CLOSED_LOOP_RISK_DIAGNOSTIC_20261015/paired_rollout.csv`.
- Scope: 12 whole LasHeR training sequences selected from the frozen 72-sequence
  train-side manifest; 3,675 rows, with 3,555 usable H10 and 3,315 usable H30
  labels.
- Label: `future_h = mean(IoU(M1-state) - IoU(S1-state))` over the next H
  frames. Positive means S1 state feedback is harmful.
- Validation: sequence-held-out logistic fit; no test sequence contributes to
  its own fit. No tracker or checkpoint was changed.
- Stop condition: deploy a gate only if an available signal has stable held-out
  ranking better than chance. This run is diagnostic, not a full-244 result.

## Results

| Horizon | Signal/model | Pooled AUROC | Sequence-mean AUROC |
|---:|---|---:|---:|
| 10 | gap-only | 0.2585 | 0.5007 |
| 10 | action scores | 0.4273 | 0.5115 |
| 10 | gap + action + instantaneous observable | 0.4250 | 0.5487 |
| 10 | instantaneous IoU harm (offline reference) | 0.5648 | NA |
| 30 | gap-only | 0.2149 | 0.5596 |
| 30 | action scores | 0.4243 | 0.5253 |
| 30 | gap + action + instantaneous observable | 0.4227 | 0.5620 |
| 30 | instantaneous IoU harm (offline reference) | 0.5655 | NA |

`instantaneous IoU harm` is included only as an offline reference. It uses the
paired M1-state rollout and therefore is unavailable to an online tracker; it
must not be treated as a deployable gate.

## Interpretation

The available current-state displacement and action score signals do not rank
future state damage reliably. The fitted combination is below 0.43 pooled
AUROC at both horizons, and its sequence-level mean is close to chance. This
supports `REACHABILITY-SIGNAL-NOT-SUPPORTED` and rejects adding another
threshold, hysteresis, alpha, or logistic state gate under the current
observation contract.

The result does not prove that a future-harm predictor is impossible. It shows
that the current gap/action observables do not contain enough information on
this 12-sequence diagnostic. A larger 72-sequence diagnostic is optional, but
it should only be run if a new feature source is first identified; expanding
the same feature set would not be a meaningful next experiment.

## Attribution and next route

- Excluded: simple commit coverage, current risk probability, state gap,
  action magnitude, and their current-frame combination as a sufficient
  state-contract signal.
- Still unresolved: whether a target-aware signal before CENTER selection can
  improve the action itself, and whether training under S1-state rollouts can
  align the correction with the altered crop distribution.
- Next route: stop state-gate engineering. Return to one train-side change in
  the correction target/center selection contract, with M1 output and state
  paths logged separately. Do not deploy this predictor.

## Provenance

- Runner: `TBSI_gai/TBSI/tools/fit_m1_s1_future_harm_predictor_20261016.py`
- Runner SHA256: `e53c10e85158212fd4b3bad0f08f2b2a12733ce4cc4a24a418c909d3ba3a6335`
- Metrics SHA256: `7de6c5c0e1b2e27e0fd5c2c81a01d98f1efde6907095c5494f085eefea8627c7`
- Git commit at analysis: `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`
- Evidence: `TBSI_gai/TBSI/output/experiments/M1_S1_FUTURE_HARM_PREDICTOR_20261016/`
