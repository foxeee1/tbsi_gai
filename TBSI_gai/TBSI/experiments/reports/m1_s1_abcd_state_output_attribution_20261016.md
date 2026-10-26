# M1-S1 A-D state/output attribution (2026-10-16)

## Protocol

Four counterfactual arms use the same protected M1 checkpoint, frozen S1 predictor/policy, dataset manifest, FP32 tracker, and official 244-sequence evaluation:

| Arm | Reported output | Next tracker state |
|---|---|---|
| A | M1 | M1 |
| B | S1 | M1 |
| C | M1 | S1 |
| D | S1 | S1 |

C was run with the new `state_only` feature gate. M1, S1, and B reuse their completed matched runs; no protected asset was overwritten.

## Results

| cohort | n | A M1/M1 AUC | B S1/M1 AUC | C M1/S1 AUC | D S1/S1 AUC |
|---|---:|---:|---:|---:|---:|
| low-AUC62 | 62 | 22.1797 | 22.1448 | 25.9612 | 25.9071 |
| normal20 | 20 | 85.8051 | 85.7506 | 85.8489 | 85.8081 |
| remaining162 | 162 | 65.5768 | 65.4903 | 64.6245 | 64.5298 |
| all244 | 244 | 56.2077 | 56.1370 | 56.5400 | 56.4599 |

The same pattern is present in the other metrics:

| cohort | arm | PR | NPR | OP50 | OP75 |
|---|---|---:|---:|---:|---:|
| remaining162 | A | 83.0499 | 78.1479 | 81.0285 | 54.1382 |
| remaining162 | B | 83.0157 | 77.9499 | 80.9430 | 53.7566 |
| remaining162 | C | 81.6116 | 77.0252 | 79.7988 | 53.3796 |
| remaining162 | D | 81.6046 | 76.8327 | 79.7435 | 52.9875 |

## Attribution

The state effect is the dominant cause of remaining162 degradation:

- B vs A: AUC `-0.0865`, while preserving S1 output and reverting the state to M1.
- C vs A: AUC `-0.9522`, while keeping M1 output and using S1 state.
- C vs B: AUC `-0.8658`, isolating the additional harm caused by S1 state feedback.
- D vs C: AUC `-0.0947`, so changing M1 output to S1 output adds only a small effect once the S1 state is already contaminated.

The state feedback is useful on low-AUC62: C improves AUC over A by `+3.7815`, and D improves it by `+3.7274`. Therefore globally disabling S1 state would recover remaining162 but erase the intended low-AUC gain. The correct solution is selective state submission, not a global output-only mode.

## Practical solution

Use a two-track controller:

1. Always compute the M1 state update and keep it as the safe state.
2. Compute S1 output and a shadow S1 state.
3. Commit the shadow state only after a failure-conditioned confirmation rule; otherwise keep M1 state.
4. Keep M1 output as the fallback. Apply center reranking only when the same failure gate is active.
5. On the first post-commit deterioration, restore the previous M1 state rather than only replacing the reported box.

The next experiment should therefore test a train-side, sequence-held-out state-commit gate on the four-way attribution target. It must be evaluated first on the frozen remaining162 loss list and low-AUC62 gain list, with no test-side threshold tuning.

## Evidence

- Summary: `output/experiments/M1_S1_STATE_ONLY_FULL244_20261016/abcd_summary.json`
- Per sequence: `output/experiments/M1_S1_STATE_ONLY_FULL244_20261016/abcd_per_sequence.csv`
- C manifest and tracking log: `output/experiments/M1_S1_STATE_ONLY_FULL244_20261016/manifest.json`
- C runner: `tools/run_m1_s1_state_only_20261016.py`
- Evaluation script: `tools/evaluate_m1_s1_abcd_20261016.py`

## Limitations

This is a single frozen M1 checkpoint and one official test set. The four arms establish mechanism attribution for the current S1 predictor; they do not prove that a new gate will generalize. Center selection remains a second bottleneck inside severe-onset states and should be addressed only after state submission is made safe.
