# M1 / S1 / B2 full-244 comparison

## Protocol

- Dataset: LasHeR official test split, 244/244 sequences.
- M1 and S1: existing protected full-244 outputs from `M1_S1_FULL244_CONFIRM_20261014`.
- B2: `M1_TARGET_COMPLEMENTARY_ADAPTIVE_PROXY_20261016`, epoch-3 proxy checkpoint, FP32, runid 244; output contains 244 bbox and 244 time files.
- B2 method: target-conditioned complementary residual after MSHA, with bounded residual and noop admission. M1 backbone and CENTER head were frozen; only the new branch was trained.
- This is a full test of a proxy-trained B2 checkpoint, not a full-training claim.

## Overall metrics

| Method | AUC/Success | Precision | Norm Precision | OP50 | OP75 | Mean IoU |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 56.21 | 70.21 | 66.20 | 68.33 | 47.13 | 57.13 |
| S1 | 56.46 | 70.54 | 66.53 | 68.77 | 47.15 | 57.41 |
| B2 | 55.65 | 69.56 | 65.73 | 67.80 | 46.69 | 56.58 |
| B2 - M1 | -0.56 | -0.64 | -0.47 | -0.54 | -0.44 | -0.55 |
| B2 - S1 | -0.81 | -0.98 | -0.80 | -0.97 | -0.45 | -0.84 |

The canonical metrics above are from the same evaluator and 244-sequence scope. B2 does not recover M1 or S1 on the aggregate metrics.

## Sequence-level result

The complete 244-row comparison is in [`per_sequence_m1_s1_b2.csv`](../../output/experiments/M1_TARGET_COMPLEMENTARY_ADAPTIVE_PROXY_20261016/full244_analysis/per_sequence_m1_s1_b2.csv). It contains AUC, PR, NPR, OP50, OP75 and IoU for all methods, plus B2-minus-M1 and B2-minus-S1 deltas.

For AUC, relative to M1, B2 has 109 wins, 125 losses and 10 ties (tie band +/-0.05); mean delta -0.556 and median delta -0.089. Relative to S1 it has 112 wins, 120 losses and 12 ties; mean delta -0.808 and median -0.038.

### Largest B2 gains over M1 (AUC points)

| Sequence | Delta |
|---|---:|
| blktribikecome | +58.18 |
| 3bike1 | +57.75 |
| 3rdfatboy | +54.61 |
| leftfarboycomingpicktheball | +49.77 |
| AQgirlwalkinrain | +14.38 |

### Largest B2 regressions from M1 (AUC points)

| Sequence | Delta |
|---|---:|
| bikefromlight | -59.16 |
| blackdown | -57.91 |
| umbrellawillopen | -57.78 |
| carcominginlight | -41.04 |
| whiteridingbike | -26.64 |
| shinycarcoming | -26.13 |
| rightblkfatboyleftwhite | -25.27 |
| rightboy504 | -21.37 |
| carcomingfromlight | -14.63 |
| ab_whiteboywithbluebag | -14.36 |

Relative to S1, the strongest gains are `blktribikecome` (+58.52), `3rdfatboy` (+55.16), and `bluegirlbiketurn` (+30.57); the strongest regressions are `bikefromlight` (-59.72), `ab_bikeoccluded` (-57.71), `blackdown` (-56.45), and `carcominginlight` (-41.23).

## Attribution

The branch is executable and the full test is complete, but the result is negative at the aggregate level. Large wins and losses are concentrated in a small set of sequences, while the overall win/loss counts are close to balanced. This pattern does not support a general target-conditioned complementary fusion gain under the current proxy supervision and closed-loop tracker contract. In particular, the severe regressions show that the noop gate did not reliably reject harmful residuals after state feedback.

No `remaining162` label manifest was found in the canonical full-244 files, so this report does not invent a remaining162 subgroup statistic. The full CSV can be joined to the existing failure-map manifest when that subgroup is explicitly available.

## Decision

`NO-GO` for full training of B2 as currently implemented. Preserve the code and checkpoint as negative evidence. The next useful experiment is a single-variable admission test that forces B2 residuals to affect output while keeping M1 state unchanged; only if that fixed-state test is positive should a state-commit variant be considered. Do not add another fusion operator before resolving whether the current branch's losses come from residual quality or state feedback.
