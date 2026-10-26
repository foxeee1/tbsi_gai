# Early-conditioned MSHA gate: operator and sequence analysis

## Question

Which part of the trained gate formula explains its sequence-specific gains and losses, and what is the next useful optimization target?

## Operator

At early ViT block 2, normalized RGB/TIR template token `i` is compared with all 256 same-modality search tokens:

`u_i^m = max_j cosine(t_i^m, s_j^m)` and `a_i = cosine(t_i^RGB, t_i^TIR)`.

The three values `[u_i^RGB, u_i^TIR, a_i]` enter a per-token `3 -> 32 -> 2` MLP. The output is bounded as:

`d_i = lambda * tanh(early_gate_scale) * tanh(MLP(input_i))`

`g_i = exp(d_i)`, clipped to `[0.25, 4]`.

At each of TBSI locations 3, 6, and 9, `g` scales only cross-modal template Value messages. Index 0 scales RGB Value sent to TIR queries; index 1 scales TIR Value sent to RGB queries. The QK attention distribution, self-message, CENTER maps, loss, and crop update are not directly changed by the gate. Because the max pools over search positions, the gate does not retain the position where a match occurred. The MLP receives no foreground/background label or local ambiguity measure in this experiment.

## Current full-15 checkpoint: representative test replays

The paired FP16 replay used the same frame order and tracker initialization for both checkpoints, with template update off. There are 3,100 aligned tracking frames across seven sequences. The sequence subset had already been selected as large gains or regressions from the prior full-test report; sequence means below are mechanistic replay evidence, not independent or full-test estimates.

| Sequence | Mean-IoU delta | Early RGB match | Early TIR match | Template agreement | RGB->TIR gate | TIR->RGB gate |
|---|---:|---:|---:|---:|---:|---:|
| `womanback2car` | -0.2387 | 0.6809 | 0.7982 | 0.6645 | 1.0006 | 0.9894 |
| `shinycarcoming` | +0.0678 | 0.7577 | 0.7265 | 0.6448 | 1.0006 | 0.9895 |
| `boydownplatform` | -0.0133 | 0.6869 | 0.7797 | 0.6214 | 1.0006 | 0.9895 |
| `4thboywithwhite` | +0.0327 | 0.6392 | 0.7088 | 0.5765 | 1.0004 | 0.9900 |
| `ab_bikeoccluded` | +0.0013 | 0.6493 | 0.7350 | 0.7203 | 1.0005 | 0.9895 |
| `3rdfatboy` | +0.0441 | 0.6339 | 0.7601 | 0.5377 | 1.0004 | 0.9900 |
| `leftfarboycomingpicktheball` | +0.0475 | 0.7221 | 0.7841 | 0.5784 | 1.0006 | 0.9896 |

The direction-level means are nearly invariant across these sequences: RGB->TIR is about `1.0004-1.0006`, while TIR->RGB is about `0.9894-0.9900`. This is closer to a learned global directional preference than sequence-adaptive reliability. It does not explain which sequence gains: `womanback2car` and `shinycarcoming` have essentially the same mean directional gates but opposite IoU outcomes. Likewise, the early similarity and agreement values do not order the sequence outcomes monotonically.

Using the actual Hann-weighted CENTER response, a selected peak changes before sustained box-center divergence only on `womanback2car`; most other sequences diverge at the box center first. Thus the observed sequence effects do not support a consistent CENTER-ranking repair by this operator.

## Prior formula tests

- **Amplitude only:** In the train-side 12-sequence closed-loop strength replay, `lambda=1` averaged `+0.612` AUC points over bypass, but `motofromdark` contributed about 87% of the gain. Excluding it, the other 11 sequences averaged only `+0.089`; there were still two losses at lambda 1. This establishes that the value path can perturb later trajectories, but rejects a broad amplitude-based explanation. Evidence: `experiments/reports/m1_early_gate_strength_replay_12seq_20261022.md`.
- **Learned peak-versus-local-background correction:** Adding a `7 -> 16 -> 2` margin branch lost `0.892` AUC points against gate-only, with 4 wins and 8 losses. `motofromdark` fell `8.844` points and predictions changed from the first few frames. This rejects the tested local-max ambiguity descriptor and short-training contract; it does not reject all target-conditioned evidence. Evidence: `experiments/reports/m1_early_gate_target_margin_quick3_12seq_20261022.md`.
- **Direct scalar local margin:** On 25 fixed events / 75 TBSI locations, the direct formula changed gates by roughly `0.0011-0.0013`, while CENTER target-background margin changed only `1.29e-5`; GT rank changed zero times. This rejects increasing rho on that same evidence signal. Evidence: `experiments/reports/m1_early_direct_margin_fixed25_20261022.md`.

## Diagnosis and route

**Conclusion: `CURRENT_FORMULA_NOT_SUPPORTED_AS_A_SEQUENCE-ADAPTIVE_TARGET_SELECTOR`; the MSHA Value path remains capable of influencing tracking state.** The current formula lacks two properties needed to correct a location ranking reliably: it collapses the search dimension with `max`, and it does not identify which template tokens belong to the target. The learned asymmetry is almost constant by sequence, while outcomes vary sharply. The margin variants tested so far add local ambiguity without fixing target identity and are not a sound basis for another MLP or a strength sweep.

The next discriminating check should use the current full-15 candidate checkpoint and fixed crops/states, comparing `TBSI_TEST_EARLY_GATE_STRENGTH=0` against `1`. Log Hann top-1, GT-vs-hard-background rank/margin, and the selected cell's size/offset and IoU. This separates the inference-time contribution of the gate from the other jointly trained checkpoint changes. Use the seven existing representative sequences plus a prespecified calibration set; do not tune from their sequence identities.

Only if active gate improves same-state target ranking across multiple sequences should the operator be changed. The next formula to test should build a foreground-weighted template representation and retain the 16x16 search axis, so target evidence and local competitors are evaluated at the same candidate location. Inject that spatial signal on the search-side fusion path with an identity fallback. Fixed-state rank and decoded-box improvement is the pretraining gate; if it fails, stop this formula family. Do not add template update or another generic fusion branch to the current model.

## Limits

The seven sequences are outcome-selected, so their gate-outcome association is descriptive. The full-244 sequence CSV previously disagreed with clean same-checkpoint replays for several sequences and remains unsuitable for sequence-level attribution pending clean replay. The full-15 point metrics remain a one-seed directional result with sequence-bootstrap intervals crossing zero.
