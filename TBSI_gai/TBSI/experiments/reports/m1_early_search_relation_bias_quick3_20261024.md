# M1 Early Search Relation Bias Quick3

## Question

Does preserving the full early template-to-search correspondence and injecting RGB/TIR relation as a bounded additive bias into TBSI-E layer 9 attention logits provide more reliable localization control than the previous template Value gate?

## Contract

- Parent: `M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023`, epoch 15.
- Candidate: only the relation-bias branch is trainable; parent parameters are frozen.
- Training: LasHeR train, 512 samples/epoch, 3 epochs, batch 2, FP32, worker 4, seed 42.
- Test: frozen 12-sequence train-side cohort, serial `threads=0`, FP32, template update off.
- Baseline: recorded `lam0` arm from `M1_EARLY_GATE_STRENGTH_REPLAY_20261022` on the same cohort.

## Result

| arm | AUC | OP50 | OP75 | Precision | NPrec |
|---|---:|---:|---:|---:|---:|
| recorded lam0 | 78.826 | 94.255 | 77.544 | 97.350 | 92.876 |
| relation bias epoch 3 | 73.579 | 87.856 | 72.148 | 90.230 | 86.274 |

Paired mean AUC delta is `-5.247` with 7 sequence wins and 5 losses. `whitebikebehind` is the decisive failure: AUC falls from `80.692` to `15.625`; the first predicted boxes match the ground truth, then the closed-loop state drifts rapidly. The trained relation parameters are present in the checkpoint and the learned head gain is nonzero, so this is not a missing-parameter or shape-contract failure.

## Decision

**Refuted as a reliable routing operator; stop this exact form.** The additive relation bias can alter attention routing, but with the current unsupervised tracking loss and unconstrained relation projections it is not calibrated against downstream state feedback. Do not extend this branch or promote it to full training.

The only justified follow-up is a low-cost contract audit: run the same checkpoint with gain forced to zero and with frozen projections, then measure the maximum relation-logit perturbation and first divergence on identical states. A new training candidate should only be designed if that audit separates routing magnitude from the current catastrophic sequence-specific drift.
