# M1 Spatial Token Bridge Learnability Diagnostic (2026-10-24)

## Question

Did the bridge's previously tiny fixed-state effect come from too few optimizer updates, or does the learned message remain weakly coupled to localization under the current insertion and readout contract?

This is a bridge-only learnability diagnostic with a train-side calibration replay. It is not full tracking evaluation and cannot establish benchmark performance.

## Protocol

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15. Parent config and checkpoint were read-only and their recorded hashes matched.
- Candidate: `M1_SPATIAL_TOKEN_BRIDGE_LEARNABILITY_10EP_20261024`; only `SpatialTokenBridge` was trainable.
- Data/optimization: LasHeR train proxy, 16 samples per epoch, batch 2, 10 epochs / 80 optimizer steps, seed 42, FP32, four workers.
- Saved checkpoints: epochs 1, 5, and 10.
- Evaluation: same 23 train-side calibration sequences and 46 pre-onset/severe-onset fixed states at each checkpoint; FP32 serial arms `off`, `target`, and equal-density `shifted`.
- Preregistered admission gate: at epoch 10, target-over-shifted localization margin must exceed `1e-3`, improve on a majority of sequence means, and not regress CENTER rank or replay IoU.
- Evidence: `output/experiments/M1_SPATIAL_TOKEN_BRIDGE_LEARNABILITY_10EP_20261024/` and the nine `M1_SPATIAL_TOKEN_BRIDGE_LEARNABILITY_E{1,5,10}_{OFF,TARGET,SHIFTED}_FIXED46_20261024/` directories.

## Contract and optimization

Training completed all 80 planned steps. The bridge received finite gradients and its parameters changed. No parent modules were intentionally trained. The bridge message increased with optimization exposure:

| Checkpoint | Mean RGB delta norm | Mean TIR delta norm |
|---|---:|---:|
| Epoch 1 | 0.000994 | 0.000483 |
| Epoch 5 | 0.002016 | 0.001264 |
| Epoch 10 | 0.002633 | 0.001563 |

Thus the 8-step undertraining explanation is incomplete: more bridge-only optimization did increase message magnitude. This does not by itself show that the message is useful.

## Paired fixed-state results

Values below are candidate-minus-control means over the same 46 states. CENTER/query rank changes count states whose integer rank differed; no state changed rank in any comparison shown.

| Epoch | Contrast | CENTER GT margin | Post-fusion query GT margin | Replay IoU | CENTER/query rank changes |
|---:|---|---:|---:|---:|---:|
| 1 | target - off | +1.333e-6 | +3.499e-8 | -1.504e-7 | 0 / 0 |
| 1 | target - shifted | +1.453e-6 | -2.060e-7 | -3.982e-7 | 0 / 0 |
| 5 | target - off | +4.444e-7 | -3.304e-8 | -6.530e-8 | 0 / 0 |
| 5 | target - shifted | +2.941e-7 | -6.913e-7 | -1.895e-6 | 0 / 0 |
| 10 | target - off | -1.018e-6 | +5.701e-8 | +2.325e-7 | 0 / 0 |
| 10 | target - shifted | -1.308e-6 | -1.046e-6 | -1.183e-6 | 0 / 0 |

At epoch 10, target-minus-shifted sequence-mean signs were mixed: CENTER margin improved on 8/23 sequences, query margin on 9/21 non-tied sequence means, and replay IoU on 6/23. Target-minus-off replay IoU improved on 6/23 sequences. These small state perturbations do not change the selected CENTER/query cells, and replay changes remain around numerical-noise scale.

Off-arm replay IoU matched the protected M1 trace on all 46 states at each saved epoch (within the recorded comparison precision), supporting the validity of the paired baseline path.

## Interpretation

**Conclusion: supports H2 for the current interface; stops the current operator before full training.** The branch can learn and produce a larger residual, but localization-facing margins remain around `1e-6`, ranks are identical on 46/46 states, and epoch-10 target-minus-shifted margins are negative. The preregistered `>1e-3` gate missed by roughly three orders of magnitude. Longer bridge-only fitting therefore does not rescue this form of spatial token bridge.

This evidence does not prove that spatial correspondence fusion in general is ineffective. It supports the narrower explanation that this particular final-backbone insertion, residual scale/selection, and existing localization readout do not transmit a useful learned signal under this training contract. The fixed states are from LasHeR train and were used for calibration, so these results are diagnostic, not held-out evidence.

## Decision and next step

- Do not run the planned full-training experiment with this unchanged operator.
- Retain the implementation, checkpoints, paired output directories, and logs as a negative mechanism result.
- The follow-up amplitude diagnostic is complete. Scale-1 reproduced this replay exactly; 32x/128x changed readouts, but target correspondence did not improve replay over shifted and the high-dose query-margin signal did not carry through CENTER to boxes. See `experiments/reports/m1_spatial_token_bridge_amplitude_fixed46_20261024.md`; stop the current bridge and amplitude-scaling line before full training.
- No benchmark metric is reported (`AUC/PR/NPR/OP50/OP75/FPS = NA`); no full test was run.
