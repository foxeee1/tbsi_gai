# TAFH Mechanism Audit

The audit is non-training and uses the frozen full-run artifacts. The paired M1/TAFH table covers 244/244 sequences. A fixed development prefix audit covers 10 sequences and 80 frames.

## Paired Full Results

- Mean per-sequence AUC delta `TAFH - M1`: `-0.010156`.
- Median delta: approximately `0`.
- Sequence win rate: `49.18%`.
- Largest losses: `bikefromlight -0.596`, `umbrellawillopen -0.585`, `mandownstair -0.512`.
- Largest gains: `ab_bikeoccluded +0.574`, `3bike1 +0.574`, `leftfarboycomingpicktheball +0.470`.

The overall regression is therefore not uniform. TAFH has both large gains and large losses, with a small negative mean. This is compatible with a conditional/coordination problem, but it does not establish that the gains are reproducible or attribute-specific.

## Injection Transfer Audit

On 10 fixed development sequences and 8 frames per sequence:

| Quantity | Mean absolute change | Mean change / reference norm |
|---|---:|---:|
| `delta_z` after TAFH | 0.05468 | 4.48e-4 |
| `delta_LN` | 0.01259 | 4.58e-4 |
| `delta_QKV_input` | 0.01259 | 4.58e-4 |
| `delta_QKV_output` | 0.01733 | 3.97e-4 |

The perturbation survives LayerNorm and QKV, so the current evidence does not support “LayerNorm completely cancels TAFH.” It is already very small before that boundary. The current audit does not yet measure `delta_A` or final MHSA output delta; those fields are explicitly recorded as `NA`, not inferred.

## Code Findings

- `TAFH_FORCE_GAMMA_ZERO=1` is an inference-only feature gate that zeros gamma at `_receive()` without changing the checkpoint.
- TAFH is enabled only at TBSI location 9 and runs before template concatenation, LayerNorm, and QKV.
- The source mask is applied after softmax in `_address()` and is not renormalized afterward. This makes effective message mass depend on mask occupancy; it is a candidate mechanism to test, not yet a proven cause.
- Training and inference use the template box-derived `[B,64]` token mask. RGB and TIR currently receive the same mask derived from the visible template annotation.

## Next Controlled Audit

Run the same TAFH epoch-15 checkpoint with `TAFH_FORCE_GAMMA_ZERO=1` on all 244 sequences. Compare its per-sequence AUC with both M1 and normal TAFH. Interpretation:

- gamma=0 close to M1: the learned TAFH residual is the likely source of harm;
- gamma=0 still below M1: training trajectory or another contract difference is implicated;
- gamma=0 improves the large-loss sequences: the current exchange is conditionally harmful and should be fixed at selection/normalization, not amplified.

No new training should start before this audit is complete. The result remains a diagnostic, not a method claim.

Evidence:

- `output/experiments/tafh_mechanism_audit_20261002/m1_tafh_per_sequence.csv`
- `output/experiments/tafh_mechanism_audit_20261002/injection_transfer_dev10x8.csv`
- `tools/run_tafh_mechanism_audit_serial.sh`
