# M1 action-space comparison protocol (2026-10-13)

## Question

For the same frozen M1 state and the same joint guidance input, is useful compensation available in the local fused feature space or only in the CENTER readout space?

Primary comparison:

1. **Feature arm (F):** bounded local residual on the fused search feature only.
2. **Readout arm (H):** bounded correction to CENTER score logits, offset, and size only.

The experiment estimates fixed-state decoded-box utility. It is not a closed-loop AUC experiment.

## Parent and isolation

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, explicit epoch-15 checkpoint.
- M1 backbone, fusion, template, crop, CENTER head, Hann rule, peak selection, offset/size decode, and tracker state are frozen.
- New experiment name, config, output directory, and asset ID are required. The protected YAML, checkpoint, logs, and full-244 results are read-only.
- Use a new outcome-blind train-side sequence cohort. Sequence membership, manifest, exclusions, and SHA256 must be recorded before fitting. Existing 44-state signal-selector results must not be used for feature, threshold, or architecture selection.
- Split by sequence: fit, calibration, and held-out test. No frame from a held-out sequence may enter fitting or threshold selection.

## Common state and guidance input

For every state, save the exact tensors and metadata once and reuse them for both arms:

- RGB search feature, TIR search feature, M1 fused feature `F0`;
- template features and previous tracker state;
- M1 CENTER score logits, offset map, size map, decoded box;
- crop transform, target-in-crop flag, and sequence/frame identifiers;
- deployment-visible joint signal `q`, including modality evidence, cross-modal disagreement, M1 score entropy/top margin, crop-boundary risk, and motion/scale state.

The two arms must receive identical `q` and the same shared encoder. Differences in input signal, train split, optimizer, or parameter budget invalidate the action-space comparison.

## Layer 0: action-space headroom before learning

Construct offline candidate actions on the fit and held-out states using only the frozen tensors. Every candidate is decoded by the unmodified M1 CENTER decode.

### Feature arm candidates

For modality `m` in `{RGB,TIR}`:

\[
F'_{m}=F_0+\rho G_m(F_m-F_0),
\]

where `G_m` is a fixed 3x3 support around the deployment-defined modality peak and `rho` is pre-registered as `{0.05, 0.10}`. Include `no-op` and both modality directions. Do not tune support size or peak rule on the held-out split.

### Readout arm candidates

Let `z0`, `o0`, and `s0` be M1 score logits, offset, and size maps. Use the same joint signal to form bounded corrections:

\[
z'=z_0+\gamma\Delta z,\qquad
o'=o_0+\gamma\Delta o,\qquad
s'=s_0+\gamma\Delta s.
\]

Evaluate the pre-registered sub-actions `score-only`, `offset-only`, `size-only`, and `joint(score,offset,size)`, plus `no-op`. Corrections are constrained to the same local support and fixed norm budget across states. A correction may use a candidate RGB/TIR peak or local evidence, but may not use GT at deployment. GT is used only afterward to calculate utility.

For both arms, first report the hindsight oracle:

\[
u_i^*=\max_{a\in A} \max(0, IoU_i(a)-IoU_i(0)).
\]

This separates “the action space has no useful headroom” from “the signal cannot select the useful action.”

## Layer 1: matched learned selectors

Train two selectors with the same shared encoder, parameter count, optimizer, sequence split, and early stopping rule.

- F selector outputs `{no-op, RGB residual, TIR residual}` and bounded strength.
- H selector outputs `{no-op, score-only, offset-only, size-only, joint}` and bounded strength.

The selector is trained against fixed-state action utilities relative to no-op. The loss must preserve the zero action and include an abstention/uncertainty term. Do not train either arm with ordinary tracking loss alone: that objective does not teach when an intervention is harmful.

Thresholds are selected once on the calibration sequences using a pre-registered risk rule. The held-out test set remains untouched until the final comparison.

## Primary metrics

Report state-level and sequence-macro results with sequence bootstrap intervals:

- selected utility and utility regret to the arm-specific oracle;
- comparison with no-op, fixed RGB, fixed TIR, and random-action controls;
- harmful intervention rate, with epsilon fixed before testing;
- no-op specificity and intervention precision;
- risk-coverage curve for the highest-confidence 5%, 10%, and 20% states;
- center-cell error, center distance, offset error, size error, and target-out-of-crop rate;
- parameter count and inference latency.

The main comparison is paired on the same states. Mean utility alone is insufficient because a selector can have lower regression error while making worse discrete decisions.

## Decision rules

### Promote readout-side correction

Promote only if H has positive held-out utility, lower harmful-intervention rate than its fixed-action controls, and a reproducible improvement in at least one of center selection, offset, or size error. The gain must exceed the sequence-bootstrap uncertainty and remain positive under the no-op threshold selected on calibration data.

### Promote feature-side local residual

Promote only if F passes the same criteria and its gain remains after accounting for the readout controls. A gain confined to states where H has no oracle headroom does not establish feature superiority.

### Diagnose the bottleneck

- Both arms have near-zero oracle headroom: the tested compensation contract is unsuitable; stop module expansion and inspect crop/state or supervision.
- F oracle headroom is positive, H is not: continue local feature compensation with a new signal/action diagnosis.
- H oracle headroom is positive, F is not: treat readout consistency as the leading bottleneck and stop feature-side expansion.
- Both have headroom but learned selectors fail: the joint signal is insufficient, utility labels are too noisy, or the action parameterization is wrong. Do not compare architecture quality yet.
- H improves only score selection while offset/size remain wrong: use a score/readout-specific follow-up; do not claim full decoding correction.

No closed-loop training or full-244 evaluation is allowed before one arm passes this fixed-state gate.

## Required artifacts

- protocol and pre-registered sequence manifest with hashes;
- frozen-state tensor manifest and checkpoint/config hashes;
- per-state candidate utility table for both arms;
- oracle headroom summary;
- fit/calibration/test predictions and thresholds;
- held-out per-state and sequence-macro metrics;
- provenance, environment, command, and final registry entries for the diagnostic, including `NA` for unavailable AUC/FPS metrics.

## Attribution boundary

This experiment can establish which tested action space has usable fixed-state headroom under the M1 decode contract. It cannot prove that the corresponding component is the global cause of M1 failure, nor can it establish closed-loop recovery without a later matched rollout.
