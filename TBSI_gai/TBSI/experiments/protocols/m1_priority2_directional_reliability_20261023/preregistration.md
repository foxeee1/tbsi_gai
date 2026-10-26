# M1 Priority 2: Direction-Separated Reliability Mechanism

Status: `COMPLETE / MECHANISM-NOT-SUPPORTED / NO-TRAINING`.

## Question

Does each modality's own foreground-template target-versus-local-competitor evidence control only the cross-modal Value message sourced from that modality, and does this directed mapping outperform shared or swapped reliability controls on identical fixed inputs?

## Parent and intervention

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15. YAML SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1`; checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- Reuse one parent forward path and one set of fixed frames, crops, templates, and states. No optimizer updates or checkpoint writes.
- At block02, compute for each modality `e_m(j)=logsumexp_i(cos(phi(t_i),phi(s_j))/tau + log(w_i_fg+eps))-log(sum_i(w_i_fg)+eps)`, with `tau=0.07` and initialized-template fractional foreground coverage.
- Independently let `p_m=argmax_j e_m(j)` and `h_m` be the maximum score outside the 3x3 neighborhood of `p_m`; set `c_m=tanh((e_m(p_m)-h_m)/0.25)` and `r_m=1+0.25*c_m`.
- `independent`: RGB evidence scales RGB->TIR Value; TIR evidence scales TIR->RGB Value.
- `shared`: both directions use `(c_rgb+c_tir)/2`.
- `swapped`: RGB evidence scales TIR->RGB and TIR evidence scales RGB->TIR.
- Self messages, attention Q/K, template-fusion residual, TBSI/search fusion, CENTER, loss, and tracker state/update remain unchanged. Mode defaults off.

## Data and measurements

- Fixed train-side calibration cohort: the existing 23 LasHeR training sequences with one pre-onset and one severe-onset state (46 fixed states), from `experiments/protocols/m1_screening_v1_20261022/calibration_sequences.txt` and frozen event records at `output/experiments/M1_SCREENING_EVENT_MINING_CALIBRATION_20261022/first_persistent_events.jsonl`.
- This cohort has prior exposure and is used only for mechanism diagnosis and capture-contract validation. It is not a fresh evaluation cohort. Do not open development or locked holdout.
- Arms: M1 parent/off, independent, shared, swapped. Record per-modality evidence and reliability, each directed Value-message norm before/after scaling, each MSHA output delta, CENTER GT rank/margin, selected peak, decoded box, and fixed-state IoU.
- Runtime: FP32, one torch thread, serial, 1x RTX 4090D. No training and no benchmark AUC claim.

## Decision criteria

Mechanism pass requires all of the following:

1. Disabled mode reproduces parent score map and box within `1e-4` maximum absolute difference.
2. The independent arm changes the intended source-direction message in at least 90% of states with nonzero `|c_m|`; the opposite direction's gate must be independent of that modality's evidence.
3. Independent, shared, and swapped controls produce distinguishable message/output tensors on at least one-third of states, with finite values and stable tensor shapes.
4. On the 23 fixed states, independent mapping must improve per-modality source-target evidence alignment versus both shared and swapped controls. CENTER rank/IoU are downstream readouts; no tracking gain is claimed.

If the intended message mapping does not pass, classify as interface/measurement failure and stop. If mapping passes but independent reliability does not outperform controls on target/background evidence or downstream readouts, mark Priority 2 `MECHANISM-NOT-SUPPORTED` and do not train it. A pass admits only a later fixed-sample learnability preflight, not a quick/full training run.

## Provenance

- Pre-change source snapshot: `experiments/snapshots/m1_priority2_directional_reliability_20261023/prechange_sources.tar.gz`, SHA256 `718761128ed8edfb0dff3560531da4bb17e8f245ae05b43190a25c89ea9c5fe2`.
- Current source changes are feature-gated and evaluated against the protected parent hashes above.

## Result

- Four-arm capture completed on the preregistered 46 fixed states. The measured Value-message norm ratios match the intended independent/shared/swapped formulas within `1.9e-7`; both directions changed on all 46 states. This validates the local message-scaling implementation contract.
- The evidence signal did not validate as a reliable target cue: RGB margin AUC for whether its candidate peak fell inside GT was `0.608`; TIR was `0.440`. Candidate peaks fell inside GT in 22/46 RGB states and 17/46 TIR states.
- Independent mapping did not outperform controls downstream. Mean fixed-state IoU deltas vs parent were `+0.00151` independent, `+0.00020` shared, and `+0.00033` swapped, with substantial per-state wins/losses. CENTER top1-in-GT remained 33/46 in every arm; average CENTER rank/margin did not improve.
- The captured trace-vs-replay IoU max absolute delta was `4.46e-4`. This is not a direct score-map/box parity measurement, so the preregistered strict disabled-parity criterion remains unverified.
- Decision: `MECHANISM-NOT-SUPPORTED`. Do not train or tune this reliability operator. The result is limited to this exposed train-side calibration cohort and fixed-state replay; it makes no LasHeR/RGBT234 tracking-performance claim.
- Analysis: `experiments/reports/m1_priority2_directional_reliability_fixed46_20261023.md`; outputs: `output/experiments/M1_PRIORITY2_DIRECTIONAL_RELIABILITY_ANALYSIS_20261023/`.
