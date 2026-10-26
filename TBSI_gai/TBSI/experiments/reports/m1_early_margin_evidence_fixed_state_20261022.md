# Early target/local hard-background evidence fixed-state audit (2026-10-22)

This diagnostic revisits the existing early margin branch. It fixes template, crop, tracker state and event frame, then compares the trained margin checkpoint with the same architecture whose `early_margin_gate.*` tensors are zeroed.

The 25-event replay found RGB/TIR local margin means of `0.65708/0.73422`, but absolute changes in the final MSHA cross-modal Value gates were only `5.85e-6/2.27e-6`. Margin-to-gate correlations were `0.0252/0.0061`. CENTER target-background margin changed by only `2.83e-5`, and CENTER GT rank changed on `0/25` events.

The evidence is therefore present numerically but is not functioning as a control signal. The current learned MLP and scale path is effectively inactive and uncalibrated: a large margin does not produce a proportional gate response. The margin also describes local uniqueness around a token's maximum, which is not guaranteed to be the GT target, so it can confidently describe a distractor.

This rejects the current learned margin router and its training contract. It does not reject the relative evidence statistic itself. The next experiment should remove the MLP and use a fixed-temperature, monotonic, bounded scalar directly on the MSHA cross-modal Value gate:

\[
g_m=\exp(\rho_{max}\tanh(E_m/T)),\qquad V'_{m\to n}=g_mV_{m\to n}.
\]

Run only `rho=0` and one small nonzero value in the same fixed-state replay. Accept the operator only if gate changes are monotonic with evidence and GT rank improves on a meaningful subset without broad non-target changes. Do not add epochs, widen the MLP, or add losses before this test.

Artifacts: `output/experiments/M1_EARLY_MARGIN_FIXED_STATE_EVIDENCE_20261022/retry/evidence_summary.json`, `evidence_per_location.csv`, `fixed_state_paired.csv`; analysis: `tools/analyze_m1_early_margin_evidence_20261022.py`.
