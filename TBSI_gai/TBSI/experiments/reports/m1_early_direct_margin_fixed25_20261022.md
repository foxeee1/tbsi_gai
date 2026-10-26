# Direct scalar early margin coupling fixed-state audit (2026-10-22)

This experiment tested the smallest proposed correction after the previous audit: remove the learned MLP from the control path and apply the early RGB/TIR local margin directly to the MSHA cross-modal Value gate with a fixed temperature and bounded scalar.

\[
g_m=\exp(\rho\tanh(E_m/T)),\qquad V'_{m\rightarrow n}=g_mV_{m\rightarrow n}.
\]

The candidate used `rho=0.05`, `T=0.25`; the control used `rho=0`. Both used the same architecture and the same checkpoint with learned `early_margin_gate.*` tensors zeroed. No training was run. The same 25 fixed template/crop/state/frame events were replayed in FP32.

Results across 75 TBSI locations:

| quantity | result |
|---|---:|
| RGB margin mean | 0.65708 |
| TIR margin mean | 0.73422 |
| absolute RGB gate change | 0.001091 |
| absolute TIR gate change | 0.001250 |
| RGB margin to gate-change correlation | -0.0349 |
| TIR margin to gate-change correlation | 0.2010 |
| CENTER target-background margin delta | +1.29e-5 |
| CENTER GT-rank delta | 0.0 |
| GT-rank improved/worsened events | 0 / 0 |

The direct scalar path fixes the previous amplitude problem: its gate effect is about two orders of magnitude larger than the learned MLP path. It still does not improve CENTER target ranking. The low evidence-to-action correlation and unchanged GT rank show that the local peak-versus-local-competitor margin is not a reliable target identity signal in this fixed-state cohort. Increasing `rho` would change the response magnitude, but would not repair this missing target specificity and is not justified by this result.

Decision: `DIAGNOSTIC-COMPLETE / DIRECT MARGIN ALONE NOT-SUPPORTED`. Stop rho sweeps and training on this evidence. The next useful design must change the evidence source, using an explicit foreground-conditioned template match and a local hard-background comparison tied to the same candidate location. The gate action can remain this simple direct bounded scalar once that evidence passes fixed-state GT-rank validation.

Artifacts: `output/experiments/M1_EARLY_DIRECT_MARGIN_FIXED25_20261022/evidence_summary.json`, `evidence_per_location.csv`, `fixed_state_paired.csv`; configs `M1_EARLY_DIRECT_MARGIN_FIXED25_20261022.yaml` and `M1_EARLY_DIRECT_MARGIN_FIXED25_RHO0_20261022.yaml`.
