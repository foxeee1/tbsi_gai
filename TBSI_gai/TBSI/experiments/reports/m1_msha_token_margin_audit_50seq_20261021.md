# M1 token-level target/background margin audit (2026-10-21)

## Purpose

This audit tests whether spatially resolved, target-conditioned signals can
predict the fixed-state CENTER target/hard-background separation. It reuses the
same 50-sequence manifest and epoch15 M1 checkpoint as the uncertainty-gate
diagnostic. No model parameters, loss, dataset split, or tracker state update
rule were changed.

## Protocol

- Manifest: `experiments/manifests/m1_msha_uncertainty_gate_50seq_20261021.txt`
  (SHA256 `a76af65939e9940aa491cb5cd1e2a645cfcb7c2d9ca0a99309656c894940e795`).
- 50 requested sequences, 26 fixed-state events, 24 sequences without an event
  under the frozen rule.
- Same template, crop, state, frame and off-arm hard-background cell.
- For loc3/6/9, record per-token template foreground match, RGB/TIR agreement,
  and target-minus-background means. Also record feature separation around
  `tbsi_fuse_search` when the tensor contract exposes a 16x16 token grid.

## Results

| signal | mean | positive events | correlation with CENTER margin |
|---|---:|---:|---:|
| RGB template match margin loc3 | 0.1453 | 26/26 | -0.257 |
| RGB template match margin loc6 | 0.1406 | 26/26 | -0.106 |
| RGB template match margin loc9 | 0.0816 | 25/26 | -0.088 |
| TIR template match margin loc3 | 0.1210 | 26/26 | -0.257 |
| TIR template match margin loc6 | 0.1099 | 26/26 | -0.228 |
| TIR template match margin loc9 | 0.0565 | 24/26 | -0.260 |
| RGB/TIR agreement margin loc3 | -0.0769 | 3/26 | 0.195 |
| RGB/TIR agreement margin loc6 | -0.0678 | 3/26 | 0.127 |
| RGB/TIR agreement margin loc9 | -0.0876 | 2/26 | 0.002 |
| post-fusion feature target/background margin | 0.0744 | 26/26 | -0.091 |

Template matching remains positive in most target regions, but its absolute
value does not track the final CENTER decision. The signal decays from loc3 to
loc9. Cross-modal agreement is consistently background-favoured (negative
target-minus-background margin), with no useful relationship at loc9. The
post-fusion feature margin is weakly related to the final peak margin.

## Decision

`DIAGNOSTIC-COMPLETE / CURRENT-UNCERTAINTY-SUPERVISION-NOT-ADMITTED`.

The result rejects a loss that directly supervises the existing global
reliability or agreement estimator. It does not reject all target-conditioned
supervision. A future uncertainty head must consume spatial target-conditioned
features and be trained against a signed ranking target, with a separate train/
dev split and an explicit check that its output changes the GT/background peak
ordering. The next architectural audit should inspect the first CENTER ranking
error and the target-conditioned feature before and after the actual search
fusion, rather than increase gate strength or add another generic module.

## Evidence

- Script: `tools/run_m1_token_margin_audit_20261021.py`.
- Output: `output/experiments/M1_MSHA_TOKEN_MARGIN_AUDIT_50SEQ_20261021_RETRY3/{per_event.csv,summary.json,sequence_list.txt}`.
- Protected config SHA256: `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`.
- Protected checkpoint SHA256: `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.

