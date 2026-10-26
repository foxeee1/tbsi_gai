# 02 Candidate B Evidence: Differential Transformer and Local DBF

## Candidate identity and paper mechanism

The candidate paper is *Differential Transformer* (Ye et al.; [ICLR 2025 Oral, OpenReview forum `OvoCm1gGhN`](https://openreview.net/forum?id=OvoCm1gGhN)). The official conference PDF, arXiv PDF, and arXiv HTML are saved under `papers/`; hashes and source URLs are in `papers/README.md`. Section 2.1, Eqs. (1)-(3), Fig. 2, Section 3.8/Table 6, and Appendix A were read from the [official arXiv v1 HTML](https://arxiv.org/html/2410.05258v1). The conference PDF was retrieved but could not be text-extracted in this environment, so its line-by-line identity with arXiv v1 remains unverified. The central operation subtracts a scaled second softmax attention map from a first map before applying shared values:

```text
(softmax(Q1 K1^T / sqrt(d)) - lambda * softmax(Q2 K2^T / sqrt(d))) V
```

The paper uses separate `Q1/Q2` and `K1/K2`, shared `V`, and `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`. Eq. (3) normalizes each head's differential output, multiplies it by `(1 - lambda_init)`, then concatenates heads and applies the output projection. Fig. 2 calls this GroupNorm to denote per-head normalization. In the paper's 1.4B language-model ablation (Section 3.8, Table 6), the default Diff Transformer has validation loss `3.062`; removing headwise GroupNorm yields `3.122`, which the authors attribute to training instability. This is evidence of importance **in that language-model setting**, not proof of a tracking gain. Appendix A discusses its gradient-scale role. The paper's primary evidence is language modeling and related LLM tasks, not RGB-T tracking. A future paper-aligned operator variant needs a new experiment identity.

## Local implementation facts

| Claim | Type | Evidence | Boundary |
|---|---|---|---|
| `DifferentialBridge` uses one query projection, two key projections, one value projection, and computes the difference of two softmax maps with `sigmoid(lambda_raw)`. | Code confirmed | `lib/models/layers/differential_bridge.py:10-70`. | Not a faithful reproduction of the paper's full parameterization or normalization. |
| The bridge normalizes query/context **inputs** with LayerNorm, but does not normalize each head's differential **output** or apply the paper's fixed scale. | Code confirmed | `lib/models/layers/differential_bridge.py:23-26,36-61`; compare paper Section 2.1/Appendix A. | This omission may affect stability; no local causal diagnosis yet. |
| The bridge is residual: `query + gamma * delta`. | Code confirmed | Same file, `forward`. | Gamma initialization affects early gradient flow and must be part of each run contract. |
| The post-backbone wrapper runs RGB-query-to-TIR-context and TIR-query-to-RGB-context, then averages the updated streams. | Code confirmed | `lib/models/layers/dbf_igf_replacement.py:10-41`. | Both contexts concatenate template and search tokens. No RGB frame/GIE or CA/SA is implemented. |
| A separate DBF slot exists inside `TBSILayer`. | Code confirmed | `lib/models/layers/tbsi_layer.py:415-428` and subsequent forward path. | This is an in-backbone intervention, not the tested post-backbone replacement. |

## Transfer hypothesis

`H1 (agent hypothesis)`: subtracting a second cross-modal attention distribution may suppress redundant or conflicting RGB/TIR context at the selected fusion slot. It may also remove useful agreement, and the current evidence has not measured whether the learned difference term contributes usefully. The hypothesis requires a comparison where the fusion operator, head-input distribution, initialization, and optimizer opportunity are controlled.
