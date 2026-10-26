# 03 Compatibility and Attribution

## Insertion-point distinction

The repository contains two non-equivalent DBF placements:

| Placement | Inputs and operation | Downstream effect | Evidence/status |
|---|---|---|---|
| In-layer `TBSILayer.dbf` | After TBSI-E template fusion, two directional search-query bridges use cross-modal context inside a transformer layer. | Updated search streams continue through later backbone blocks. | Configurable with `BACKBONE.DBF_ENABLE/DBF_LOC`; not the four-arm post-backbone proxy experiment. |
| Post-backbone `post_fusion` | Reads final RGB/TIR search tokens and both template token sets; each search stream attends over the other modality's template+search context; averages updated search streams. | Replaces the single-stream fusion immediately before the shared head. | Tested by `DBF_IGF_REPLACE_SHORT_*`; this is the present task's direct evidence. |

The post-backbone wrapper's output contract is `[B,Ns,C]`, then converted to `[B,C,16,16]` under the fixed 640-token layout (64 template + 256 search per modality). `forward_fusion_only` asserts the token count and search grid. The short config disables temporal tokens, degradation-aware fusion, and in-layer DBF.

## Compatibility matrix

| Dimension | M1/project contract | Local post-backbone DBF | Assessment |
|---|---|---|---|
| Task and modality | RGB-T single-object tracking | Bidirectional cross-modal search fusion | Semantically adjacent; no task-specific evidence from candidate paper. |
| Feature layout | `[RGB template 64, RGB search 256, TIR template 64, TIR search 256]`, channel 768 | Consumes template/search token tensors and emits one search stream | Shape-compatible under current fixed layout; not proven for dynamic token counts. |
| Insertion level | M1 TBSI-E backbone plus learned post-backbone fusion | Tested at head-side post-backbone slot | Compatible as a replacement route, but distinct from the in-layer DBF route. |
| Legacy fusion | 3x3 Conv-BN-ReLU over concatenated search maps | Mean of residual-updated streams | Head shape matches; function and feature distribution do not. |
| Original paper IGF | Illumination estimate from RGB search frame, weighted fusion, CA/SA | No RGB frame, GIE, CA, or SA | Not an original-IGF implementation or strict interface comparison. |
| Differential Transformer details | Two Q/K pairs, lambda parameterization, headwise normalization | Shared Q, two K, shared V, sigmoid scalar lambda, external residual gamma | Mechanism-inspired adaptation; not a faithful paper reproduction. |
| Initialization/optimizer | M1 checkpoint has trained legacy fusion and head | New `post_fusion.*` parameters are newly initialized | For this short run, new module LR was 10x lower than `tbsi_fuse_search`; important training confound. |

## Result and causal boundary

Existing 60-sequence proxy AUC: Legacy 52.4663, Mean 49.5412, Vanilla 50.5768, Differential 49.0681. Differential minus Vanilla is -1.5087 AUC (25 sequence wins, 34 losses, 1 tie). Differential minus Legacy is -3.3982 AUC. These aggregate and paired counts are in `output/experiments/dbf_igf_replacement_short_eval_20261009/comparison.json`. No bootstrap interval is cited here because its result artifact was not located during this audit.

The result supports: this specific three-epoch, one-seed, template-context post-backbone Differential arm underperformed the matched Vanilla arm on this 60-sequence proxy. It does not support: Differential attention generally fails in RGB-T, the original paper mechanism is invalid, or DBF is inferior to original IGF.

Main confounds and boundaries:

1. Legacy-to-replacement changes the learned 3x3-BN-ReLU operator to mean/bridge output, changing the head input function and distribution.
2. `tbsi_fuse_search` is in the `1e-4` optimizer group while newly initialized `post_fusion.*` is in `1e-5`; the three-epoch schedule applies another 10x decay after epoch 2. This limits adaptation opportunity.
3. Differential-versus-Vanilla is the most informative available contrast because they share the wrapper, insertion point, and broad training contract. It remains single-seed and short-budget, and saved logs do not report per-parameter gradient magnitudes, gamma/lambda trajectories, or residual-to-mean ratios.
4. The wrapper includes template context and differs from both search-only IGF semantics and the original illumination-guided fusion design.

Therefore attribution is incomplete: the local Differential operation is a plausible contributor to the negative D-V observation, but evidence cannot separate its effect from short adaptation, initialization, or sampling uncertainty.

## Focused failure diagnosis for the next iteration

| Level | Observable evidence today | Distinguishing measurement | One possible intervention if confirmed |
|---|---|---|---|
| L1, learning/update failure | New post-fusion parameters had `1e-5` LR; no saved per-parameter gradient/update trace. | FP32 gradient/update norm for q/k1/k2/v/out/gamma/lambda and effective residual over steps. | Correct only the new-module LR, then rerun matched controls. |
| L2, representation or attention cancellation | D below V; local subtraction has no headwise output normalization, while paper Table 6 links removing it to instability in language modeling. | Per-head differential output spread, map difference, residual-to-query norm, out-bias contribution. | Add only per-head normalization; a positive D result still needs V+norm for mechanism attribution. |
| L3, position/context mismatch | Current post-backbone DBF reads template+search; paper IGF's directly fused objects are search streams. | If a dedicated L3 diagnosis becomes necessary, compare search-only with template-context under the same readout and budget. The planned search-only G1 alone does not isolate this variable from the old run. | Choose search-only as the primary new contract; reserve a matched context comparison for a justified follow-up. |
| L4, head coordination/distribution | Mean alone trails L by 2.9251 AUC; all replacement arms bypass trained Conv-BN-ReLU. | Paired head-input distribution and gamma-zero M1 parity under a preserved readout. | First screen with the legacy readout retained, then perform a separate standalone-replacement gate. |

These are competing hypotheses, not four established causes. The earlier D-V contrast largely controls the L4 readout change, so its negative sign cannot be explained solely by the L-to-Mean gap. The ICLR paper's normalization result is from language modeling and does not directly identify the local RGB-T failure mechanism.

## Guide screening score (priority only)

| Item | Score / 2 | Basis |
|---|---:|---|
| Mechanism matches a demonstrated M1 bottleneck | 1 | The M1 failure map documents tracking failures, but has not isolated redundant cross-modal attention as their cause. |
| IGF-slot interface compatibility | 1 | Output shape and position fit; the current wrapper still reads templates and the RGB frame is not routed to fusion. |
| Candidate-paper ablation evidence for this task | 1 | The paper supports differential attention in its own language-model setting; RGB-T transfer remains a hypothesis. |
| Compute cost | 1 | Existing local D measured 70.85 FPS versus Legacy 73.73 on the proxy; the new search-only version has no measurement yet. |
| Controlled comparison can be built | 1 | Mean/Vanilla/Additive/Differential controls are feasible, but original IGF and GIE weights are unavailable. |
| **Total** | **5 / 10** | Engineering priority score, not evidence of expected AUC improvement. |
