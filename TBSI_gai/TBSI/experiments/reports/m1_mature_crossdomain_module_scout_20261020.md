# Mature Cross-domain Module Scout for AMCM Replacement

## Decision target

The previous `orthogonal_uncertainty` implementation was a hand-built router:
it used search-only statistics, did not consume template tokens, and produced a
very small additive residual. The next candidate must be an independently
published, publicly implemented operator whose internal computation can be
audited and whose output can preserve the AMCM contract:

```text
search_rgb/search_tir: [B,C,16,16]
template_rgb/template_tir: [B,64,C]
  -> two corrected search streams with the same shapes
  -> existing tbsi_fuse_search
```

No backbone, TBSI, CENTER head, tracker state, or external loss is part of this
scout.

## Audited candidates

### 1. NLformer Bayesian Self-Attention with Intra-Token Modeling (ECCV 2026)

Official implementation: `github.com/Ryann2001/NLformer_Lightweight_Denoising`.
The repository has a complete training/evaluation pipeline, pretrained models,
an MIT-style license, and a directly inspectable `Attention_block`.

The block has three explicit branches:

1. First-order local window attention over overlapping neighborhoods.
2. Second-order channel covariance attention (`qc @ qc^T`) for intra-token
   channel relationships.
3. A low-rank Woodbury complementary branch.

The branches are concatenated and projected back to the original channel
count. It is a mature operator rather than a manually designed scalar gate.
Its local branch addresses crop-local background peaks; its covariance branch
is a defensible mechanism for measuring channel/modality agreement without
assuming that a global mean is reliable.

**AMCM mapping:** apply the same NLformer attention weights to a packed dual
stream representation, then split the projected output into RGB/TIR streams.
Template tokens should enter only as a conditioning bias for the query/key
normalization or as a separate template summary token; this is the minimum
adapter needed to make the operator target-conditioned. The core attention,
covariance, and Woodbury computations remain unchanged.

**Compatibility:** 16x16 search maps satisfy the official window partition
requirements for a window of 8. The output projection can return `2C`, which
splits exactly into two `C`-channel streams. The template branch can remain
token-shaped and need not alter the external contract.

**Risks:** it is heavier than AMCM-lite; the Woodbury branch may be unnecessary
for 16x16 maps; the template-conditioning adapter is still a task adaptation
and must be isolated as one variable.

### 2. H-WeightFormer dynamic weight generation (2026)

Official implementation: `github.com/LeapLabTHU/WeightFormer`.
The audited `DynCnn` predicts a per-sample depthwise 3x3 kernel from pooled
features, normalizes the generated kernel, and adds it to a static kernel. The
`DynLinear` branch uses low-rank generated weights for token mixing.

This is a mature input-conditioned operator and would be a better replacement
than a manually predicted residual if the hypothesis were “the local operator
itself must change by scene.” However, the official mechanism does not model
uncertainty, target correspondence, or two-stream complementarity. Supplying
template context to its generator would be a substantial new wrapper, and the
result would still need a separate RGB/TIR routing rule.

**Decision:** retain as a second-stage fallback only. It cannot be the first
clean test of modality usefulness versus uselessness.

### 3. SDANet DSCA dynamic sparse channel attention (2026)

Official implementation: `github.com/oucailab/SDANet`.
The `DSCA` block predicts a sparsity factor, uses a hard top-k mask with a
straight-through estimator, and combines global sparse attention with a local
depthwise branch. The repository also couples the block to a frequency-guided
branch.

The implementation is mature and technically reproducible, but hard top-k,
STE behavior, and the frequency path introduce three confounders unrelated to
the current target/background diagnosis. The project rules also currently keep
the frequency line paused.

**Decision:** do not use for the first AMCM replacement.

## Final recommendation

Use **NLformer `Attention_block` as the only first candidate**. It is the only
audited module whose native decomposition matches both unresolved mechanisms:

- local spatial correspondence for suppressing crop-internal distractor peaks;
- second-order channel statistics for distinguishing complementary and
  redundant modality content.

The first implementation should preserve the official three-branch operator
and add only a minimal template-conditioning adapter at the AMCM boundary. It
must not copy the full NLformer encoder/decoder or its denoising loss.

The acceptance test remains the existing fixed ten-event cohort, but before
training the module must pass:

1. exact zero/identity parity when its residual scale is disabled;
2. shape and save/load parity for both streams;
3. nonzero gradients through local, covariance, and complementary branches;
4. spatially varying output changes, with template-conditioned diagnostics;
5. target/background residual and peak-margin measurements before any larger
   training run.

If the NLformer core passes the contract but its ten-event result remains
unstable, the evidence will distinguish operator capacity from the remaining
supervision/state problem. WeightFormer should then be tested once as a
dynamic-local-operator ablation. SDANet and frequency-enhanced modules should
remain out of this line.

## Provenance

- NLformer README and `model/nlformer_arch.py`, official repository commit
  `e20911303c1529218fc55c607ca19b9e338ea44f`.
- WeightFormer `model/h_wfm.py`, official repository commit
  `95d5359ec69bdca0e8b4267e30af9e06fca36037`.
- SDANet `basicsr/archs/SDAnet_arch.py`, official repository commit
  `8527d9da191d29b11b6e8737bd9f00f5c412de0a`.
