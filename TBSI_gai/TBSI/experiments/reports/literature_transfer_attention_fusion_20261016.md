# Cross-domain operator review for TBSI fusion (2026-10-16)

## Current problem to solve

M1 failure analysis shows that the useful headroom is mainly in spatial center selection, while submitting a corrected box to the next crop causes large state feedback damage. The recent target-conditioned pooled residual trained and updated correctly but reduced difficult5 AUC by `0.697`. A transferable operator therefore needs spatial target correspondence and a reliable no-op/identity path.

## Paper and code audit

### 1. Spectral Dynamic Attention Network (SDANet)

- Paper: *Spectral Dynamic Attention Network for Hyperspectral Image Super-Resolution*, IEEE GRSL 2026, DOI `10.1109/LGRS.2026.3689855`.
- Official code: `https://github.com/oucailab/SDANet`.
- Core file: `basicsr/archs/SDAnet_arch.py`, class `DSCA`.
- Code mechanism: predicts a per-input sparsity factor from depthwise features, constructs a hard top-k plus straight-through soft mask over channel attention, and combines a global branch with a local depthwise branch. The same file also contains FFT amplitude gating and frequency-domain feed-forward blocks.
- Transfer judgment: **medium**. The dynamic sparse routing can be adapted to decide where RGB/TIR evidence is trustworthy. The FFT branch is not a good first transfer because it has no target correspondence and the repository frequency route is currently paused.

### 2. Bayesian Self-Attention with Intra-Token Modeling (NLformer)

- Paper: *Bayesian Self-Attention with Intra-Token Modeling for Lightweight Denoising Transformers*, ECCV 2026, DOI `10.1007/978-3-032-37356-4_33`.
- Official code linked by the paper: `https://github.com/Ryann2001/NLformer_Lightweight_Denoising`.
- Core file: `model/nlformer_arch.py`, class `Attention_block`.
- Code mechanism: first-order local inter-patch attention, second-order intra-patch channel covariance, and a low-rank Woodbury branch are concatenated and projected. It explicitly separates local spatial correspondence from channel statistics.
- Transfer judgment: **highest for diagnosis and a lightweight fusion gate**. Apply the second-order channel covariance to RGB/TIR search features to estimate modality agreement/reliability, and use it to gate a local cross-modal correction. Do not copy the full denoising block or its window/padding assumptions.

### 3. Frequency domain-enhanced spectral-spatial fusion transformer

- Paper: *Frequency domain-enhanced spectral-spatial fusion transformer for semantic segmentation of remote sensing images*, Information Fusion 2026, DOI `10.1016/j.inffus.2026.104248`.
- Authors: Xin Li et al.; publisher metadata identifies the paper, but no official public repository or inspectable source was found in this review.
- Code judgment: **unverified**. The paper-level idea is low/high-frequency decomposition and frequency-guided spectral-spatial fusion. That can improve edges and texture, but does not supply target-to-search correspondence or a center-selection objective.
- Transfer judgment: **low priority / currently excluded**. Reopening the frequency line would be a separate research decision and would not address the strongest observed failure chain directly.

### 4. Linear-Time Global Visual Modeling without Explicit Attention (WeightFormer)

- Paper: arXiv `2605.01711`, DOI not yet listed in the retrieved metadata.
- Official code: `https://github.com/LeapLabTHU/WeightFormer`.
- Core files: `model/wfm.py` (`DynLinear`, `DynCnn`) and `model/h_wfm.py`.
- Code mechanism: predicts a low-rank residual to linear or depthwise-convolution weights from pooled feature context; the dynamic parameters then act on every token without explicit pairwise attention. The predicted residual is normalized before adding to a static weight.
- Transfer judgment: **high as an operator design, with one required change**. Global pooled context alone already failed in the TBSI proxy. Use a small spatial pooled context or local target-search context to generate a bounded dynamic depthwise kernel/1x1 projection for the fused search map. Keep the static M1 path and initialize the dynamic residual at zero.

## Ranking for TBSI

1. **NLformer-inspired second-order reliability gate + local cross-modal correction**: best match to the observed problem; separates spatial correspondence from modality reliability and can be inserted after TBSI fusion before CENTER.
2. **WeightFormer-inspired target-conditioned dynamic local operator**: promising for a stronger correction because the target signal changes the local operator itself, rather than adding a broadcast residual. Use spatial context, not one global pooled vector.
3. **SDANet DSCA local/global dynamic sparse routing**: useful as a compact implementation reference for hard/soft routing, but requires target conditioning and should omit the frequency branch.
4. **Frequency-domain spectral-spatial fusion**: defer; code is unverified and the task evidence does not show a frequency bottleneck.

## Minimal experiments implied by the review

### A. NLformer-style reliability gate

Input: frozen M1 RGB/TIR search feature maps at 16x16. Compute per-cell channel covariance/agreement and a local RGB-to-TIR cross-attention correction. Train only a bounded zero-init residual and a scalar gate. Add a center-cell auxiliary loss using the existing Gaussian target map. Evaluate the same difficult5 first.

Expected signature: center-cell top-1 recall rises on severe-onset states while normal sequences remain near M1. If recall rises but tracking does not, the state/crop contract remains the bottleneck.

### B. WeightFormer-style dynamic local operator

Input: fused search map plus template/search local context. Generate a low-rank residual to a depthwise 3x3 kernel per sample or per 4x4 spatial region, normalize it, and add it to the static operator. Train only this adapter with the existing tracking losses plus the center-cell auxiliary term.

Expected signature: the output score map changes spatially and improves the selected center without relying on a global broadcast correction. If it only improves loss but not center recall, the supervision remains insufficient.

## Decision

Do not port any paper wholesale. The next single-variable implementation should be **A**, because it directly tests whether second-order cross-modal reliability is the missing signal. If A fails on difficult5, test B once; if both fail with valid gradients and center diagnostics, the learned fusion route is likely limited by crop/recovery rather than operator capacity.
