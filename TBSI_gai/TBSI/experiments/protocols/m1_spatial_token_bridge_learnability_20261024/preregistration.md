# M1 Spatial Token Bridge Learnability Diagnostic

## Objective

Determine why the candidate spatial token bridge produced nonzero but very small residual messages after its 8-step fit smoke. Separate insufficient optimization exposure from a weak localization-facing interface. This is an auxiliary diagnostic, not a full-training or performance run.

## Hypotheses

- H0: Additional bridge-only optimizer steps do not materially increase the learned message or improve target-conditioned fixed-state readouts.
- H1: The 8-step smoke undertrained the bridge; 80 steps increase its message and produce target-over-shifted improvements in localization-facing readouts.
- H2: The bridge learns larger messages, but CENTER rank and fixed replay remain unchanged; the present injection/readout contract is weakly coupled to localization.

## Intervention

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42` epoch15 checkpoint, read-only.
- One trainable component: `SpatialTokenBridge`; all parent parameters frozen.
- Operator: final backbone RGB/TIR search tokens at the existing pre-neck boundary; independent top-8 target-conditioned positions per modality; bidirectional soft token correspondence using feature cosine plus spatial distance; rank-32 zero-initialized residual, clipped to at most 10% source-token norm.
- Data: LasHeR train; 16 samples per epoch; batch 2; 10 epochs; 80 optimizer steps; seed 42; worker 4, prefetch 2, persistent workers false; FP32.
- Checkpoints: epochs 1, 5, and 10. Do not use validation samples as training data.

## Comparators and measurements

After training, evaluate each saved checkpoint at epochs 1, 5, and 10 on the same 23-sequence/46-state train-side calibration manifest with FP32 serial replay. Arms are bridge `off`, target correspondence, and equal-density half-grid shifted correspondence. Record bridge parameter norms, bridge delta norm and relative source norm, correspondence entropy, CENTER GT rank/margin, post-fusion query rank/margin, and fixed replay IoU. Compare target-minus-off and target-minus-shifted by state and sequence.

## Decision rules

- Contract failure, non-finite loss/gradient, or no parameter update: stop as an implementation/training-contract failure.
- Message remains near the prior `~0.001` token norm after 80 steps and target/shifted readouts remain indistinguishable: current training objective does not drive the branch under this contract; stop this operator.
- Message grows but target/shifted margin differences remain below `1e-3`, with rank/replay unchanged: classify as interface/readout coupling not supported; stop this operator.
- Only if target-over-shifted CENTER or query margin exceeds `1e-3`, improves across a majority of sequence means, and neither CENTER rank nor replay IoU regresses, design a separate M1-matched full-training experiment. These calibration states remain train-side screening only.

## Provenance

- Parent YAML SHA256: `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`.
- Parent checkpoint SHA256: `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- Earlier 1-epoch screen and fixed-state results: `experiments/reports/m1_spatial_token_bridge_fixed46_20261023.md`.
- New config: `experiments/tbsi_track/M1_SPATIAL_TOKEN_BRIDGE_LEARNABILITY_10EP_20261024.yaml`.
- No research code is changed for this diagnostic; implementation snapshot remains `experiments/snapshots/M1_SPATIAL_TOKEN_BRIDGE_20261023/postfix/spatial_token_bridge_sources.tar.gz`.
