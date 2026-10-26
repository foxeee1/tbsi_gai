# M1 Spatial Token Bridge Mechanism Screen

## Question

Does explicit, sparse RGB-to-TIR/TIR-to-RGB correspondence among target-conditioned search tokens change the fused representation and reach CENTER more than an equal-density spatially shifted correspondence control?

## Hypotheses

- Null: target-conditioned bridge does not improve the localization-facing readouts over bridge-off and shifted correspondence.
- Alternative: target-conditioned correspondence improves post-fusion target/background separation and at least one discrete localization readout (CENTER GT rank or fixed replay IoU), while the shifted control does not reproduce the effect.

## Operator

At the existing search-neck boundary, late RGB/TIR search tokens have shape `[B,256,768]`, with a 16x16 grid. Each modality scores its own search tokens against its masked template foreground prototype and selects top-8 positions. For every selected RGB location, a soft correspondence distribution over the selected TIR locations is computed from feature cosine similarity plus a spatial-distance penalty; the reverse direction is computed independently. A rank-32 residual MLP receives source, aligned peer, their difference, source/peer template similarity, and correspondence entropy. Its output is zero-initialized and norm-limited to 10% of the source token norm, then added only at selected positions. The original `tbsi_fuse_search` neck and CENTER head remain downstream.

The shifted control rolls both modalities' selected indices by eight cells in each grid axis before correspondence and message construction. It preserves candidate count and all learned weights while moving the bridge support away from the target-conditioned positions.

## Runs

1. Contract smoke: disabled-path parity, zero-init identity, shape/device/dtype, finite outputs, nonzero finite `up` gradients, optimizer update, state-dict save/load.
2. Fit smoke: one epoch, 16 LasHeR train samples, batch 2, worker 0, only bridge parameters trainable, initialized from protected M1 epoch15. This is implementation/optimization screening only.
3. Fixed-state comparison: same trained checkpoint, 23 LasHeR train calibration sequences, 46 pre/onset states, FP32 serial; arms `off`, `target`, `shifted`. Record event manifest, per-event CENTER rank/margins, query rank/margins, replay IoU, bridge delta norms, correspondence entropy, and candidate fraction.

## Decision gate

Stop before full training if any contract check fails, intended parameters do not update, target and shifted arms cannot be distinguished numerically, or target-conditioned changes do not reach a localization-facing readout. A positive mechanism result requires target-vs-shifted improvement in target/background margin across states and no worse CENTER rank/replay IoU; only then may a separate M1-aligned 15-epoch full-training configuration be prepared. The fixed-state cohort is train-side calibration and cannot support a generalization or full-test claim.

## Provenance

- Parent config SHA256: `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`.
- Parent checkpoint SHA256: `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- Pre-change source snapshot: `experiments/snapshots/M1_SPATIAL_TOKEN_BRIDGE_20261023/prechange_sources.tar.gz`.
- Candidate screen config: `experiments/tbsi_track/M1_SPATIAL_TOKEN_BRIDGE_SCREEN_20261023.yaml`.
- Fixed event list: `output/experiments/M1_SCREENING_EVENT_MINING_CALIBRATION_20261022/first_persistent_events.jsonl`.
- Fixed sequence list: `experiments/protocols/m1_screening_v1_20261022/calibration_sequences.txt`.

## Execution outcome (2026-10-23)

- Contract checks passed: disabled parity, zero-init identity, shapes, gradients, optimizer update, and state-dict round-trip. Fit smoke completed one epoch over 16 LasHeR training samples with 8 optimizer steps; this is not a performance run.
- Fixed-state `off` and `target` completed 46/46 states. `shifted` first failed before producing event metrics because its eight top-k indices were incorrectly reshaped as a 16x16 grid. The control was fixed to translate each selected 2D grid coordinate by half-grid using modular indexing; a coordinate/uniqueness smoke passed, then `shifted` retry completed 46/46.
- The fixed-state three-arm comparison did not pass the mechanism gate. CENTER and post-fusion query GT rank were unchanged in all 46 states. Target-vs-shifted mean CENTER margin difference was `+1.07e-6` (26 wins / 20 losses); post-fusion query margin difference was `+3.99e-7` (20 wins / 20 losses / 6 ties); fixed replay IoU difference was `-2.62e-6` (20 wins / 22 losses / 4 ties). These are negligible and inconsistent changes.
- Decision: `暂不支持` the current operator as a localization-improving mechanism; do not proceed to full training with this operator. A follow-up, if pursued, must be a separately preregistered learnability diagnostic explaining why the 8-step bridge fit produces only small messages.
- Detailed paired results, exact provenance, and incident: `experiments/reports/m1_spatial_token_bridge_fixed46_20261023.md`.
- Completed outputs: `output/experiments/M1_SPATIAL_TOKEN_BRIDGE_OFF_FIXED46_20261023/`, `output/experiments/M1_SPATIAL_TOKEN_BRIDGE_TARGET_FIXED46_20261023/`, and `output/experiments/M1_SPATIAL_TOKEN_BRIDGE_SHIFTED_FIXED46_RETRY1_20261023/`. The first failed shifted output directory is retained as an empty directory; its traceback was observed in the execution session but was not exported to a log file.
- Post-fix source bundle SHA256: `3ccce72120ac9bef00828f6b3cc5224304d543e581fe37b4375d22300af70051`.
