# M1 Spatial Token Bridge Output-Amplitude Diagnostic

## Question and hypotheses

Question: did the prior fixed-state null result arise mainly because a directionally useful learned bridge message was too small, or because the learned message direction/content is not useful to localization under the current insertion point?

- H0 (amplitude-limited): increasing only the post-bridge message scale produces a dose response, with target correspondence outperforming an equal-density shifted correspondence on localization readouts.
- H1 (content/interface-limited): scaling the learned message through a large range does not produce a target-specific localization benefit; target and shifted remain tied or both respond similarly.
- H2 (overdrive): large scaling changes localization but target does not outperform shifted, or both arms regress; magnitude alone is not useful and may destabilize the readout.

## Parent and single intervention

- Parent: `M1_SPATIAL_TOKEN_BRIDGE_LEARNABILITY_10EP_20261024`, epoch 10 checkpoint.
- Same 23 LasHeR train-side calibration sequences and 46 fixed states as the E10 off/target/shifted replay. Events are reconstructed from the saved E10 off `per_event.csv`, preserving frame, state, crop, GT, and M1 trace IoU. Event JSONL and sequence list are persisted and hashed before replay.
- Same explicit bridge-enabled inference config and FP32 serial evaluation.
- One variable: multiply the bridge's already-computed output delta immediately after `SpatialTokenBridge.forward`; do not change checkpoint, selected indices, correspondence, parameters, clipping internal to the bridge, backbone, or tracker state.
- Arms: off; target and shifted at scale `1`, `32`, and `128`. Scale 1 reproduces the native learned residual. Scale 128 is a diagnostic stress dose, not a proposed deployment setting.
- Output hook is confined to the fixed-state runner. It leaves model files and checkpoint untouched. The pre-change source bundle is preserved at `experiments/snapshots/M1_SPATIAL_TOKEN_BRIDGE_AMPLITUDE_20261024/prechange_sources.tar.gz`.

## Measurements and decision rules

Primary measurements: CENTER GT hard-background margin/rank, post-fusion query GT margin/rank, and fixed replay IoU. Analyze paired state deltas, sequence means, and pre-onset/severe-onset separately. Record native bridge delta norms and the explicit applied scale.

Validity check: scale-1 target and shifted arms must reproduce their E10 learnability replay values within floating-point tolerance; off must reproduce the prior M1 replay. A material mismatch invalidates dose-response attribution and requires investigation before interpretation.

Evidence for amplitude limitation requires a target-over-shifted CENTER or query margin greater than `1e-3` at some scale, positive on a majority of sequence means, plus no CENTER-rank or replay-IoU regression. A monotonic target-specific change below those conditions is exploratory only.

If scale 128 produces no target-specific rank/readout benefit, classify “small native magnitude alone is insufficient” and stop amplitude scaling. If both modes change comparably or regress, classify overdrive/non-specific perturbation. No arm produces a benchmark result; no full training follows this diagnostic automatically.

## Provenance

- Parent event table SHA256: recorded in run manifest.
- Candidate config: `experiments/tbsi_track/M1_SPATIAL_TOKEN_BRIDGE_AMPLITUDE_FIXED46_20261024.yaml`.
- Candidate checkpoint: E10 bridge-only checkpoint from the parent diagnostic; SHA256 recorded in run manifest.
- Runner and exact commands: `tools/run_m1_screening_fixed_state_flow_20261022.py`, `tools/run_m1_spatial_token_bridge_amplitude_fixed46_20261024.py`, and generated `output/experiments/M1_SPATIAL_TOKEN_BRIDGE_AMPLITUDE_FIXED46_20261024/commands_and_logs.txt`.
