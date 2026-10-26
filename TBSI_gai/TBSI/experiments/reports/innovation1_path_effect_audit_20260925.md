# Innovation-1 Path-Effect Code Audit

Date: 2026-09-25  
Config: `v1.0.3-单卡4090D-24G-等效全局128-seed3407`  
Checkpoint: `TBSITrack_ep0015.pth.tar`  
Scope: 8 fixed RPP-v1 train sequences, 3 points per sequence, 24 audit points  
Layer: TBSI index 2, configured location 9  
Inference: FP32, seed 3407

## Audit results

- Checkpoint key audit: 0 missing, 0 unexpected.
- TBSI layers: 3, configured locations `[3, 6, 9]`.
- Selected `ca_t2s_f2v` and `ca_t2s_f2i` hooks: each called 13,549 times in the full audit.
- Hook input shape: `[1, 320, 768]`; the attention message after the 64 template tokens had non-zero norm.
- Unset gate versus explicit `F`/scale 1: maximum box difference `0`, all saved states equal.
- Manual IoU checks: identical box `1.0`, half overlap `1/3`, disjoint `0.0`.
- After adding CUDA RNG to the state snapshot, the order probe `F-GR-GT-F` versus `F-GT-GR-F` produced identical outputs and state hashes for each action. The earlier apparent order difference was an audit-harness omission, not model-state contamination.

## Local response

Across 24 points and scales `{0.75, 0.5, 0}`:

- GR mean short-horizon delta: approximately `-0.00204`, `-0.00154`, `-0.00275`.
- GT mean short-horizon delta: approximately `-0.000149`, `-0.000074`, `-0.00253`.

The intervention changes the selected attention message and reaches the tracker output, but the measured IoU effect is weak and not a stable positive gain. This is a diagnostic result, not a full tracking result.

Evidence: `output/experiments/innovation1_path_effect_audit/audit_results.json`; the one-point CUDA-RNG probe is in `output/experiments/innovation1_path_effect_audit_probe/audit_results.json`.

## Decision

The code audit passes. No checkpoint-loading, hook, token-shape, state-reset, or IoU-scoring defect was found. Retain the original stop decision: do not train a path selector or rerun the 288-point branch experiment.
