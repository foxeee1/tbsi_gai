# Sparse Candidate-Conditioned Neck Fusion: Shifted Control (2026-10-22)

## Question and frozen comparison

Does sparse cross-modal search fusion help because its block02 RGB/TIR foreground candidates cover the target, rather than because any mask of the same density changes the fusion map?

The comparison was frozen at top-8 responses per modality, 3x3 dilation, RGB/TIR candidate union, and `tbsi_fuse_search`. Candidate cells use the mixed fusion output; other cells use the mean of RGB-only and TIR-only outputs. The control circularly shifts the resulting union mask by 8 cells in both grid axes. No parameter was tuned from the outcomes. All three arms used the same 23 calibration events and 46 fixed states, with identical template, crop, state, M1 config and epoch-15 checkpoint.

## Provenance and execution

- Parent: `M1_SPARSE_PARENT_REPLAY1_CALIB46_20261022/`
- Target-conditioned candidate: `M1_SPARSE_NECK_UNION_REPLAY1_CALIB46_20261022/`
- Equal-density shifted control: `M1_SPARSE_NECK_SHIFTED_CONTROL_REPLAY2_CALIB46_20261022/`
- Event source: `output/experiments/M1_SCREENING_EVENT_MINING_CALIBRATION_20261022/first_persistent_events.jsonl`; 23 events with pre-onset and onset frames.
- Sequence list: `experiments/protocols/m1_screening_v1_20261022/calibration_sequences.txt`.
- Runner: `tools/run_m1_screening_fixed_state_flow_20261022.py`; source snapshot: `experiments/snapshots/m1_sparse_shifted_control_runner_20261022/final_sources_v2.tar.gz`, SHA256 `43c31f6583e7572ac348c32ffc5004b77791c898b841a96cf292c13faf651e99`.
- Protected config SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`; checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- Environment: 1x NVIDIA GeForce RTX 4090 D; Python 3.8.20, PyTorch 1.9.0+cu111, CUDA 11.1; FP32, one CPU thread. The three-arm fixed-state replay took about 30 seconds of wall time (0.008 GPU-hours).
- All parent replay IoUs matched the source trace for 46/46 states. Both sparse arms recorded candidate density and the fraction of GT grid cells selected by the mask.

## Results

| Comparison | CENTER GT-hard-background margin | CENTER GT rank | Fixed-state IoU |
|---|---:|---:|---:|
| Candidate vs parent | mean `+0.0111`, median `+0.0069`; 29/46 improved | 4 improved, 3 worsened, 39 tied | mean `+0.0327`, median `+0.00027`; 23 wins, 20 losses, 3 ties |
| Shifted control vs parent | mean `-0.0335`, median `-0.0281`; 19/46 improved | 6 improved, 4 worsened, 36 tied | mean `-0.0211`, median `-0.00044`; 21 wins, 23 losses, 2 ties |
| Candidate vs shifted control | mean `+0.0446`, median `+0.0436`; 29/46 improved | candidate better in 5, worse in 6, tied in 35 | mean `+0.0537`, median `+0.00261`; 23 wins, 22 losses, 1 tie |

The candidate mask covered a mean `91.35%` of GT grid cells (`94.41%` pre-onset; `88.30%` at onset). The shifted control covered `0%`, while both arms selected the same mean `20.01%` of the search grid. Across 23 sequences, the candidate-minus-control mean margin was positive in 17 and negative in 6; the largest positive sequence contributed 13.4% of the sum of positive sequence deltas.

## Interpretation and decision

**Supports the next single-variable screen; does not establish tracking gain.** At matched mask density, the target-conditioned arm improves CENTER's GT-versus-hard-background margin relative to both parent and shifted control, with the effect distributed across sequences. This supports the spatial target-conditioning part of the hypothesis. GT rank is mostly unchanged and is not better than the shifted control by win count; the IoU median advantage is small and win/loss counts are nearly even. Therefore this result does not establish stable localization improvement or a sequence-level performance gain.

The frozen screening gate asks whether target/background ordering changes in the expected direction, reaches CENTER, and differs from a non-target spatial control. The margin criterion passes (29/46 states and 17/23 sequence means favor the candidate; the shifted control does not show the same margin change). Proceed only to the already configured matched parent/candidate 16-epoch quick screen on the 176-sequence train proxy, with no new loss or mask tuning. Compare fixed-state margin alongside per-sequence tracker results. The development split has already been used for the block11 direction and will not be reused as an independent check; the locked holdout remains closed until this candidate is frozen after training. Historical full-244 results remain selection-exposed.

## Prior output provenance repair

The pre-existing `M1_SPARSE_NECK_SHIFTED_CONTROL_CALIB46_20261022/` output is retained but excluded from this analysis: its `run_manifest.json` was byte-identical to the target-conditioned candidate manifest and named the arm as observational. It cannot establish that the shifted environment control was applied. The explicit CLI arm and manifest fields in the replay outputs supersede that ambiguous directory; no old files were removed.

## Post-training fixed-state check

The matched 16-epoch quick runs were evaluated at epoch16 on the same 23 calibration events and 46 pre/onset states. The parent used `M1_SPARSE_CROSS_NECK_PARENT_QUICK_20261022`; the candidate used `M1_SPARSE_CROSS_NECK_QUICK_MATCHED_RETRY2_20261022`. Both evaluations used explicit config and checkpoint paths and FP32 fixed-state replay.

| Metric | Parent | Candidate | Candidate minus parent |
|---|---:|---:|---:|
| CENTER GT-hard-background margin | 0.19043 | 0.19657 | +0.00615; 25/46 states improved |
| Post-fusion query GT-hard-background margin | 0.05230 | 0.05578 | +0.00349; 34/46 states improved |
| CENTER GT rank | 1.587 | 1.674 | worse by 0.087; 5 improved, 1 worsened, 40 tied |
| Fixed-state replay IoU | 0.47647 | 0.46636 | -0.01012; 27/46 states improved, 16 worsened |

This preserves the mechanism signal after training, but the rank and replay IoU do not improve. The quick screen remains **inconclusive for localization/tracking benefit** and does not justify opening the locked holdout or tuning top-k/dilation. The candidate is frozen as a diagnostic result; the next decision is to audit the train/eval state contract or stop this route, rather than expand the mask search.

Post-training evidence:

- Parent fixed-state output: `output/experiments/M1_SPARSE_CROSS_NECK_PARENT_QUICK_MATCHED_RETRY2_FIXED46_PARENT_20261022/`
- Candidate fixed-state output: `output/experiments/M1_SPARSE_CROSS_NECK_QUICK_MATCHED_RETRY2_FIXED46_CANDIDATE_20261022/`
- Runner: `tools/run_m1_screening_fixed_state_flow_20261022.py`, SHA256 `06c2a501ddafc0ad6e17c34edf41ce9ef67190d2cda3daaff9ea34806eeb5e27`
- Updated source snapshot: `experiments/snapshots/m1_sparse_shifted_control_runner_20261022/final_sources_v3_explicit_override.tar.gz`, SHA256 `f06212efd2ad47cf6f2f090fc3d56d4c9ade9c16c5d0f9990e03f117db7df2c8`
