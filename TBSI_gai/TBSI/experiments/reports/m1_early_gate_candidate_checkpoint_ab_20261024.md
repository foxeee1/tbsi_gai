# Early-conditioned MSHA gate candidate-checkpoint A/B replay

## Question and decision

This diagnostic asks whether the trained gate itself changes target selection or localization when every learned checkpoint tensor is held fixed. The candidate epoch-15 checkpoint is used for both arms; only test-time `TBSI_TEST_EARLY_GATE_STRENGTH` changes from 0 (bypass) to 1 (gate enabled).

**Conclusion: `REFUTED-AS-A-CENTER-RANKING-MECHANISM / DOWNSTREAM-REGRESSION-SENSITIVITY-REMAINS`.** Do not start full training of a spatially routed successor on the strength of this gate. The gate does not change Hann top-1 or valid GT rank on any sampled identical state; its measured direct localization effect is effectively zero. It does perturb the box regression output slightly, and closed-loop trajectories can consequently separate, but the direction is inconsistent.

## Provenance and protocol

- Candidate: `M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023_RETRY1`, epoch 15. Checkpoint SHA256: `e38d18fe50c2895e75e4b9392bf22d9323bb8e6ad31ccbe39ad8c038e3fc1811`.
- Config SHA256: `7f73d1fab162f5b869a8fb43c51784b3aea0cfec9099bcfabd51f42ea2d34b58`.
- Parent anchor integrity was rechecked: M1 config SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`; M1 epoch-15 checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- Git HEAD: `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`; no model, tracker, or evaluator source changes were made for the experiment.
- Runtime environment: Python `3.8.20`, PyTorch `2.0.0+cu118`, CUDA `11.8`, 1× NVIDIA GeForce RTX 4090 D; FP16 inference, `threads=0`, no DataLoader workers. This is a matched within-run A/B environment; it is not evidence that the earlier training environment was reproduced.
- Seven LasHeR test sequences were preselected from the existing outcome report: four losses and three gains. This cohort is outcome-selected and supports mechanism screening only.
- Phase 1: 14 serial closed-loop replays, template updates off, all paired sequence frame counts matched. Arms shared only initialization state; the first decoded box differed slightly and state/crop diverged from frame 2.
- Phase 2: five fixed frame states per sequence sampled by the preregistered first/25%/50%/75%/last rule from the gate-off trace, deduplicated. Both arms used the exact same candidate checkpoint, first-frame template, saved tracker state, image, and crop. All 35 crop and pre-state pairs were identical.
- Phase 3: the same 35 events preserved raw score, size, and offset maps for source attribution. No additional sequences or training were used.

## Results

### Fixed-state localization

| Measure | Result |
|---|---:|
| Paired fixed states | 35 |
| Exact crop/state pairs | 35/35 |
| Events with valid target-in-crop CENTER rank | 28 |
| Hann top-1 changes | 0/28 |
| GT rank improved / worsened | 0 / 0 of 28 |
| Mean / median decoded IoU delta (gate on minus off) | `-0.0000080` / `0.0000000` |
| Positive / negative / unchanged IoU events | `9 / 9 / 17` |
| Mean Hann target-background margin delta | `+0.0000550` |

The remaining 7 fixed states had the target outside the search crop, so GT-overlap rank is undefined there. They are retained for box-output comparison and excluded from rank claims.

### Head-map attribution

Across the 35 exact-state pairs, mean absolute map changes were:

- Raw CENTER score map: `1.11e-5`.
- Hann-weighted CENTER response: `6.03e-6`.
- Size map: `8.19e-5`.
- Offset map: `6.64e-4`.

This identifies a small perturbation that is larger in the offset regression map than in the CENTER response. It does not improve the sampled target ordering. Due to nonlinearity and argmax/discrete decoding, map magnitudes alone do not establish a general causal contribution, but they explain how an initially tiny box perturbation can alter the next crop and amplify later.

### Closed-loop trajectories

Across phase-1 full clips, the gate-on minus gate-off mean IoU differences were `-0.0376`, `+0.1841`, `+0.1748`, `+0.3226`, `+0.0012`, `-0.0038`, and `+0.0450` in the fixed sequence order. These mixed effects arose after state/crop divergence on frame 2. They are closed-loop sensitivity evidence, not direct frame-level improvements or an independent performance estimate.

## Interpretation

The experiment separates two claims:

1. **CENTER selection repair:** not supported on these 28 valid identical-state events. There were no changes to Hann top-1 or GT rank and effectively no mean IoU effect.
2. **Ability to perturb downstream tracking:** supported narrowly. The gate produces tiny score/regression changes; early box changes can trigger large and sign-varying closed-loop trajectories. That does not establish the gate as the cause of the full-15 checkpoint's small aggregate benchmark delta, because the full checkpoint also contains jointly trained shared parameters.

The gate hypothesis is therefore insufficient as the target-selection mechanism for the proposed innovation. A new spatial fusion router is **not** justified by these results alone. Existing spatial-token bridge and search-side reliability diagnostics also failed to change CENTER/query ranking materially; do not simply replay those operators under a new name.

## Next decision

Stop expanding the current gate formula and do not start tonight's full training from this evidence. The valuable result is a sharper problem definition: any next fusion candidate must show target-specific change in search-side interaction and CENTER response on identical crop/state inputs, against a spatially shifted/decoy control, before training. A new idea should first pass a minimal tensor-contract and fixed-state score-map diagnostic. If no candidate can change target rank under that control, redirect the method question away from MSHA message routing rather than adding another gate.

## Artifacts

- Protocol: `experiments/protocols/m1_early_gate_candidate_checkpoint_ab_20261024/preregistration.md`.
- Run manifest and logs: `output/experiments/M1_EARLY_GATE_CANDIDATE_CHECKPOINT_AB_20261024/`.
- Full-clip trace: `trace/off/`, `trace/on/`; paired analysis: `paired_summary.csv`, `paired_frames.json`.
- Fixed-state trace and map outputs: `fixed_state_maps/event_manifest.json`, `paired_trace.json`, `head_maps_off.npz`, `head_maps_on.npz`, `fixed_state_paired.csv`, `fixed_state_summary.json`.
- Runner: `tools/run_m1_early_gate_candidate_checkpoint_ab_20261024.sh`.
- Fixed-state runner and analysis: `tools/replay_m1_early_gate_fixed_state_ab_20261024.py`, `tools/analyze_m1_early_gate_fixed_state_ab_20261024.py`.
- Phase source bundles: `experiments/snapshots/M1_EARLY_GATE_CANDIDATE_CHECKPOINT_AB_20261024/`.
