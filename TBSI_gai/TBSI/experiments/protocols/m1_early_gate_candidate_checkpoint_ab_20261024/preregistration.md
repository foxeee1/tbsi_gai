# Early-conditioned gate candidate-checkpoint A/B replay

## Question and hypothesis

Does the trained early-conditioned MSHA gate itself change target selection or decoded localization when all learned checkpoint weights are held fixed? The candidate checkpoint is used in both arms; only `TBSI_TEST_EARLY_GATE_STRENGTH` changes (`0` bypass vs `1` trained gate).

Null: on identical frame, crop, and tracker state, enabling the gate does not consistently improve Hann-weighted target rank, target-background margin, or decoded box IoU across the selected sequences.

Alternative: enabling the gate improves one or more of those direct localization signals across multiple sequences before the tracker states diverge.

## Contract

- Parent/checkpoint: `M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023_RETRY1`, epoch 15; checkpoint SHA256 `e38d18fe50c2895e75e4b9392bf22d9323bb8e6ad31ccbe39ad8c038e3fc1811`.
- Config: `M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023_RETRY1.yaml`; SHA256 `7f73d1fab162f5b869a8fb43c51784b3aea0cfec9099bcfabd51f42ea2d34b58`.
- Data: LasHeR test; fixed representative cohort from the existing trace diagnosis: four reported regressions (`womanback2car`, `boydownplatform`, `4thboywithwhite`, `shinycarcoming`) and three reported gains (`ab_bikeoccluded`, `3rdfatboy`, `leftfarboycomingpicktheball`). This outcome-selected set supports mechanism screening only.
- Arms: same checkpoint/config/seed and tracker settings, template update off, serial `threads=0`, FP16 as in the prior representative trace; gate strength 0 versus 1.
- Primary analysis: compare only aligned frames with identical `current_crop_xywh` and `tracker_state_before` as direct-effect evidence. Report Hann top-1 changes, GT Hann rank and margin, predicted-box IoU, and center displacement. Once state/crop differs, label subsequent changes as closed-loop effects.
- Stop/decision: if direct matched-state evidence does not improve in at least four of seven sequences, or if rank/box changes are mixed without target-specific direction, do not train a new spatial-routing module. First inspect the exact frame-level discrepancy and improve the fixed-state harness. If multiple sequences show target-directed direct improvement, proceed to a single spatially indexed fusion-routing candidate with zero-init identity fallback; fixed-state target-vs-shifted controls must pass before any training.
- Not measured: full-dataset AUC/PR/NPR/OP50/OP75/FPS. This is not a full test or generalization claim.

## Evidence ladder

Minimum: successful paired traces, frame and state alignment audit, direct-versus-closed-loop separation, complete command/config/checkpoint/environment provenance.

Solid: repeat the fixed-state effect on a prespecified train-side calibration cohort with target-vs-shifted spatial controls; establish the location and direction of the routing signal before implementing the module.

Maximum: a new module passes disabled parity, gradients, frozen-state checks, and small-set fit; then run a full M1-matched 15-epoch joint training and full-244 test under a distinct experiment identity. No full training is authorized by this diagnostic contract.

## Provenance and execution log

- Source worktree contains pre-existing unrelated modifications. No model/tracker source code is changed by this diagnostic.
- Git HEAD, source snapshot, environment, command lines, resource status, output paths, and observed results are appended before/after execution.

## Fixed-state follow-up (pre-registered after phase 1)

Phase 1 produced only one same-crop/state frame per sequence because the gate changed the first decoded box slightly and the state diverged on frame 2. To prevent this feedback from contaminating direct localization evidence, phase 2 selects five frame states per sequence from the already saved gate-off trace using the fixed rule `first, 25%, 50%, 75%, last` (deduplicated). For each event and arm, initialize from the same first-frame template, set the same saved gate-off `tracker_state_before`, and run only that event frame. The arms differ only by gate strength 0 vs 1. Report direct Hann peak/rank/margin and decoded IoU/box changes; this follow-up remains a selected-sequence mechanism screen.

Phase 1 execution: 14/14 serial sequence runs completed; all arm pairs have equal frame counts. The initial analysis found one exact same-crop/state frame per sequence, no Hann top-1 or GT-rank changes on that frame, and mixed later closed-loop IoU effects. This motivated phase 2; it does not support a performance claim.

## Head-map follow-up (pre-registered after phase 2)

Phase 2 achieved identical crop and `tracker_state_before` for all 35 paired events. To distinguish score-selection changes from box-regression changes, rerun the same 35 events and preserve raw score, size, and offset maps for both arms. For the score map also compare the Hann-weighted response. This uses the same saved event manifest and candidate-control states; no event or condition is added. Outputs go to a new `fixed_state_maps/` directory so the previous phase is retained.
