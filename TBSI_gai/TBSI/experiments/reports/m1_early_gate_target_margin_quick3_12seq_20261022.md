# Early Target-Margin Correction to the MSHA Gate: Quick Train/Test (2026-10-22)

## Hypothesis and operator

The previous strength replay showed a useful but highly concentrated closed-loop gain: the trained Early-Conditioned MSHA Gate improved `motofromdark`, while several other sequences regressed. The hypothesis here was that the gate lacks target-specificity because its early query feature is only the maximum template-to-search match, which can be a background response. Add a learned, zero-initialized residual based on a local uniqueness margin and cross-modal peak consistency.

At TBSI fusion locations 3, 6, and 9, early block-2 normalized template/search features produce modality similarity maps:

\[
S^m_{ij}=\langle \hat t_i^m,\hat s_j^m\rangle,\quad
u_i^m=\max_j S^m_{ij},\quad
b_i^m=\operatorname{meanTopK}_{j\notin\mathcal N_3(j_i^*)}S^m_{ij},\quad
\Delta_i^m=u_i^m-b_i^m.
\]

The 7 input channels are RGB/TIR peak similarity, cross-modal template agreement, RGB/TIR peak-to-hard-background margin, and each modality's similarity evaluated at the other modality's peak. A small `7 -> 16 -> 2` MLP predicts an additive correction to the existing gate logits:

\[
\ell'_i=\ell_i+\tanh(\alpha)\tanh(f([u_i^r,u_i^t,a_i,\Delta_i^r,\Delta_i^t,S^r_{i,j_t^*},S^t_{i,j_r^*}])).
\]

The last layer is re-zeroed after ViT initialization, so the new path is initially an exact no-op. The feature is a *local match uniqueness proxy*, not a GT-labelled target/background margin: the highest response is not guaranteed to be the target. No new loss was introduced; the unchanged tracker training losses provide the only supervision.

## Experimental contract

- Parent: `M1_EARLY_CONDITIONED_MSHA_GATE_PROXY_20261021`, epoch 3; only its 15 trained Early-Conditioned Gate tensors are used as the train/test parent. Parent checkpoint SHA256 `1704b42f0cf0bb8dbb598b85c55a3eec0c303516c9fe7f055d4062a540bfb8c5`.
- Training: LasHeR train; 512 samples/epoch, batch 2, 256 optimizer steps/epoch, 3 epochs / 768 steps total; seed 42 config, FP32, worker 4, original GIoU/L1/location objectives and optimizer/schedule; only 15 tensors / 489 scalar parameters in the new margin branches are trainable. No new ranking loss or training-strategy change.
- Evaluation: same frozen 12 train-side clips as the prior strength replay (6 selected failure cases, 6 stable controls), full closed-loop clips, same crop/state policy, FP32 serial tracker `threads=0`. M1 and gate-only columns reuse the exact predictions from the immediately preceding replay; the margin arm changes only the trained margin branch. This is a small train-side screen, not a generalization test; exact candidate video exposure is unknown.
- Environment: 1x NVIDIA GeForce RTX 4090 D, Python 3.8.20, PyTorch 1.9.0+cu111, CUDA 11.1, OMP/MKL 1. GPU hours were not instrumented (`NA`).
- Code HEAD before edits: `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`. Config SHA256 `393beaf2b6a10cb4f6dfe1b218aae827331d48b822c83cf485a448acb1569489`. Source snapshot before change SHA256 `72d41c01afc7ead24601877580dbf6ea1d7d9243fe9ff09652b0233b532993ab`; source/config/tools snapshot after change SHA256 `99d34f1a65c861836a236f9fcaae1f389e9cd5ce19fe3b3f442b6e4b676c4a4b`.
- Trained checkpoint SHA256 `26f8679f8571462973fc49c2de8f25a432b609397fdccfc29a6b65d8a41ca1f9`. In training state, 39 BN buffers changed; the evaluation merge restores all 282 parent state entries and overlays only 15 margin tensors. Merged checkpoint SHA256 `f9bfd5cb69d3d2b580fb1bec26a762875d6b5ce0d3bf1cab57a93a6d07ab9596`. No other shared parameter changes were found.

## Results

Macro values are per-sequence means on the fixed 12 clips.

| Arm | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| M1 | 78.826 | 94.255 | 77.544 | 97.350 | 92.876 |
| Early gate only | 79.438 | 95.085 | 77.535 | 98.216 | 93.705 |
| Early gate + trained margin correction | 78.546 | 93.877 | 77.380 | 96.467 | 92.712 |

The margin arm versus M1 has mean AUC delta `-0.280` points, median `-0.117`, with 4 wins and 8 losses. Against the gate-only parent it has mean `-0.892`, median `-0.206`, with 4 wins and 8 losses. Largest drops are `motofromdark -8.844`, `guardman -1.519`, `whitebikebehind -0.558`, and `lightcarfromnorth -0.419` AUC points. Largest gains are `moto2trees +0.667`, `2rdtribike +0.388`, and `whiteman +0.253`.

Predicted boxes first diverge from gate-only between frame 1 and frame 3 in all 12 sequences; 70%-94% of boxes differ over the full clip. `motofromdark` mean center shift is 7.05 pixels (maximum 46.80), and its mean IoU drops 0.092. This is a strong immediate perturbation with the wrong direction on the sequence that previously supplied most of the gate's aggregate gain.

## Interpretation

**Conclusion: refutes this trained local-margin correction under the present interface and short-training contract; do not promote or extend training.** The new branch is expressive enough to move predictions, but the correction is not target-selective in this test: it changes the initial boxes broadly and removes the large `motofromdark` recovery. This points to a representation/conditioning mismatch in the proxy, not an inability to optimize the branch: all 489 parameters receive finite gradients, and the training loss remains finite. The branch has no GT-defined target/background correspondence; its maximum-vs-secondary response contrast can still describe a distractor peak.

The learned final-layer weight norms remain small (`~0.0012-0.0026` across the three blocks), yet the closed-loop outcomes differ substantially. That is consistent with argmax/state feedback sensitivity and means raw parameter magnitude is not evidence of a small tracking effect.

Do not add epochs, widen the MLP, or add another loss yet. The next discriminative step is fixed-state first-frame analysis on `motofromdark`, `guardman`, `whitebikebehind`, `moto2trees`, and `2rdtribike`: compare old/new gate logits, margin-head correction per template token, GT-center and hard-background CENTER score/rank, decoded box, and then repeat the same frame with the margin residual forced to zero. Include a non-target spatial-token control. If margin-head changes do not improve GT-vs-hard-background ranking on the same crop, reject this margin definition and redesign the evidence extraction before retraining.

## Implementation validation and provenance

- Added default-off `EarlyTargetMarginGate`, config gate, and a margin-only trainable-parameter filter. Shared model, CENTER head, loss, optimizer family, dataset pipeline, tracker state logic, and M1 anchors were not edited for this experiment.
- Verified syntax/config parsing, `(B,64,7) -> (B,64,2)` module contract, 489 total added parameters across 3 locations, finite gradients, and exact zero-residual `V2TemplateFusion` output parity on fixed random inputs.
- Similarity maps are now reused by the existing query-max path and the new margin path; this is algebraically identical to the tested implementation and avoids a duplicate matrix multiplication. Exact zero-residual parity passed after this change.
- Training command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 /root/autodl-tmp/conda_envs/tbsi/bin/python -u tracking/train.py --script tbsi_track --config M1_EARLY_GATE_TARGET_MARGIN_QUICK3_20261022 --save_dir output --mode single --use_lmdb 0 --use_wandb 0`.
- Evaluation command: `bash tools/run_m1_early_gate_margin_replay_20261022.sh`; analysis: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/autodl-tmp/conda_envs/tbsi/bin/python tools/analyze_m1_early_gate_margin_replay_20261022.py`.
- Training logged finite loss throughout 3 epochs; training IoU at epoch-end was approximately `0.776/0.792/0.798`; validation IoU was `0.716/0.728/0.701`. The validation trend does not provide evidence that more training would fix the tracking result.
- A first merge utility invocation lacked repository `sys.path` and failed to unpickle the checkpoint; the utility was corrected and merge provenance validation then passed. No training/evaluation result was produced by the failed utility invocation.
- Protected M1 YAML SHA256 remains `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`; protected M1 checkpoint SHA256 remains `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.

## Artifacts

- `output/experiments/M1_EARLY_GATE_TARGET_MARGIN_QUICK3_20261022/`
- `output/experiments/M1_EARLY_GATE_MARGIN_OPERATOR_20261022/{summary.json,per_sequence_metrics.csv,analysis_stdout.json,checkpoint_merge_provenance.json,logs/}`
- `output/test/tracking_results/tbsi_track/M1_EARLY_GATE_TARGET_MARGIN_QUICK3_20261022_margin_ep3/`
- `experiments/tbsi_track/M1_EARLY_GATE_TARGET_MARGIN_QUICK3_20261022.yaml`
- `tools/merge_m1_early_gate_margin_checkpoint_20261022.py`
- `tools/run_m1_early_gate_margin_replay_20261022.sh`
- `tools/analyze_m1_early_gate_margin_replay_20261022.py`
