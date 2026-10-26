# M1 Early-Conditioned Gate Strength Replay (2026-10-22)

## Question and design

This diagnostic tests whether scaling the trained Early-Conditioned MSHA Gate by `lambda in {0, 0.5, 1}` can reduce later closed-loop drift while preserving gains on difficult sequences. Only the multiplier changes. The gate correction is `lambda * tanh(scale) * tanh(logit)`, applied through an exponential to cross-modal Value messages; `lambda=0` bypasses the gate and is the M1-state control.

The replay uses the same merged checkpoint, full-clip tracker initialization, state update, crop policy, FP32 inference, and serial `threads=0` setting for each arm. The frozen manifest contains 12 LasHeR training sequences: six previously selected failure cases and six stable controls. Candidate checkpoint training used LasHeR training data; sequence-level exposure to these clips is unknown, so this is a mechanism screen, not independent generalization evidence.

## Provenance

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15. Checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- Gate source: `M1_EARLY_CONDITIONED_MSHA_GATE_PROXY_20261021`, epoch 3. Checkpoint SHA256 `1704b42f0cf0bb8dbb598b85c55a3eec0c303516c9fe7f055d4062a540bfb8c5`.
- Replay checkpoint consists of the 267 M1 state entries plus only the 15 trained gate tensors; all non-gate candidate shared changes are zero. SHA256 `40b69b33a52b5c7522c5d936aefd474a28048f50a12ee0ef88e1f0fd0f789861`.
- Config SHA256 `b64e664b0a6c2ff77ce488f0187bcf47f2f23143b49bcec424a417d1c24c14bf`; manifest SHA256 `c19ea10ac02a630e47da059bc8447b37656abcf0106425802296a0a7e3542262`.
- Code was run from working tree at HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`; relevant source/config/scripts snapshot SHA256 `8ed25d7224ceb7555e1d4b47180e1ec176f18594ed69429c7559539afc693ceb`.
- Environment: 1x NVIDIA GeForce RTX 4090 D; Python 3.8.20, PyTorch 1.9.0+cu111, CUDA 11.1; FP32; OMP/MKL threads 1; tracker processes serial. GPU-hours were not instrumented (`NA`).

## Results

Macro averages below are per-sequence means on these 12 training clips, not full-test metrics.

| Gate strength | AUC | OP50 | OP75 | Precision | Norm Precision |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 78.826 | 94.255 | 77.544 | 97.350 | 92.876 |
| 0.5 | 79.364 | 94.901 | 77.618 | 98.216 | 93.689 |
| 1.0 | 79.438 | 95.085 | 77.535 | 98.216 | 93.705 |

| Paired against lambda 0 | Mean AUC delta (points) | Median delta | Wins / losses / ties |
|---|---:|---:|---:|
| lambda 0.5 | +0.538 | 0.000 | 4 / 5 / 3 |
| lambda 1.0 | +0.612 | +0.041 | 7 / 2 / 3 |

| Sequence | AUC delta, lambda 0.5 | AUC delta, lambda 1.0 |
|---|---:|---:|
| 2girlgoleft | -0.138 | -0.104 |
| 2rdtribike | +0.071 | +0.071 |
| 2runone | -0.194 | +0.049 |
| boywithshorts2 | +0.102 | +0.034 |
| farwhiteboy | -0.160 | +0.192 |
| guardman | 0.000 | +0.369 |
| lightcarfromnorth | 0.000 | 0.000 |
| moto2trees | +0.476 | +0.476 |
| motofromdark | +6.370 | +6.370 |
| whitebikebehind | -0.074 | -0.112 |
| whitecarturn85 | 0.000 | 0.000 |
| whiteman | 0.000 | 0.000 |

The `motofromdark` result accounts for about 87% of the total lambda-1 AUC gain. Excluding it, lambda 1 averages only `+0.089` points across the other 11 sequences; the median is `+0.034`. On the six selected failure cases, lambda 1 averages `+1.145` points, but excluding `motofromdark` the remaining five average about `+0.100`. Stable controls average `+0.080`. Thus the apparent failure-cohort benefit is not distributed broadly.

## Where trajectories first diverge

Most arm predictions are identical for the initial frames. The first changed predicted boxes occur between frames 17 and 112, with no five-frame run meeting the predeclared severe IoU regression threshold (`delta IoU < -0.25`). The first 20-frame mean IoU change is near zero in all sequences. This does not show improved initial target selection; it shows that small changes can appear later and then interact with the tracker state/crop loop.

Notable cases:

- `motofromdark`: lambda 1 first differs at frame 65; the fraction of later frames with IoU below 0.1 falls from `0.118` to `0.013`; AUC improves by `6.37` points. This is the clear positive mechanism case, but dominates the aggregate.
- `farwhiteboy`: lambda 0.5 first differs at frame 32 and loses `0.160` AUC points; lambda 1 first differs at frame 112 and gains `0.192`. This strength dependence is consistent with later closed-loop sensitivity, not a broad monotonic effect.
- `2girlgoleft`: lambda 0.5 and 1 first differ at frames 28 and 102, respectively, and lose `0.138` / `0.104` points.
- `whitebikebehind`: both strengths first differ at frame 57; lambda 1 loses `0.112` points.
- `guardman`: lambda 1 first differs at frame 96 and gains `0.369` points, with maximum center displacement `10.55` pixels. The AUC gain therefore coexists with a substantial localized trajectory deviation.

## Interpretation and decision

**Conclusion: supports one local mechanism follow-up; does not support global strength selection or a claim that the gate reliably suppresses drift.** Lambda 1 is directionally better than lambda 0.5 in aggregate, but almost all aggregate gain comes from one sequence; two sequences regress at lambda 1, and the median paired change is near zero. This is not enough evidence to promote the gate or choose lambda 1 for full evaluation.

The intervention can alter later tracker trajectories while leaving the initial output unchanged. That establishes downstream sensitivity to the cross-modal Value path, but does not identify whether a changed gate is correcting a target peak or simply perturbing the state update. The selected training-side cohort is small and potentially exposed during candidate training.

Next, keep the gate architecture fixed and analyze `motofromdark`, `farwhiteboy`, `2girlgoleft`, `whitebikebehind`, and `guardman` at the first-divergence frame. Pair each closed-loop divergence with a same-state/same-crop open-loop replay and log GT versus hard-background peak margin, gate values, decoded center/size, and the next-state/crop displacement. Include a non-target spatial-token control. This separates direct frame-level peak correction from drift mediated by feedback. Do not extend the lambda grid or start longer training until this mechanism test shows a repeatable target-specific signal across multiple sequences.

## Execution issue

The first batch attempt failed before model construction because the new diagnostic config omitted `TRAIN.TBSI_DROP_PATH`; the transformer indexed the three TBSI blocks and raised `IndexError: list index out of range`. No bbox predictions were written. The config was aligned with protected M1 (`[0.0, 0.0, 0.0]`), a one-sequence smoke passed, and the three full 12-sequence arms then completed. The original failed batch log was overwritten by the successful `lam0` rerun; the reproduced one-sequence traceback is preserved separately at `output/experiments/M1_EARLY_GATE_STRENGTH_REPLAY_20261022/logs/config_failure_traceback.log`. Successful logs and outputs are in the same experiment directory.

## Artifacts

- `output/experiments/M1_EARLY_GATE_STRENGTH_REPLAY_20261022/summary.json`
- `output/experiments/M1_EARLY_GATE_STRENGTH_REPLAY_20261022/per_sequence_metrics.csv`
- `output/experiments/M1_EARLY_GATE_STRENGTH_REPLAY_20261022/trajectory_summary.csv`
- `output/experiments/M1_EARLY_GATE_STRENGTH_REPLAY_20261022/manifests/sequence_list.txt`
- `output/test/tracking_results/tbsi_track/M1_EARLY_GATE_STRENGTH_REPLAY_20261022_{lam0,lam05,lam1}/`
- `tools/run_m1_early_gate_strength_replay_20261022.sh`
- `tools/analyze_m1_early_gate_strength_replay_20261022.py`
