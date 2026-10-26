# M1 MSHA DER Quick Train/Test (2026-10-22)

## Scope and hypothesis

This exploratory run tests whether a DiMP-style supervised target-versus-hard-negative evidence map can condition the existing MSHA cross-modal Value gate. The parent is the epoch-3 Early-Conditioned MSHA gate model; only DER parameters were trained. CENTER, tracking losses, optimizer family, and tracker state/update behavior were not redesigned. This is a train-side 12-sequence screen, not a generalization evaluation.

The DER design and literature boundary are documented in `m1_msha_discriminative_router_design_20261022.md`. The 16-sample screen preceding this run was inconclusive because the dataset resamples pairs each epoch. The current run was therefore an exploratory quick test, not a learnability gate pass.

## Run contract and provenance

- Training config: `experiments/tbsi_track/M1_MSHA_DER_QUICK3_20261022.yaml`, SHA256 `1d8ca8bac510b0a02d07c9d503f6afe88c4a6833e2145031bfcbef8ea64f8f4f`.
- Parent initialization: `M1_EARLY_CONDITIONED_MSHA_GATE_PROXY_20261021/checkpoints/TBSITrack_ep0003.pth.tar`.
- Training: LasHeR train 512 sampled pairs/epoch × 3 epochs, batch 2, 256 steps/epoch, FP32, worker 4; 1× NVIDIA GeForce RTX 4090 D; Python 3.8.20, PyTorch 1.9.0+cu111, CUDA 11.1.
- DER `Loss/der_rank` at epoch ends: train `0.06152, 0.05141, 0.04797`; validation `0.05756, 0.05026, 0.04899`. This indicates the auxiliary objective was optimized, but it does not show that its score maps correctly rank target against distractors at inference.
- Checkpoint: `output/experiments/M1_MSHA_DER_QUICK3_20261022/checkpoints/TBSITrack_ep0003.pth.tar`, SHA256 `a8172e9a13fb8c2d90179bc1cb87b3f9868c5ec11eb9507ea42be29e4acac139`.
- Evaluation cohort: fixed 12-sequence train-side manifest, SHA256 `c19ea10ac02a630e47da059bc8447b37656abcf0106425802296a0a7e3542262`; predictions are full closed-loop FP32, serial tracker threads 0. Baseline is the matching `lam0` parent from `M1_EARLY_GATE_STRENGTH_REPLAY_20261022`.
- Protected M1 config/checkpoint hashes were verified unchanged: `fab8e818...16b4e1` and `1302de3b...f54e04ef`.
- Evaluation command: `TBSI_TRACE_TRAIN_LIST=<frozen 12-sequence list> TBSI_CHECKPOINT_OVERRIDE=<DER epoch3 checkpoint> TBSI_INFER_FP16=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 TBSI_DISABLE_TRACKER_CACHE=1 python -u tracking/test.py tbsi_track M1_MSHA_DER_QUICK3_20261022 --runid der_ep3 --dataset_name m1_trace_lasher_train --threads 0 --num_gpus 1`.
- Metric analysis: `python tools/analyze_m1_msha_der_quick3_20261022.py`.

## Results

| Arm | AUC | OP50 | OP75 | Precision | Normalized Precision |
|---|---:|---:|---:|---:|---:|
| Early-gate parent (`lam0`) | 78.826 | 94.255 | 77.544 | 97.350 | 92.876 |
| DER epoch 3 | 78.625 | 94.143 | 77.036 | 97.094 | 92.895 |
| Paired difference | -0.201 | -0.112 | -0.507 | -0.256 | +0.019 |

Per-sequence AUC deltas, DER minus parent: `2girlgoleft -0.242`, `2rdtribike -0.529`, `2runone +0.146`, `boywithshorts2 -0.442`, `farwhiteboy -0.607`, `guardman -0.862`, `lightcarfromnorth 0`, `moto2trees 0`, `motofromdark -1.051`, `whitebikebehind +1.414`, `whitecarturn85 +0.056`, `whiteman -0.289`. This is 3 wins, 7 losses, 2 ties; median AUC delta is -0.265 points.

Predicted boxes first diverge from the parent at frame 1–4 across the 12 clips; changed-box fraction ranges from 64.8% to 98.6%. This confirms that the trained DER path changes the closed-loop output early. The predominantly negative paired result shows that this intervention is not reliably target-selective. It does not localize whether the problem is score-map ranking, gate correction direction/calibration, or subsequent state feedback.

## Decision

**NOT-SUPPORTED / STOP CURRENT DER TRAINING BRANCH.** The ranking loss decreases, but the 12-sequence closed-loop metrics do not improve, with broad early output changes and 7/12 sequence losses. Do not extend epochs, widen the DER head, or tune its strength from this cohort. This result does not establish that discriminative target/background supervision is generally ineffective; it rejects this implementation and training/inference contract on the current evidence.

The remaining discriminating check is a fixed-state audit on an outcome-blind cohort: for identical crop, template, and state, measure DER's per-modality GT rank and GT-versus-hard-background score margin, then record the resulting gate correction at GT and non-target locations. If those ranks do not improve, the score-map supervision/representation is the failure point. If ranks improve but gate direction or CENTER ranking degrades, the score-to-MSHA interface is the failure point. Until that audit is run, the specific failure stage remains unresolved.

## Artifacts

- Paired summary: `output/experiments/M1_MSHA_DER_QUICK3_20261022/paired_summary.json`.
- Per-sequence metrics: `output/experiments/M1_MSHA_DER_QUICK3_20261022/per_sequence_metrics.csv`.
- First-divergence summary: `output/experiments/M1_MSHA_DER_QUICK3_20261022/trajectory_summary.csv`.
- Training log/checkpoint: `output/experiments/M1_MSHA_DER_QUICK3_20261022/logs/training.log`, `output/experiments/M1_MSHA_DER_QUICK3_20261022/checkpoints/TBSITrack_ep0003.pth.tar`.
- DER predictions: `output/test/tracking_results/tbsi_track/M1_MSHA_DER_QUICK3_20261022_der_ep3/`.
- Parent predictions and frozen cohort: `output/test/tracking_results/tbsi_track/M1_EARLY_GATE_STRENGTH_REPLAY_20261022_lam0/`, `output/experiments/M1_EARLY_GATE_STRENGTH_REPLAY_20261022/manifests/sequence_list.txt`.
- Analysis implementation: `tools/analyze_m1_msha_der_quick3_20261022.py`.
