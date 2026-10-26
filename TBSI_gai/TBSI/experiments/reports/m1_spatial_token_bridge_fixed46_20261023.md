# M1 Spatial Token Bridge Fixed-State Screen

## Decision

**NO-GO for full training of the current operator.** In the preregistered 23-sequence, 46-state LasHeR training calibration cohort, target-conditioned bridge messages did not improve CENTER rank, post-fusion query rank, or fixed replay IoU. Target-versus-shifted differences were around `1e-6` in localization-facing margins and were inconsistent across states. This is `暂不支持`; it is not evidence that all spatial cross-modal correspondence methods are ineffective.

## Question and intervention

The screen asks whether sparse, target-conditioned RGB/TIR search-token correspondence changes the fused representation in a useful direction more than an equal-density shifted correspondence control. The bridge selects 8 of 256 search tokens per modality from each modality's own masked template foreground prototype, performs bidirectional soft correspondence, and applies a learned rank-32 residual to selected tokens. The original fusion neck, CENTER head, crop, and replay state are unchanged.

The candidate weights are from a one-epoch bridge-only fit smoke on 16 LasHeR training samples: 8 optimizer steps, batch 2, initialized from protected M1. This checkpoint is a mechanism-screening checkpoint, not a performance-trained model.

## Contract and provenance

- Parent: `TBSI_E_MHSA_FULL_4090D_seed42`; protected parent YAML and checkpoint hashes were checked and are documented in the preregistration.
- Candidate config: `experiments/tbsi_track/M1_SPATIAL_TOKEN_BRIDGE_SCREEN_20261023.yaml`, SHA256 `a557fcce678d441658c37622042592fceb9e7af7f1044e5284cf532dc75b5101`.
- Candidate checkpoint: `output/experiments/M1_SPATIAL_TOKEN_BRIDGE_SCREEN_20261023/checkpoints/TBSITrack_ep0001.pth.tar`, SHA256 `644c2c455f4e3e928be4115bef7ddde70b020d571a7855223f3cfe7024a88757`.
- Fixed event manifest: `output/experiments/M1_SCREENING_EVENT_MINING_CALIBRATION_20261022/first_persistent_events.jsonl`, SHA256 `89f2f17b6cbec84c3d9de937d73e9241495d8e85a418590a67e6f3f88e56658d`.
- Fixed sequence list: `experiments/protocols/m1_screening_v1_20261022/calibration_sequences.txt`, SHA256 `af81eeacc1db375218a075991a0c799e0e01022c60d2790f141c96bfbb8091a8`.
- Pre-change snapshot: `experiments/snapshots/M1_SPATIAL_TOKEN_BRIDGE_20261023/prechange_sources.tar.gz`, SHA256 `c77b6780e8fdb92de8511b5d5588d3da78e649561eb4065fa4301631072c36d7`.
- Source bundle including shifted-control fix: `experiments/snapshots/M1_SPATIAL_TOKEN_BRIDGE_20261023/postfix/spatial_token_bridge_sources.tar.gz`, SHA256 `3ccce72120ac9bef00828f6b3cc5224304d543e581fe37b4375d22300af70051`.
- Runtime: 1x NVIDIA GeForce RTX 4090 D; Python 3.8.20; PyTorch 1.9.0+cu111; CUDA 11.1; FP32; one torch thread. GPU hours were not captured (`NA`).
- Each completed arm has 46 rows in `per_event.csv` and 1,472 rows in `stage_metrics.csv`. The `off` arm matched M1 fixed-state replay on all 46 states.

## Paired results

All deltas below are paired by `event_id`; positive margin delta is favorable, positive rank delta is unfavorable, and positive IoU delta is favorable. `target` and `shifted` use the same candidate checkpoint.

| Readout | Target - off | Shifted - off | Target - shifted |
|---|---:|---:|---:|
| CENTER GT vs hard-background margin, mean | -2.64e-7 | -1.34e-6 | +1.07e-6 |
| CENTER rank changes | 0/46 | 0/46 | 0/46 |
| Post-fusion query GT vs hard-background margin, mean | +7.39e-8 | -3.25e-7 | +3.99e-7 |
| Post-fusion query rank changes | 0/46 | 0/46 | 0/46 |
| Fixed replay IoU, mean | -1.97e-6 | +6.51e-7 | -2.62e-6 |

For CENTER margin, target beat shifted on 26 states and lost on 20; the mean advantage was only `1.07e-6`. For post-fusion query margin it was 20 wins, 20 losses, and 6 ties, with mean difference `3.99e-7`. For replay IoU, target beat shifted on 20, lost on 22, and tied on 4; mean difference was slightly negative. These changes are too small and inconsistent to establish a useful localization effect.

The bridge did execute: candidate fraction was `0.03125` (8/256) per modality. Mean selected-token delta norms were `0.00126` RGB and `0.000664` TIR for target, and `0.00122` RGB and `0.000660` TIR for shifted. The learned branch therefore produced nonzero messages, but those messages did not materially alter the localization readouts in this screen. The observed evidence locates the failure at effective signal magnitude/reach under this short-fit checkpoint; it does not distinguish whether longer training could increase useful messages.

## Execution incident

The first shifted run, `output/experiments/M1_SPATIAL_TOKEN_BRIDGE_SHIFTED_FIXED46_20261023/`, stopped at the first state because the control attempted to reshape eight top-k indices as a 16x16 grid. No complete comparison result was produced. The control was corrected to shift each selected `(y,x)` coordinate by half the grid with modular indexing; a shape/uniqueness smoke passed. The retry completed all 46 states at `output/experiments/M1_SPATIAL_TOKEN_BRIDGE_SHIFTED_FIXED46_RETRY1_20261023/`. The failed directory and retry directory are both retained.

## Interpretation and next action

The preregistered mechanism gate is not met: target correspondence does not yield a stable target-over-shifted margin gain and gives no discrete rank or replay improvement. Do not full-train this operator in its current form. The next justified work is to inspect why the learned residual remains this small after 8 updates and whether the training objective can drive localization-relevant bridge messages on a fixed small batch; this is a separate learnability diagnostic, not a reason to claim the current mechanism works. Any revised operator needs a new preregistration and independent run identity.

## Exact evaluation commands

`target`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
/root/autodl-tmp/conda_envs/tbsi/bin/python -u tools/run_m1_screening_fixed_state_flow_20261022.py \
  --events output/experiments/M1_SCREENING_EVENT_MINING_CALIBRATION_20261022/first_persistent_events.jsonl \
  --sequence-list experiments/protocols/m1_screening_v1_20261022/calibration_sequences.txt \
  --output-dir output/experiments/M1_SPATIAL_TOKEN_BRIDGE_TARGET_FIXED46_20261023 \
  --config experiments/tbsi_track/M1_SPATIAL_TOKEN_BRIDGE_SCREEN_20261023.yaml \
  --checkpoint output/experiments/M1_SPATIAL_TOKEN_BRIDGE_SCREEN_20261023/checkpoints/TBSITrack_ep0001.pth.tar \
  --spatial-token-bridge-mode target
```

`shifted` retry used the same command and inputs, with output directory `output/experiments/M1_SPATIAL_TOKEN_BRIDGE_SHIFTED_FIXED46_RETRY1_20261023` and mode `shifted`.
