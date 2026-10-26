# M1 AMCM E1-E4 serial diagnosis (2026-10-17)

All four runs use the protected M1 epoch15 checkpoint, LasHeR proxy training (256 samples/epoch, 3 epochs, batch 2, FP32, worker 1), and the same fixed difficult5 test manifest. Each run has an independent config, checkpoint, and result directory.

## Results

| Arm | Single changed factor | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---|---:|---:|---:|---:|---:|
| M1 | Protected parent | 66.71 | 78.00 | 54.34 | 89.36 | 78.00 |
| E1 | AMCM after `tbsi_fuse_search` | 65.28 | 76.26 | 53.80 | 86.35 | 77.14 |
| E2 | E1 + target-cell RGB/TIR similarity supervision | 66.70 | 78.34 | 54.05 | 88.13 | 78.58 |
| E3 | E1, jointly train AMCM + `tbsi_fuse_search` | 63.33 | 75.51 | 52.93 | 82.68 | 76.40 |
| E4 | E3 + jointly train CENTER head | 63.46 | 75.41 | 52.42 | 82.55 | 75.34 |

## Diagnosis

### E1: injection position remains relevant

Moving AMCM after the learned fusion projection improves AUC by `+0.58` relative to the earlier pre-fusion AMCM screen (`64.70`), but remains `-1.43` below M1. Position is therefore a real variable, but post-fusion modulation alone is insufficient.

### E2: supervision is the strongest positive signal

The auxiliary target-cell supervision uses detached RGB/TIR template-search similarity at the ground-truth cell and supervises the relative gate. It recovers M1 AUC (`66.70` vs `66.71`) and gives descriptive gains in OP50 (`+0.34`) and Norm Precision (`+0.58`). This does not prove robustness, but it shows the previous constant gate was at least partly caused by an underdetermined objective.

### E3: fusion projection co-adaptation is harmful under the current proxy

Unfreezing `tbsi_fuse_search` without the E2 supervision lowers AUC to `63.33`. The validation IoU also falls to `0.70835` versus `0.72382` for E2. This is consistent with distribution drift in the learned fusion projection, not evidence that joint training is impossible under a longer or better calibrated schedule.

### E4: head co-adaptation does not rescue the route

Adding CENTER head updates lowers AUC to `63.46` and Norm Precision to `75.34`. The current short proxy does not support widening the trainable set. A full from-scratch run is not justified by these results and would lose clean attribution.

## Decision

`E1 POSITION: INCONCLUSIVE-BUT-RELEVANT`; `E2 SUPERVISION: SUPPORTS-NEXT`; `E3 JOINT-FUSION: NOT-SUPPORTED`; `E4 JOINT-HEAD: NOT-SUPPORTED`.

The remaining research candidate is a supervised post-fusion reliability/candidate utility route. It should be evaluated with a longer matched training schedule or an OOF utility target before any backbone or head unfreezing. Do not run full-244 for E1, E3, or E4. E2 is only a difficult5, single-seed proxy result and must not be called a robust improvement.

Evidence directories:

- `output/experiments/M1_AMCM_E1_POST_FUSION_PROXY_20261017/`
- `output/experiments/M1_AMCM_E2_TARGET_SUPERVISED_PROXY_20261017/`
- `output/experiments/M1_AMCM_E3_JOINT_FUSION_PROXY_20261017/`
- `output/experiments/M1_AMCM_E4_JOINT_HEAD_PROXY_20261017/`
