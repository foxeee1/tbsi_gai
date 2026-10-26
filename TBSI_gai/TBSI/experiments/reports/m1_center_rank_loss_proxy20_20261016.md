# M1 CENTER candidate ranking loss proxy20

## Question

Can a small train-side pairwise ranking term improve the discrete CENTER
selection contract without changing the fusion operator, regression heads, or
state update?

## Intervention and contract

The original focal, GIoU, and L1 losses remain active. For each training
sample, the cell with the maximum Gaussian target is treated as positive; the
eight highest scoring cells outside the target region are hard negatives. The
additional loss is

`L_rank = 0.10 * mean(relu(0.05 - score_positive + score_hard_negative))`.

Training used 3 epochs, 256 samples/epoch, batch 2, FP32, one DataLoader
worker, initialized from protected M1 epoch15. Evaluation used 20 fixed
test sequences: the previous difficult5 plus 15 additional sequences selected
before training. No sequence-specific threshold was fitted.

## Overall result

| Method | AUC / Success | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| M1, same 20 sequences | 44.36 | 52.48 | 33.43 | 57.42 | 50.50 |
| CENTER rank loss, epoch3 | 47.05 | 56.43 | 36.97 | 60.88 | 53.94 |
| Delta | +2.69 | +3.95 | +3.54 | +3.46 | +3.44 |

These are 20-sequence proxy metrics and cannot replace full-244 results.

## Sequence-level comparison

| Sequence | M1 AUC | Rank AUC | Delta |
|---|---:|---:|---:|
| manoncall | 70.07 | 67.09 | -2.98 |
| boyss | 54.56 | 51.93 | -2.63 |
| girldownstairfromlight | 47.12 | 53.43 | +6.32 |
| bike2left | 81.36 | 82.70 | +1.34 |
| girlinrain | 86.15 | 84.86 | -1.29 |
| boyaroundtrees | 54.84 | 59.26 | +4.42 |
| midboyNo_9 | 70.09 | 69.91 | -0.18 |
| catbrownback2bush | 78.34 | 79.07 | +0.73 |
| rightbluewhite | 53.41 | 64.32 | +10.91 |
| midrunboywithwhite | 46.14 | 43.94 | -2.19 |
| midgreyboyrunningcoming | 55.63 | 55.42 | -0.22 |
| boyunder2baskets | 12.50 | 13.13 | +0.63 |
| midredboy | 67.46 | 67.28 | -0.18 |
| rightbottlecomes | 29.49 | 32.15 | +2.66 |
| lastleftgirl | 9.18 | 9.85 | +0.67 |
| 2girlup | 8.07 | 8.32 | +0.25 |
| 3rdfatboy | 20.45 | 54.09 | +33.64 |
| bottlebetweenboy`sfeet | 10.13 | 11.52 | +1.38 |
| bikeboyintodark | 50.28 | 50.71 | +0.42 |
| girl`sblkbag | 7.05 | 7.51 | +0.46 |

The candidate wins 13/20 sequences by AUC and loses 7/20. The largest gains
are `3rdfatboy`, `rightbluewhite`, and `girldownstairfromlight`; the two
known severe sequences `manoncall` and `boyss` still regress. Thus the loss
improves average candidate ranking across varied scenes, but does not yet
solve the remaining162 failure class.

## Decision

`DIAGNOSTIC-COMPLETE / PROXY-POSITIVE-BUT-NOT-PROMOTED`. This is the first
train-side correction objective to show a positive direction on a 20-sequence
diverse screen. It is not sufficient for full training because it is one seed,
3 epochs, and has two important regressions. The next experiment should keep
the same loss and run a matched longer proxy or full training only after a
20-sequence control confirms the ranking term itself is the changed variable.
Do not add state gates or Hann loss to this candidate.

## Provenance

- Config SHA256: `100c7e41f3ce1dfc967dbd8a5fc4688f6783ce624156660c0896ee35c7f93270`
- Manifest SHA256: `0955ddb3cf7e17ef0aaec6b87f0b8cfa504d458937a3e013da54ad76f9afb10e`
- Checkpoint SHA256: `9182e56b48562e51b64498eb370868fabdc51c36bb8cf405170db51ad0c9e8d5`
- Training output: `TBSI_gai/TBSI/output/experiments/M1_CENTER_RANK_LOSS_PROXY20_20261016/`
- Tracking output: `TBSI_gai/TBSI/output/test/tracking_results/tbsi_track/M1_CENTER_RANK_LOSS_PROXY20_20261016/`

## Full-244 follow-up evaluation (2026-10-17)

The epoch-3 proxy checkpoint is being evaluated on the complete official
LasHeR `lasher_test` split as requested. This is a fixed-checkpoint test of a
model trained on only 256 samples/epoch for 3 epochs; it does not complete or
replace the interrupted 15-epoch full-training experiment, and it must not be
described as a full-training result.

- Checkpoint: `output/experiments/M1_CENTER_RANK_LOSS_PROXY20_20261016/checkpoints/TBSITrack_ep0003.pth.tar`, SHA256 `9182e56b48562e51b64498eb370868fabdc51c36bb8cf405170db51ad0c9e8d5`.
- Config: `experiments/tbsi_track/M1_CENTER_RANK_LOSS_PROXY20_20261016.yaml`, SHA256 `100c7e41f3ce1dfc967dbd8a5fc4688f6783ce624156660c0896ee35c7f93270`.
- Dataset: `lasher_test`, 244 unique sequences after the dataset's registered corrupt-sequence exclusions. Resolved manifest: `output/experiments/M1_CENTER_RANK_LOSS_PROXY20_FULL244_20261017/test_manifest.json`, SHA256 `3e57025b08a4b8eff4183aece4eb31f913028b282411ad6a0761c946590c6189`; source `testingsetList.txt` SHA256 `16eef326975aacccb94a7e133e8ef0aceb734ece4264caf6f82bc94442c20e46`.
- Evaluation: FP32, 1x NVIDIA GeForce RTX 4090 D, serial tracker (`threads=0`), run ID `20261017`; explicit checkpoint override to the hash-verified epoch-3 proxy checkpoint. Results are isolated under `output/test/tracking_results/tbsi_track/M1_CENTER_RANK_LOSS_PROXY20_20261016_20261017/`.
- Environment: Git `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`; Python 3.8.20, PyTorch 1.9.0+cu111, CUDA 11.1.
- Test command: `TBSI_INFER_FP16=0 TBSI_CHECKPOINT_OVERRIDE=output/experiments/M1_CENTER_RANK_LOSS_PROXY20_20261016/checkpoints/TBSITrack_ep0003.pth.tar python -u tracking/test.py tbsi_track M1_CENTER_RANK_LOSS_PROXY20_20261016 --dataset_name lasher_test --threads 0 --num_gpus 1 --runid 20261017`.
- Status: `RUNNING`; command output is captured at `output/experiments/M1_CENTER_RANK_LOSS_PROXY20_FULL244_20261017/full244_test.log`. Metrics will be added after 244/244 prediction files are verified.
