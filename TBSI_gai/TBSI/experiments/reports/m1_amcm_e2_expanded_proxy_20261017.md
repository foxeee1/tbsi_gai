# M1 AMCM E2 expanded proxy (2026-10-17)

## Protocol

E2 post-fusion AMCM and target-cell RGB/TIR similarity supervision were kept unchanged. Training used the protected M1 epoch15 checkpoint, LasHeR, batch 2, FP32, worker 1, 5 epochs and 512 samples per epoch. The first attempted 1024-sample schedule stopped at epoch 1 during checkpoint writing because of filesystem space; it produced no usable checkpoint and is retained as a failed execution record. The completed run saved only epoch 5.

Testing used the first 30 sequences of the fixed LasHeR test ordering, FP32 serial evaluation, with the same search/template parameters. This is an expanded 30/244 proxy, not a full-test result.

## Result

| Scope | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| E2 difficult5, 3 epoch | 66.70 | 78.34 | 54.05 | 88.13 | 78.58 |
| E2 expanded first30, 5 epoch | 49.50 | 61.71 | 41.83 | 63.92 | 60.07 |

The expanded result is substantially below M1 difficult5 and is not directly comparable to the difficult5 result because both the sequence scope and training schedule changed. It does establish that the short E2 proxy result does not yet generalize to this broader fixed subset under the expanded schedule.

## Status

`EXPANDED-PROXY-COMPLETE / NOT-SUPPORTED-FOR-PROMOTION`. Do not run full-244 E2 or unfreeze fusion/head based on this result alone. The next diagnostic should determine whether the drop is caused by sequence distribution, five-epoch overfitting or checkpoint selection, or the similarity target itself.

Evidence:

- `experiments/tbsi_track/M1_AMCM_E2_FULL244_20261017.yaml`
- `output/experiments/M1_AMCM_E2_FULL244_20261017/`
- `output/test/tracking_results/tbsi_track/M1_AMCM_E2_FULL244_20261017/` (30/244 completed)
