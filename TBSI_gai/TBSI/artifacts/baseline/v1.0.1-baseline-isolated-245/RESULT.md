# Isolated Baseline Result

Both evaluation sets completed with 245 prediction files, strict checkpoint loading, clean result directories, and no recorded runtime errors.

| Run | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| Existing v1.0.0 ep15 reference | 54.53 | 66.26 | 45.82 | 68.21 | 64.40 |
| Fresh 15-epoch seed-42 run | 54.42 | 66.06 | 45.79 | 67.89 | 64.09 |
| Fresh minus reference | -0.11 | -0.20 | -0.03 | -0.32 | -0.31 |

## Decision

The fresh run reproduces the currently stored v1.0.0 checkpoint within 0.11 AUC, so the repaired config-isolation and training path are stable enough for controlled comparisons under the current checkpoint contract.

The historical 55.46 result is valid under its original 244-sequence protocol: applying the current, unchanged metric implementation directly to the historical prediction files produces 55.4566 AUC, 67.2762 OP50, 46.7347 OP75, 68.9763 precision, and 65.2298 normalized precision. The metric implementation is therefore not the source of the discrepancy.

The historical prediction files were generated on July 13, whereas the checkpoint currently stored at the v1.0.0 ep15 path was generated on July 30. Re-evaluating that current checkpoint on the same historical 244 sequences gives 54.4680 AUC, and none of its 244 prediction files exactly matches the historical output. The checkpoint that produced 55.46 was overwritten or otherwise lost. Consequently, 54.53 is only the score of the currently stored checkpoint on 245 sequences; it must not be presented as a re-evaluation of the original 55.46-producing model.
