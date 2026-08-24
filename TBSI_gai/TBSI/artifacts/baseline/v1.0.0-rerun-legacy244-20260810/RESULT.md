# v1.0.0 Legacy-244 Rerun Result

Training completed for 15 epochs and produced a new ep15 checkpoint. Testing completed on all 244 legacy-protocol sequences.

| Run | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| Historical July 13 predictions | 55.46 | 67.28 | 46.73 | 68.98 | 65.23 |
| August 10 v1.0.0 rerun | 54.13 | 65.75 | 45.19 | 67.69 | 63.68 |
| Rerun minus historical | -1.33 | -1.53 | -1.54 | -1.29 | -1.55 |

## Artifact Integrity

- New ep15 SHA-256: `98a1531b216ede70d4ce63f472259bb4c7139fa10c8ebf0a55e81c58e95720d2`
- Preserved July 30 ep15 SHA-256: `8421da771831a63522fb29ad5e53393e4ccea7dae2e4db23dca00a5e24b15ce5`
- Historical predictions were restored byte-for-byte from Git after the legacy result-directory symlink caused the first test command to overwrite their working-tree copies.
- New predictions are preserved separately in `output/experiments/v1.0.0-完美基线-全量/test_results_rerun_20260810`.

## Interpretation

The historical 55.46 score cannot be reproduced from the current training code and initialization contract by rerunning the original YAML alone. The missing July 13 checkpoint or an unrecorded training-code/environment difference remains necessary to explain the gap.
