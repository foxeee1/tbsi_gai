# Baseline Reproduction Manifest

## Objective

Verify that the isolated, enhancement-off TBSI implementation reproduces the canonical baseline on the official 245-sequence LasHeR test protocol.

## Run Contract

- Reference checkpoint: `output/experiments/v1.0.0-完美基线-全量/checkpoints/TBSITrack_ep0015.pth.tar`
- Fresh initialization: `pretrained_models/TBSITrack_SOT_Pretrained.pth.tar`
- Reference config: `v1.0.1-baseline-ep15-reference-245`
- Fresh config: `v1.0.1-baseline-isolated-245`
- Seed: 42
- Training: 15 epochs, 60,000 samples per epoch, batch size 32
- Evaluation: LasHeR test, exactly 245 sequences
- Required metrics: AUC, normalized precision, precision, OP50, OP75
- Historical comparator: 55.46 AUC, but that result used 244 sequences and is not protocol-identical

## Integrity Gates

1. Every experiment load starts from a deep-copied default config.
2. The all-off bridge must exactly match the canonical serial TBSI path.
3. Strict checkpoint loading must report no missing or unexpected keys.
4. Each completed evaluation must contain exactly 245 prediction files.
5. Reference and freshly trained results use distinct parameter names and cannot overwrite each other.

## Storage Policy

Only the final fresh ep15 checkpoint is retained because the workspace has approximately 12 GB free. Logs, per-sequence outputs, and both analysis reports are retained.
