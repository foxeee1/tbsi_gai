# v1.0.0 Legacy-244 Baseline Rerun

## Contract

- Config: `experiments/tbsi_track/v1.0.0-完美基线-全量.yaml`
- Initialization: `pretrained_models/TBSITrack_SOT_Pretrained.pth.tar`
- Seed: 42
- Training: 15 epochs, 60,000 samples per epoch, batch size 32
- Evaluation: historical LasHeR 244-sequence protocol
- Expected historical target: 55.46 AUC
- Required metrics: AUC, OP50, OP75, precision, normalized precision
- Result reuse: disabled
- Checkpoint loading: strict

Intermediate checkpoint persistence is disabled through `TBSI_FINAL_CHECKPOINT_ONLY=1` because less than 10 GB was free at launch. This changes storage only; the optimizer and training schedule are unchanged.

## Preserved Control

- Path: `output/experiments/v1.0.0-完美基线-全量/archive/pre-rerun-20260810/TBSITrack_ep0015-20260730.pth.tar`
- SHA-256: `8421da771831a63522fb29ad5e53393e4ccea7dae2e4db23dca00a5e24b15ce5`
- Size: 2,428,320,243 bytes
- Historical 244-sequence predictions remain in `output/experiments/v1.0.0-完美基线-全量/test_results`.
