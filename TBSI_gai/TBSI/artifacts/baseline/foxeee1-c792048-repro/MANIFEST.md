# foxeee1 / c792048 Baseline Reproduction

## Source

- Git commit: `c79204832e6ec1a99665114f5ab87d6bdb595ea3`
- Commit date: 2026-07-15 13:59:31 +0800
- Isolated worktree: `/root/autodl-tmp/foxeee1_repro`
- Config: `experiments/tbsi_track/完美基线全量.yaml`
- Model source: `/root/autodl-tmp/TBSI_gai/TBSI/pretrained_models/TBSITrack_SOT.pth.tar`
- Seed: 42
- Dataset protocol: commit-native LasHeR legacy 244 sequence list
- Runtime target: `/root/autodl-tmp/conda_envs/tbsi` (`Python 3.8.20`, `PyTorch 1.9.0+cu111`, CUDA 11.1, cuDNN 8005)

## Scope

This run reconstructs the exact commit-level baseline contract. The current repository's later model, actor, optimizer, configuration-isolation, or evaluator changes are not imported into the worktree.

The source commit does not contain `lib/train/data/`: that directory is ignored by the repository's `data` rule. The worktree therefore receives the existing local `lib/train/data/` package as an environment dependency; its code is not treated as a c792048 commit change. This is a commit-model-strict reproduction with an explicitly recorded local-data-package caveat, not a self-contained archive reproduction.

One storage-only guard, `FOXEEE1_FINAL_CHECKPOINT_ONLY=1`, suppresses intermediate ep3/6/9/12 checkpoint persistence. It does not change the data sampler, loss, optimizer, learning-rate schedule, or model computation.

## Expected Outputs

- Checkpoint: `output/experiments/完美基线全量/checkpoints/TBSITrack_ep0015.pth.tar`
- Results: `output/test/tracking_results/tbsi_track/完美基线全量`
- Metrics: AUC, OP50, OP75, precision, normalized precision
- Historical reference: AUC 55.46, OP50 67.28, OP75 46.73, precision 68.98, normalized precision 65.23
