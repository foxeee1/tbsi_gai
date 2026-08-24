# v5 Fixed Rerun Manifest

Started: 2026-08-02T01:51:52Z

## Purpose

Re-run the v5 CUTR-Lite router-only experiment after fixing experiment-system bugs that could invalidate earlier negative results.

## Config

- Config: `experiments/tbsi_track/v5.0.1-cutr-lite-routeronly-fixed.yaml`
- Baseline checkpoint: `output/experiments/v1.0.0-完美基线-全量/checkpoints/TBSITrack_ep0015.pth.tar`
- Baseline reference metric: LasHeR test AUC `55.46`
- Output directory: `output/experiments/v5.0.1-cutr-lite-routeronly-fixed`

## Fixed Before Run

- Test result reuse is opt-in via `--resume_results`; default runs no longer skip existing sequence outputs.
- Test checkpoint loading is strict by default.
- Stage-2 and CUTR-Lite baseline loads now fail on unexpected checkpoint/config mismatch.
- Router-only mode now fails if no router parameters are trainable.
- Stage-2 optimizer now refuses empty trainable sets and drops empty param groups.
- Training no longer silently replaces failed GIoU with zero unless explicitly allowed.

## Sanity Check

- Trainable parameters: `utility_router.*` only.
- Trainable tensors: `6`.
- Trainable scalars: `300,945`.
- Optimizer groups: one group, LR `0.0005`.

## Train Command

```bash
python -u lib/train/run_training.py \
  --script tbsi_track \
  --config v5.0.1-cutr-lite-routeronly-fixed \
  --save_dir ./output \
  --use_lmdb 0 \
  --use_wandb 0 \
  --seed 42
```

## Planned Evaluation

After checkpoint `TBSITrack_ep0006.pth.tar` is produced, run full `lasher_test` with default non-resume behavior and compare against baseline AUC `55.46`.

## Result

- Checkpoint: `output/experiments/v5.0.1-cutr-lite-routeronly-fixed/checkpoints/TBSITrack_ep0006.pth.tar`
- Test split: full `lasher_test`, `244/244` sequences.
- Result directory: `output/test/tracking_results/tbsi_track/v5.0.1-cutr-lite-routeronly-fixed`
- Eval log: `artifacts/rerun_v5_fixed/eval_lasher_ep0006.log`

| Tracker | AUC | OP50 | OP75 | Precision | Norm Precision |
|---|---:|---:|---:|---:|---:|
| baseline `v1.0.0-完美基线-全量` | 55.46 | 67.28 | 46.73 | 68.98 | 65.23 |
| `v5.0.1-cutr-lite-routeronly-fixed` | 54.51 | 66.23 | 45.65 | 68.32 | 64.37 |

Delta vs. baseline:

- AUC: `-0.95`
- OP50: `-1.05`
- OP75: `-1.08`
- Precision: `-0.66`
- Norm Precision: `-0.86`

Conclusion: CUTR-Lite router-only remains negative after the experiment-system fixes. This weakens v5 as a publishable route, but it does not invalidate the bug diagnosis because the fixed harness did expose cleaner training/test behavior and strict parameter auditing.
