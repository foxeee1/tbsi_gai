# M1 CENTER rank loss full training and test

## Pre-registered contract

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch15, seed42.
- Candidate: the same CENTER pairwise ranking loss that was positive on the
  fixed 20-sequence proxy: weight `0.10`, margin `0.05`, eight hard negatives.
- Training: full LasHeR training set, 59,904 samples/epoch, 15 epochs,
  batch16 with gradient accumulation8 (effective batch128), FP32/AMP setting
  from the parent, and `NUM_WORKER=1` because the protected worker4 setting
  exceeds the current global process budget. This is therefore a matched
  method run but not a strict M1 reproduction.
- Test: LasHeR official testingset, all 244 sequences, serial FP32 tracker,
  explicit epoch15 checkpoint.
- Primary metrics: AUC/Success, Precision, Norm Precision, OP50, OP75, FPS;
  report all 244 per-sequence results and subgroup deltas against M1.
- Stop/decision: do not add gates or tune the ranking loss after launch. If
  full-244 does not preserve M1 on normal/remaining sequences, do not promote
  the method despite proxy gains.

## Evidence paths

- Config: `experiments/tbsi_track/M1_CENTER_RANK_LOSS_FULL15_W1_seed42_20261016.yaml`
- Training output: `output/experiments/M1_CENTER_RANK_LOSS_FULL15_W1_seed42_20261016/`
- Full test output: `output/test/tracking_results/tbsi_track/M1_CENTER_RANK_LOSS_FULL15_W1_seed42_20261016/`
- Parent checkpoint: protected M1 epoch15; never overwritten.

Status at launch: `TRAINING`; stopped at the user's request on 2026-10-17.

Final run state: **interrupted during epoch 12 at step 1,500/3,744**. The
training and hourly monitor processes were terminated; no tracking test was
running or started. The configured save point was epoch 15, and the checkpoint
directory is empty, so this run has no candidate checkpoint and no full-test
metrics. Training logs and diagnostics remain in the output directory. The
hourly monitor log stopped updating earlier than the training log, so it is
not evidence of continuous hourly monitoring.

Conclusion: `INCONCLUSIVE / USER-REQUESTED-INTERRUPTION`. The proxy20 gain was
not confirmed on full training or full-244 testing; do not compare this run's
training losses with M1 tracking performance or treat it as a negative method
result.
