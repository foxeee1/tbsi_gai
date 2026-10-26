# M1 adaptive post-fusion router P0 quick tracking comparison (2026-10-13)

## Contract

The protected M1 epoch-15 checkpoint and the P0 quick checkpoint were evaluated with the same FP32 tracker entrypoint, the same 20-sequence LasHeR normal-control manifest, and one tracker process. P0 was trained for one tiny proxy epoch on 16 train samples and is therefore an engineering/proxy result, not a full-training claim.

| Condition | AUC | PR | NPR | OP50 | OP75 | AUC win/loss/tie |
|---|---:|---:|---:|---:|---:|---:|
| M1 epoch-15 | 85.83 | 98.42 | 98.78 | 98.93 | 94.65 | baseline |
| P0 quick router | 85.84 | 98.45 | 98.83 | 98.96 | 94.62 | 7/5/8 |
| Difference | +0.01 | +0.03 | +0.05 | +0.03 | -0.03 | exploratory |

The P0 per-sequence differences are very small: 7 wins, 5 losses, and 8 ties after the evaluator's displayed precision. This is consistent with a lightly trained, bounded residual on already-good sequences. It does not establish improvement, and no full-244 or difficult-sequence conclusion is allowed from this run.

Artifacts:

- M1 results: `output/test/tracking_results/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42_9902/`
- P0 results: `output/test/tracking_results/tbsi_track/M1_ADAPTIVE_POST_ROUTER_P0_QUICK_seed42_9903/`
- P0 checkpoint: `output/experiments/M1_ADAPTIVE_POST_ROUTER_P0_QUICK_V2_20261013/experiments/M1_ADAPTIVE_POST_ROUTER_P0_QUICK_seed42/checkpoints/TBSITrack_ep0001.pth.tar`

## Decision

`P0 quick` passes the basic non-degradation smoke on this 20-sequence normal proxy, but the result is inconclusive. The next meaningful comparison must train P0 with the intended utility/no-op supervision and evaluate both the held-out difficult cohort and a normal-control cohort.

