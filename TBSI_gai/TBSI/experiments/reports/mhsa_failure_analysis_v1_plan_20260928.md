# MHSA Failure Analysis v1

## Run contract

- Research question: are M1 regressions relative to the canonical tracker concentrated in measurable cross-modal or spatial failure conditions?
- Parent: `TBSI-E-MHSA-FULL-20260927`.
- Control: protected canonical `v1.0.0-单卡3090-48G-等效全局128基线` prediction directory, read-only.
- M1: `TBSI_E_MHSA_FULL_4090D_seed42`, seed 42, full LasHeR test predictions, read-only.
- Stage A scope: strict common-sequence/frame pairing and fixed failure labels. No training and no new model behavior.
- Required outputs: frame-level pairs, sequence-level aggregation, immutable input manifest, and a durable command log.
- Stop condition: abort if sequence names or frame counts cannot be aligned without silently truncating or remapping frames.

## Fixed labels

`hard_regression <= -0.10`, `regression <= -0.05`, `gain >= 0.05`, and `common_failure` when both IoUs are below `0.30`; thresholds are fixed before inspection.

## Next gates

After Stage A, add attention statistics only behind a disabled-by-default diagnostic gate and verify zero-difference parity. The first full diagnostic pass uses all 244 test sequences and a deterministic prefix of at most 64 frames per sequence to bound long-sequence cost; it is mechanism evidence, not a tracking metric rerun. Do not train a replacement module until the sequence-level analysis supports one of H1/H2/H3.

Current resource note: disk availability was approximately 3 GB on 2026-09-28, below the RPP-v1 100 GB safety threshold; no RPP training or full rerun is authorized by this run.
