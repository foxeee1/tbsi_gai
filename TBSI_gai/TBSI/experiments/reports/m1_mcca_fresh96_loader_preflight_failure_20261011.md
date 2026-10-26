# MCCA Fresh96 Loader Preflight Failure (2026-10-11)

- Parent: `M1_MCCA_G0_G1_20261010`, fixed G0/G1 epoch-3 checkpoints.
- Intended slice: evaluate G0, G1-full, G1 TIR-to-RGB-only, and G1 RGB-to-TIR-only on the frozen 96-sequence train-side attribution cohort.
- Failure: the first runner used `rpp_lasher_test`, whose dataset class correctly resolves under `testingset`; the cohort belongs to `trainingset`. Loader stopped at missing `/root/autodl-tmp/TBSI_gai/data/lasher/testingset/biketonorth/init.txt` before any sequence or tracker frame ran.
- GPU work: none. Prediction count: 0. No checkpoint or existing run output was overwritten.
- Resolution: add a dedicated `mcca_lasher_train_eval` dataset entry bound to `TBSI_MCCA_EVAL_LIST` and `trainingset`; retry in a versioned `_V2` run directory.
- Status: `FAILED-PREFLIGHT / DATASET-SPLIT-ROOT`; retained as a protocol failure, not a model result.
