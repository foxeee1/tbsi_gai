# Causal TAM Serial Screen v2 (2026-10-05)

## Execution

Runner: `tools/run_causal_tam_screen_v2.sh`

Output: `output/experiments/causal_tam_screen_v2_20261005/`

The runner verified the protected baseline SHA256, acquired a single-run lock,
ran v2 P0, and then ran the training-history contract audit. It stopped with
exit code `20` at `PRE_G0_STOP`; no G0, short training, or full-244 command was
launched.

## P0 v2

`p0_v2.json` passed all registered checks:

- mixed batch: the non-overlap sample is exact identity while the overlap sample changes;
- conservative same-crop alignment max error `0.0`;
- partial invalid queries are exact identity;
- actual write ratio is bounded at `0.10` for `gamma=0.1/1/5`;
- zero-gate gradient and post-gate Q/K/V/output gradients are finite and nonzero;
- same-frame/future-frame and cross-sequence history are rejected;
- no NaN/Inf.

The adapter now masks invalid history keys per sample, clears invalid queries
after the biased output projection, uses conservative token-center coverage,
and limits the actual `tanh(gamma)` write rather than the pre-gate message.

## Pre-G0 decision

`real_batch.json` remains `STOP_TRAINING_HISTORY_CONTRACT`. The actor/model
interface now carries explicit frame and sequence provenance and derives the
previous raw head tokens from `prev_search` under `no_grad`. The remaining
failure is substantive: the sampler has no frozen-M1 actual crop trace, so
current/history crop rectangles are unavailable. Consequently no real
training batch can currently prove that TAM learns from the same deployed crop
contract. Changing the search count or starting a one-epoch run would be
invalid under this contract.

## Decision

`STOP_BEFORE_G0`: generate and freeze a training-side M1 trajectory trace with
actual predicted-state crops, then run the three-batch audit. Preserve the
current P0 and single-sequence tracker parity assets; do not train TAM or run
full-244 from this state.

Canonical baseline YAML remains unchanged with SHA256
`4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fbf83d696538251a2d8fadd24b`.
