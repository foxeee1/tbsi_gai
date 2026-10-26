# TAM trace-backed training smoke v1 (2026-10-07)

## Scope

This is a bounded training-contract smoke test, not full-data TAM training and
not a tracking evaluation. It uses the existing frozen M1 temporal trace from
three LasHeR train sequences, with seed 42, batch size 4, and 100 optimizer
steps. The original M1 epoch-15 checkpoint and CenterPredictor head are used;
only `causal_tam` parameters are optimized.

Command:

```bash
bash tools/run_causal_tam_trace_smoke.sh
```

Output: `output/experiments/tam_trace_smoke_v1_20261007/`

## Result

- decision: `TAM_TRACE_SMOKE_PASS`;
- 100/100 optimizer steps completed;
- all loss/output/gradient values finite;
- same-sequence rate: `1.0`;
- causal ordering: passed for every batch;
- gamma: `0.000000` initial to `0.0096834` final;
- maximum p95 residual ratio: `0.0268552`;
- maximum residual ratio: `0.0415127`, below the `0.10` cap;
- non-TAM state digest unchanged;
- checkpoint save/restore passed.

The first loss was `1.1571` and the last was `2.9650`; the trace-backed batch
loss is visibly variable and did not establish optimization improvement. This
run therefore supports training-path stability only, not TAM tracking benefit.

## Failure and repair

The first invocation reached the first batch but failed while assembling the
diagnostic record because `actual_ratio_mean` was omitted from the actor return
dictionary. The field was added and the experiment was rerun from the original
M1 checkpoint. The initial partial artifact is retained as a tooling failure.

## Boundary

This smoke test does not authorize full-244. The actual online full-data
sampler still requires a separate history/crop implementation and validation.
The three-sequence trace is not a training set and no tracking result is
claimed.
