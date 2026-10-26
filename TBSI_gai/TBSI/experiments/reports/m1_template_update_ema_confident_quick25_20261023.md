# M1 confident template EMA quick comparison (2026-10-23)

## Question

Can a conservative inference-time template update recover M1 failure events without changing the fusion model?

## Contract

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42` epoch15.
- Dataset: the fixed M1 25-event LasHeR proxy manifest.
- Inference: FP32, same `run_m1_candidate_closed_loop_20261020.py` entry, same event prefixes.
- Candidate: `TBSI_TEMPLATE_UPDATE_MODE=ema_confident`, alpha `0.10`; update only when response peak margin >= `0.08`, normalized center shift <= `0.50`, and relative scale change <= `0.30`.
- Control: identical current code entry with `TBSI_TEMPLATE_UPDATE_MODE=off`.
- No training, no weight changes, no locked holdout.

## Results

| Comparison | Events | Mean event IoU delta | Wins | Losses |
|---|---:|---:|---:|---:|
| EMA update vs current off, failure | 20 | -0.01389 | 8 | 6 |
| EMA update vs current off, control | 5 | -0.00461 | 2 | 3 |
| EMA update vs current off, all | 25 | -0.01203 | 10 | 9 |

The update was accepted on 1,392/1,641 tracked frames (84.83%). Large event-level recoveries occurred on `leftboy2jointhe4` (+0.8080) and `leftunderbasket` (+0.1029 versus the same-entry control), but large regressions occurred on `basketball849` (-0.5445), `rightgirltakingcup` (-0.3303), and `middrillmaster` (-0.1845).

## Decision

`QUICK-COMPLETE / TEMPLATE-UPDATE-NOT-SUPPORTED / STOP-CURRENT-GATE`.

The earlier comparison against stored M1 result text showed a positive proxy delta, but the current off control differs from that archived replay. That comparison is therefore not causal evidence. The same-entry paired result is the valid comparison and is negative. The high acceptance rate also shows that the gate is not selective enough for a recovery mechanism; it can write a contaminated crop into the template before the tracker has recovered.

## Evidence

- Candidate (final-source retry2): `output/experiments/M1_TEMPLATE_UPDATE_EMA_CONFIDENT_QUICK25_RETRY2_20261023/closed_loop_25/`
- Candidate audit: `output/experiments/M1_TEMPLATE_UPDATE_EMA_CONFIDENT_QUICK25_RETRY2_20261023/template_update_audit/`
- Control: `output/experiments/M1_TEMPLATE_UPDATE_OFF_CONTROL_QUICK25_20261023/closed_loop_25/`
- Source hash after feature-gated change: `lib/test/tracker/tbsi_track.py` SHA256 `efedfd1e1dc85a9b119069e116eb3695d878810fad2dba22dad5baf7985c60d2`

The first candidate directory is retained as an audit artifact. Retry2 used the final source after correcting the normalized template-box coordinates; its paired metrics and acceptance rate are identical.

## Next action

Do not couple this template updater to AMCM yet. A follow-up should first use fixed-state or held-out sequence evidence to learn a genuinely selective update admission rule, or stop the template-update branch if no pre-update signal separates recoverable from contaminating frames.
