# M1 Spatial Token Bridge Output-Amplitude Diagnostic (2026-10-24)

## Question

The E10 bridge-only fit increased the learned residual, but target-conditioned replay still left CENTER and query ranks unchanged. This diagnostic asks whether a larger version of the same learned message exposes a useful target-specific signal, or whether the message direction/correspondence itself is not useful to localization under the current insertion point.

This is an auxiliary mechanism diagnostic on LasHeR train-side calibration states. It is not full tracking evaluation.

## Contract and execution

- Parent: `M1_SPATIAL_TOKEN_BRIDGE_LEARNABILITY_10EP_20261024`, epoch 10; checkpoint SHA256 `eecceb40e3bd273d16afe6c4638154d92d189f1334caef3a55c08fbdc5e125cf`.
- Data: the same 23 sequences and 46 saved fixed states used by the E10 replay. Each state preserves its original frame, crop, tracker state, GT box, and M1 trace IoU.
- Single intervention: scale the bridge's computed RGB/TIR output deltas after `SpatialTokenBridge.forward`. No weights, selected token indices, correspondence parameters, tracker state, or metrics were changed.
- Arms: bridge off; target and equal-density shifted correspondence at `1x`, `32x`, and `128x`.
- Evaluation: FP32, serial, one tracker process, torch threads 1; RTX 4090 D; Python 3.8.20, PyTorch 1.9.0+cu111, CUDA 11.1.
- First attempt: `M1_SPATIAL_TOKEN_BRIDGE_AMPLITUDE_FIXED46_20261024` stopped before model initialization because the generated event file was per-state but the runner expects grouped pre/onset fields per sequence. It evaluated zero states; the failure log is retained.
- Retry: `M1_SPATIAL_TOKEN_BRIDGE_AMPLITUDE_FIXED46_RETRY1_20261024` completed all 7 arms and 322 states.
- Executed source snapshot SHA256: `f48ad151b23eae209b83ead8fa7cb70da815dbfae7a8052816ab018d64d3329`.

## Validity check

All 46 event IDs matched the parent replay. The off, target-1x, and shifted-1x arms reproduced all three measured values in the preceding E10 replay exactly at recorded precision: mean and maximum absolute error were zero for CENTER margin, post-fusion query margin, and replay IoU. The dose-response comparison is therefore aligned to the earlier replay contract.

## Results

Table entries are target-minus-shifted paired means over the same 46 states. Rank changes are counts of states where the integer rank differed; the better/worse direction is stated separately.

| Scale | CENTER GT margin | Query GT margin | Replay IoU | CENTER rank changes | Query rank changes |
|---:|---:|---:|---:|---:|---:|
| 1x | `-1.308e-6` | `-1.046e-6` | `-1.183e-6` | 0/46 | 0/46 |
| 32x | `-4.272e-5` | `-3.475e-5` | `-0.01021` | 1/46, target worse | 0/46 |
| 128x | `-1.996e-4` | `+5.228e-4` | `-0.02054` | 3/46, 1 better / 2 worse | 2/46, 1 better / 1 worse |

At 128x, target-minus-shifted sequence-mean signs were mixed: CENTER margin improved on 9/23 sequences, query margin on 9/23 (2 ties), and replay IoU on 7/23 (the remaining 16 were negative). The preregistered `>1e-3` localization-margin threshold was not met; CENTER margin and replay IoU were negative on average. The strongest replay-IoU regressions were concentrated in `bike150` (`-0.2375`) and `theleftestrunningboy` (`-0.2348`); this small, selected calibration cohort does not support attributing a general scene cause.

The scale-32 and scale-128 target arms also reduced replay IoU against off by `-0.01017` and `-0.01011`; shifted arms changed replay IoU by about `+0.00004` and `+0.01043` against off. Thus the large-dose effect is not a target-specific gain: the shifted control is better on replay IoU at the same scale. At scale 128, target improves the query margin relative to shifted during severe-onset states (`+0.00126`), but this does not carry into CENTER margin (`-0.00072`) or replay IoU (`-0.02048`) on those same states. This is direct evidence of a query-to-decision transfer failure for this intervention.

The 32x/128x multipliers are stress doses applied after the bridge's internal per-token delta clip. They intentionally test whether signal direction can affect readouts; they are not valid deployment settings and may exceed the configured bridge output bound.

## Conclusion

**Conclusion: amplitude-only explanation is refuted for the current bridge; stop scaling this operator.** The native `1x` message is too small to alter ranks, but increasing it does make the network respond. Those responses are not target-specific localization improvements: target does not beat shifted on replay IoU, worsens replay at larger doses, and the one positive high-dose query-margin contrast fails to improve CENTER selection or boxes. The data point toward a correspondence/content or insertion/readout mismatch, not merely insufficient message magnitude.

This does not rule out all spatial correspondence fusion. It rejects the claim that multiplying this learned bridge message is a sufficient fix. No full training or benchmark evaluation was run; AUC, PR, NPR, OP50, OP75, and FPS are `NA`.

## Next action

Do not full-train or continue amplitude sweeps for this bridge. Keep the source, checkpoints, first-attempt error, seven paired outputs, and summaries as experimental assets. A future Candidate 2 revision would need to change the correspondence/content construction and pre-register a readout-level prediction; another scale or longer fit would not distinguish a remaining hypothesis.

## Artifacts

- Paired summary: `output/experiments/M1_SPATIAL_TOKEN_BRIDGE_AMPLITUDE_FIXED46_RETRY1_20261024/paired_summary.json`
- Per-state contrasts: `output/experiments/M1_SPATIAL_TOKEN_BRIDGE_AMPLITUDE_FIXED46_RETRY1_20261024/per_state_contrasts.csv`
- Per-sequence contrasts: `output/experiments/M1_SPATIAL_TOKEN_BRIDGE_AMPLITUDE_FIXED46_RETRY1_20261024/per_sequence_contrasts.csv`
- Commands and logs: `output/experiments/M1_SPATIAL_TOKEN_BRIDGE_AMPLITUDE_FIXED46_RETRY1_20261024/commands_and_logs.txt`
- Protocol: `experiments/protocols/m1_spatial_token_bridge_amplitude_20261024/`
