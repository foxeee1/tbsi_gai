# Early-Gate Low-Confidence Suppression Quick3

## Question and decision

This matched quick3 asks whether the previous balanced token residual failed because it both attenuated low-confidence template tokens and amplified high-confidence ones. The candidate keeps Early-conditioned MSHA gating and the same RGB/TIR match confidence, but leaves above-median tokens unchanged and suppresses only below-median tokens.

**Decision: do not promote this operator or proceed to full15.** The candidate macro AUC is `-1.72 pp` versus its matched control, with negative changes in all six aggregate metrics. It did recover `shinycarcoming` relative to this run's control, but `ab_bikeoccluded` still collapses (`-28.79 pp` AUC). The signal was active and correctly bounded; absence of activation is ruled out. The result does not establish that conservative reliability modulation is universally harmful because the short quick3 control itself showed large run-to-run tracking variation.

## Prior-sequence attribution

Using preserved paired predictions from the previous quick3, the first box difference above one pixel occurred at frame index 18 on `ab_bikeoccluded`; the first frame with candidate IoU below 0.1 while control remained above 0.5 was index 78 (candidate IoU 0.000, control 0.704). On `shinycarcoming`, box differences began at index 8, with the first severe split at index 203 (candidate IoU 0.067, control 0.646). Both regressions developed after initialization, consistent with closed-loop divergence. The available logs do not bind per-token confidence values to each frame, so the causal role of high-confidence amplification remains unproven.

## Candidate operator

For each modality and template token, compute the existing confidence `c_i` from normalized entropy, top-2 margin, and RGB/TIR agreement. Let `l_i=log(c_i)`, `m=median_i(l_i)`, `d=median_i(|l_i-m|)`, and `u_i=tanh(relu(m-l_i)/max(d,1e-3))`. The candidate scale is `1-0.10u_i`, hence `[0.9,1.0]`: it cannot amplify high-confidence tokens and does not add parameters. The overall cross-modal Value budget may decrease; this is part of the tested mechanism and is recorded.

## Protocol and results

Both arms started from `TBSITrack_SOT_Pretrained.pth.tar`, seed 42, fresh AdamW/StepLR, LasHeR 512 samples/epoch × 3, batch 16, accumulation 8, LR `1e-4`, AMP, worker 4, and 12 optimizer updates. The only runtime difference was `TBSI_EARLY_GATE_TOKEN_REL_MODE=off` versus `suppress_low`. Both used the same reconstructed seven-sequence cohort, FP32, template update off, and serialized arms. Test `threads=1` was used to keep the main process plus one worker within the repository's process cap.

| Metric | Control | Candidate | Delta |
|---|---:|---:|---:|
| AUC | 49.82 | 48.10 | -1.72 pp |
| AO | 51.28 | 48.96 | -2.32 pp |
| Precision | 63.10 | 60.70 | -2.40 pp |
| Normalized precision | 61.35 | 59.23 | -2.13 pp |
| OP50 | 62.95 | 60.28 | -2.67 pp |
| OP75 | 42.31 | 39.06 | -3.25 pp |
| Mean sequence FPS | 60.80 | 52.31 | -14.0% |

The candidate won 3 and lost 4 sequence AUC comparisons. `ab_bikeoccluded` changed by `-28.79 pp`; `shinycarcoming` changed by `+18.19 pp`. The candidate's per-token scale was active: test mean RGB/TIR scale `0.9676/0.9666`, observed minimum `0.9000`, maximum `1.0`; no high-confidence token was amplified. Final validation loss/IoU were `1.58335/0.67177` for candidate and `1.59861/0.66703` for control, while training loss was nearly identical (`1.11088/1.11082`). These small validation differences did not predict the mixed sequence outcomes.

Training diagnostics reported `2/243` zero-gradient parameters in both arms. The new operator adds no trainable parameters, so this count is not specific to the candidate.

## Repeatability and frame attribution

The new control used the same seed, training contract, and pre-change source SHA as the earlier token-reliability quick3 control. Nevertheless, its quick7 AUC was `49.82` versus the earlier control's `56.99`. The two control prediction sets first differed by more than one pixel at frame index 25 on `ab_bikeoccluded` and index 4 on `shinycarcoming`; their mean absolute bbox-coordinate differences were 52.4 and 58.3 pixels. This indicates substantial short-run closed-loop sensitivity and limits attribution from either single-seed quick3 pair.

The training code does initialize Python, NumPy, and PyTorch RNGs in `lib/train/run_training.py`, but `lib/train/train_script.py` explicitly enables `torch.backends.cudnn.benchmark=True`; `TrackingSampler` also randomly selects sequence/frame examples in `__getitem__`. This run did not save the actual sampled-example stream, and the inspected training path does not enforce deterministic algorithms. These are plausible sources of run variation, not a proven attribution. The run-to-run difference could also be small parameter variation amplified by closed-loop tracking.

Within this new pair, the first >1-pixel difference began at index 26 on `ab_bikeoccluded`; the first severe split (candidate IoU <0.1 while control >0.5) occurred at index 220. It still had 325 frames below IoU 0.1 versus 192 for control. On `shinycarcoming`, the severe split occurred at index 196; candidate had 25 frames below IoU 0.1 versus 156 for control. These delayed trajectories are consistent with sequence-specific closed-loop drift, not proof of a token-confidence cause.

The same new control checkpoint was evaluated with `threads=1` and `threads=0`. All 7 sequences had exactly identical predicted box arrays, so the evaluator scheduling difference did not change numerical results. The observed control run-to-run difference is associated with independent short training runs, not test scheduling.

Exact commands, source snapshot, environment, checkpoint hashes, and per-sequence outputs are in `snapshots/m1_early_gate_lowconf_suppress_quick3_20261026/`; training and test logs are in the two run directories under `output/experiments/`.

## Limitations

The seven-sequence manifest is retrospective and development-only. Three epochs give 12 optimizer updates per arm. A quick7 outcome is an exploratory screen, not evidence of stable full-LasHeR performance. The observed control variability makes the paired point estimate provisional, but its direction and severe ab-bike regression do not justify promotion.

## Next decision

Stop expanding this token-reliability branch. Two formulas at the same cross-Value interface have failed to show a reliable paired gain, while independent quick3 control runs vary substantially. First run a repeatability calibration with the same control config/seed and logged sampled sequence/frame IDs, and check whether matched runs reproduce the checkpoint and quick7 predictions. If not, compare a deterministic-kernel setting in a new paired protocol. Then use the existing fixed-state paired replay to test whether any retained candidate changes the cross-Value-to-CENTER chain before spending on another operator iteration.
