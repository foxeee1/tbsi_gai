# Search-channel pilot screening gate

Status: tooling prepared; no P0, training, or tracking run has been performed. The pilot is **未验证基线** and has no performance result.

## Entry order

1. Preserve the current repository state as a commit, branch, or complete code snapshot before model behavior changes.
2. Integrate the adapter only at the verified tensor entering the existing head, behind an off-by-default config option. Do not modify the protected canonical baseline YAML.
3. Run the whole-tracker P0 and write the evidence JSON described below. P0 must pass before any training.
4. RPP-v1 history admission and serial/parallel consistency must already pass. This workspace contains the frozen manifests, but no qualifying admission result was found during setup; until recorded, proxy screening is blocked.
5. If admitted, train C0/C1/C2 under the same frozen RPP-v1 train manifest, full 15-epoch schedule, seed pair 42/3407, parent checkpoint, sample order, and optimizer/scheduler policy. C1/C2 must begin with byte-identical adapter tensors. Export sequence-level C0/C1/C2 and trained C2-with-gamma-disabled metrics on the frozen 60-sequence test list.
6. Run the result gate. Only `ELIGIBLE_FOR_FULL_CONFIRMATION` permits planning a separate full confirmation. It is not itself evidence of a paper-level gain.

The proposed 2-epoch continuation and optional extension to epoch 3 do not match the general RPP-v1 schedule. The CTVM-only 3-epoch exception does not apply to this pilot.

## P0 evidence JSON

`tools/screen_search_channel_pilot_v1.py p0 --input p0.json --output p0_gate.json` expects:

- `parent_checkpoint_loaded`: true; `missing_keys` may contain only `search_channel.*`; `unexpected_keys` must be empty.
- `actual_head_input_verified`, `called_in_train`, and `called_in_serial_inference`: true.
- `gamma_zero_head_max_abs_diff`, `gamma_zero_box_max_abs_diff`, and `template_max_abs_diff`: maximum absolute differences, default tolerance `1e-6`.
- `step2_gradient_norms`: positive finite values for `qkv`, `depthwise`, `projection`, `gamma_rgb`, `gamma_tir` after the second optimizer step.
- `enabled_output_max_abs_diff`: measurable nonzero-gamma effect above tolerance.
- `extra_parameters`, `peak_memory_mb`, `latency_ms`, and `fps`: measured values, not estimates.

## Paired result CSV

`tools/screen_search_channel_pilot_v1.py results --csv per_sequence.csv --rpp-admission admission.json --output gate.json` requires columns `seed,sequence,arm,auc,pr,npr,op50,op75`. `arm` is `C0`, `C1`, `C2`, or `C2_GAMMA_OFF`. Every arm must have the same complete sequence set for both seeds, exactly matching the frozen RPP-v1 test manifest. Metric values use percentage points.

For each seed, C2 must beat C0, C1, and C2 with gamma disabled on mean AUC, median AUC, 10% trimmed mean AUC, and win rate (>50%). The report also includes wins/ties/losses, catastrophic losses (delta <= -5 AUC points), large gains (delta >= +5), and mean PR/NPR/OP50/OP75 for every arm. Both seeds must pass. The admission JSON must record at least four historical anchors, Pearson, direction agreement >=0.75, Spearman >=0.60, and true values for `rank_consistency_passed`, `smsa_above_baseline`, `old_ctvm_negative_or_later`, and `serial_parallel_consistency_passed`.

No training or tracking is launched by this gate script. A failed or incomplete prerequisite is not silently interpreted as a pass.
