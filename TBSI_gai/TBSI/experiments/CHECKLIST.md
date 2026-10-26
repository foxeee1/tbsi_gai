# MCCA G0/G1 Checklist

- [x] Read repository experiment rules and current M1/DBF/MHSA evidence.
- [x] Verify protected M1 config/checkpoint hashes.
- [x] Freeze RPP-v1 train176/test60 manifests and hashes.
- [x] Preserve pre-change working source snapshot.
- [x] Implement MCCA behind default-off feature gate.
- [x] P0: disabled and zero-output parity, checkpoint-key contract, full-ViT post-init zero output, two-step gradients and updates.
- [x] Create independent matched G0/G1 configs and serial runner.
- [x] Record both runs as RUNNING before launch.
- [x] Run G0 3-epoch continuation; retain explicit epoch-3 checkpoint.
- [x] Run G0 60-sequence serial FP32 evaluation.
- [x] Run G1 3-epoch continuation; retain explicit epoch-3 checkpoint.
- [x] Run G1 60-sequence serial FP32 evaluation.
- [x] Produce per-sequence paired comparison and decision.
- [x] Update both registry tables and report with hashes, environment, commands, resource use, and final states.
- [x] Recheck protected M1 hashes and confirm their files remained unchanged.

## Follow-up audit and attribution

- [x] Audit train176/test60 sequence-level overlap and matched configs/parent/checkpoints.
- [x] Bootstrap paired sequence deltas for AUC, PR, NPR; store CIs.
- [x] Preserve first fresh96 loader-root failure; it stopped before any tracking frame.
- [x] Add direction gates, verify default-both parity and each trained direction changes outputs.
- [x] Freeze/use fresh96 sequence holdout disjoint from train176/test60.
- [x] Complete serial G0, G1-full, TIR-to-RGB-only, RGB-to-TIR-only fresh96 tracking and paired analysis.
- [ ] Run repeated alternating G0/G1 timing check; sequential one-pass FPS is insufficient to attribute stable overhead.
- [x] Update follow-up report and both registry tables with final attribution and fresh96 results.
