# Early-Gate Token Reliability Quick3

## Question and decision

Test whether token-local match reliability adds value to the existing Early-conditioned MSHA gate, whose two directional Value scales are predicted from max RGB/template-to-search similarity, max TIR/template-to-search similarity, and paired RGB/TIR agreement.

**Decision: does not support this operator for promotion.** The candidate signal is numerically active and respects its bound, but the fixed seven-sequence screen has a strongly negative paired result, with severe losses on two sequences and a sizeable throughput cost. Do not start a full training run from this result. This does not establish that every token-reliability design is harmful; it rejects this formula under the current short-training and retrospective quick7 contract.

## Operator

For each modality token i, compute cosine similarity s_ij from its early template token to each early search token. With p_ij=softmax(s_ij/0.07), define normalized entropy H_i=-sum_j p_ij log(p_ij)/log(N_s), top-two margin m_i=s_i,(1)-s_i,(2), and paired agreement a_i=sigmoid(cos(t_i_rgb,t_i_tir)/0.10).

    c_i = (1 - H_i) * sigmoid(m_i / 0.07) * (0.5 + 0.5 * a_i)

The token residual is q_i=tanh(log(c_i+epsilon)-mean_i[log(c_i+epsilon)]), centered again to zero mean and divided by max(1,max_i|q_i|). The source scale is r_i=1+0.10q_i. Thus the per-sample/per-modality mean is 1 and r_i lies in [0.9,1.1]. The factor is applied only to the matching modality's source Value when it contributes to the opposite modality's queries. Early gate scaling remains active. Q, K, attention ordering, self-messages, CENTER and template update are untouched. There are no new trainable parameters.

The residual reads the four early feature tensors only; it does not consume a template mask. The historical DCVR training-mask fallback difference is therefore not part of this comparison.

## Protocol

- Matched configs: M1_EARLY_GATE_TOKEN_REL_QUICK3_CONTROL_RETRY1_20261026.yaml versus M1_EARLY_GATE_TOKEN_REL_QUICK3_CANDIDATE_RETRY1_20261026.yaml. Parsed configs are identical. The only changed variable is runtime TBSI_EARLY_GATE_TOKEN_REL_MODE=off/on.
- Both arms initialize from TBSITrack_SOT_Pretrained.pth.tar (SHA256 1f6ebf5905ed0a404a11beac944ef5d544c9e81605120c2052a2518467d2651b) and use fresh AdamW/StepLR, seed 42, LasHeR train 512 samples/epoch for 3 epochs, batch 16, accumulation 8, LR 1e-4, LR drop epoch 3, weight decay 1e-4, AMP, worker 4, and epoch-3 checkpoint. The standard config freezes backbone layer 0; the other configured parameters train jointly. The logs show 32 batches per epoch, i.e. 4 optimizer updates per epoch and 12 total.
- Train paths resolve to /root/autodl-tmp/TBSI_gai/TBSI/data/lasher/trainingset; test paths resolve to /root/autodl-tmp/TBSI_gai/data/lasher/testingset. These paths resolve to the same device/inode dataset tree.
- Test uses the retrospective m1_dcvr_quick7_legacy_20261025.txt manifest (SHA256 44247b6dbc9259c3e17c61a2d513ca1dff9588ec977946934c3ec4e87bc8d087), FP32, tracker threads=0, and template update off. The cohort was rebuilt from historical files after the earlier run and is a development screen, not a locked validation set.
- Hardware/software: 1x NVIDIA GeForce RTX 4090 D 24 GiB; Python 3.8.20, PyTorch 1.9.0+cu111, CUDA 11.1. Paired training and testing were serial; total elapsed GPU-task wall time was 298 seconds (about 0.083 GPU-hours).
- Train command per arm: python -u lib/train/run_training.py --script tbsi_track --config CONFIG --save_dir output --seed 42 --use_lmdb 0.
- Test command per arm: TBSI_RPP_TEST_LIST=MANIFEST TBSI_CHECKPOINT_OVERRIDE=EPOCH3_CHECKPOINT TBSI_EARLY_GATE_TOKEN_REL_MODE=off-or-on python -u tracking/test.py tbsi_track CONFIG --runid RUN_ID --dataset_name rpp_lasher_test --threads 0 --num_gpus 1.

The first launch stopped before optimizer construction because the initial control YAML omitted TRAIN.TBSI_DROP_PATH=[0,0,0] for the three fusion locations. It produced no update or checkpoint and is preserved as FAILED-EXECUTION / CONFIG-CONTRACT. The matched retry used new config/output identities and passed model construction. After both tests completed, the first offline-analysis invocation failed because its script did not add the repository root to sys.path; analysis was rerun on the existing predictions after fixing the path. The failed traceback remains in snapshots/m1_early_gate_token_reliability_quick3_20261026/paired_analysis.json.

## Results

Macro mean across the seven sequences; values are percentage points.

| Metric | Control | Candidate | Delta |
|---|---:|---:|---:|
| AUC / success | 56.99 | 48.12 | -8.87 |
| AO | 59.11 | 49.00 | -10.11 |
| Precision | 71.70 | 60.82 | -10.89 |
| Normalized precision | 70.22 | 59.98 | -10.24 |
| OP50 | 71.46 | 60.68 | -10.78 |
| OP75 | 46.54 | 40.59 | -5.96 |

Sequence AUC changes:

| Sequence | Delta |
|---|---:|
| 3rdfatboy | -1.19 |
| 4thboywithwhite | +0.18 |
| ab_bikeoccluded | -60.39 |
| boydownplatform | -0.09 |
| leftfarboycomingpicktheball | 0.00 |
| shinycarcoming | -18.00 |
| womanback2car | +17.39 |

Two sequences improved, four declined, and one was unchanged. ab_bikeoccluded moved from AUC 66.37 to 5.98; shinycarcoming fell from 56.16 to 38.16. womanback2car gained 17.39 pp. The cohort therefore gives a strong negative point estimate but an unstable, sequence-dependent profile.

## Information flow and attribution

- Both arms had nearly identical quick-train aggregate losses. At epoch 3, train total loss was 1.11091 in both; validation total loss was 1.59847 control and 1.59892 candidate, with IoU 0.66707 and 0.66703. This 12-update schedule does not establish a mature training response.
- Candidate diagnostics contain 18 training and 144 test samples. Across test samples, mean source scales were approximately 1.0000; observed RGB range was 0.9098–1.0935 and TIR range 0.9098–1.0924. Mean residual standard deviation was 0.331 RGB and 0.289 TIR. The signal did not collapse and respected the specified bound.
- The negative result is not explained by an absent residual or a mask fallback. The mechanism changes the intended cross-modal Value path; it does not improve this cohort. Since the reliability uses local sharpness and paired agreement, a confident but wrong correspondence can still be amplified. The catastrophic ab_bikeoccluded and shinycarcoming outcomes are consistent with that risk, but sequence-level prediction files alone do not prove this was the causal scene mechanism.
- Test FPS fell from 70.11 to 58.72 (about 16.2%). The added matching-distribution computations are a measurable cost under this tracker/test setup.
- The control and candidate training/validation behavior is close while test outputs diverge. That pattern, combined with two extreme sequence losses, leaves training progress and the retrospectively selected quick7 cohort as limitations. It does not justify extrapolation to full LasHeR.

## Evidence and artifacts

- Paired metrics: snapshots/m1_early_gate_token_reliability_quick3_20261026/analysis_retry1/{summary.json,per_sequence_metrics.csv}.
- Run provenance and exact commands: snapshots/m1_early_gate_token_reliability_quick3_20261026/provenance.txt.
- Protocol and failed-first-launch record: snapshots/m1_early_gate_token_reliability_quick3_20261026/protocol.md.
- Control checkpoint SHA256: e96cf7b27cd86be6b5e3ea97e0a0c2f7f98e2bf07b9f806985df2900df67c8b5.
- Candidate checkpoint SHA256: c03706767e7ddd4edc17626a2f8b36e88d914e564ddd853e1080e07184c73b22.
- Operator code SHA256: 900214619c066a0f4639cd595143e93675b390d1f5ac1907ea5c3fce56aff99f.

## Next step

Do not promote this formula or run it full15. First inspect the paired frame-level trajectories for the two catastrophic losses under the exact same initial crop, and verify when the first divergence begins relative to the new per-token Value scale. That is a read-only attribution pass. Only a distinct failure mode tied to a specific high-confidence wrong token would justify another operator design; a generic amplitude sweep is not supported by these data.
