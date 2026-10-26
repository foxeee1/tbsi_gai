# M1 Query-conditioned Template Participation Stage Diagnostic (2026-10-20)

## Decision target

Explain whether the query-conditioned participation signal separates target from background at the template-key boundary, and locate any loss of that signal through TBSI search features, `tbsi_fuse_search`, and the CENTER response. This is a frozen-weight, fixed-state diagnostic. No training or protected M1 asset was changed.

## Protocol

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15. Config SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`; checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- Code: Git HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`; diagnostic runner SHA256 `b5a628ad7a54953e0db7f16414353702f8d7530a9c87a1dd932332a403d8c575`; `tbsi_layer.py` SHA256 `be42aabc41c358c06c44d1c41d91fa354aaed53365ec83cb105e9ad807bcd883`; tracker SHA256 `6b2b673b833ab3c0c25afb2846062654febee8ad1e0a459ce8df7b79e765824a`.
- Query formula: `q_i = 0.5 max_j cos(t_rgb_i,s_rgb_j) + 0.5 max_j cos(t_tir_i,s_tir_j)`; `r_i = 0.5 cos(t_rgb_i,t_tir_i) + 0.5 q_i`; `p_i = sigmoid((r_i - 0.5)/0.15)`. Only `V2TemplateFusion` mode changed between paired arms (`off` vs `query_soft`).
- Train-side validation: LasHeR `trainingset` at `/root/autodl-tmp/TBSI_gai/data/lasher/trainingset`, using the frozen 18-sequence list `experiments/diagnostics/m1_identity_calibration_v2/validation_sequences.txt` (SHA256 `bf238dae12b25699eb08f02519022d234897aca20e70726ca88c5c95b812e5a2`). Event rule: first frame up to frame 160 where the M1 off trajectory has IoU `<0.30` while the GT box is fully inside the predicted-state search crop. Five events were found.
- Test case studies: LasHeR `testingset` at `/root/autodl-tmp/TBSI_gai/data/lasher/testingset`; `whitesuvturn` frame 17 and `leftunderbasket` frame 76, selected from the existing M1 failure-event manifest. They are reported separately and excluded from validation summaries.
- Each fixed-state pair uses the same initial template, exact `state_before`, frame and crop. The hard-background location is selected once as the highest M1-off `CENTER * Hann` cell outside GT and reused for query mode. The capture includes template-token query/reliability/participation at TBSI locations 3/6/9, template features before/after each `V2TemplateFusion`, search features after each TBSI layer, the input/output of `tbsi_fuse_search`, and CENTER GT rank and GT-to-hard-background response margin.
- Environment: 1x NVIDIA GeForce RTX 4090 D, Python 3.8.20, PyTorch 2.0.0+cu118, CUDA 11.8, FP32. No training; FPS and GPU-hours were not separately measured (`NA`).
- Command: `python tools/run_m1_template_query_stage_diagnostic_20261020.py`.

## Results

| Validation event | Fixed IoU delta | GT rank off -> query | GT/hard-BG margin off -> query | Query participation FG - BG at loc9 |
|---|---:|---:|---:|---:|
| `5runthree:94` | 0.00000 | 2 -> 2 | -0.03980 -> -0.04294 | -0.01337 |
| `girlplayingphone:82` | -0.00018 | 1 -> 1 | +0.17167 -> +0.17217 | -0.07333 |
| `blkgirlbike:84` | -0.00005 | 5 -> 5 | -0.22269 -> -0.22554 | -0.15092 |
| `manaftercar:6` | +0.00025 | 3 -> 3 | -0.14838 -> -0.14796 | -0.05933 |
| `boyplayingphone366:36` | -0.00042 | 2 -> 2 | -0.04979 -> -0.04921 | -0.03407 |

Across five validation events, mean fixed-state IoU delta was `-0.00008`; GT rank improved in `0/5`; mean CENTER GT-to-hard-background margin delta was `-0.00090`. At loc3, mean query match was `0.6449` on foreground template tokens and `0.6921` on background tokens; mean participation was `0.7023` vs `0.7660`. At loc9 these were `0.6007` vs `0.6951`, and `0.7488` vs `0.8150`. Thus the statistic being used to assign reliability is consistently more favorable to background template tokens in these events.

The query mode barely changed that foreground/background participation gap relative to `off`: the mean gap change at loc9 was `-0.00016`. The search tokens returned by TBSI location 3 are unchanged by construction because `V2TemplateFusion` only rewrites template tokens there. By loc9, RGB search target-to-fixed-background feature L2 changed by only `+0.00570` on a mean baseline of `84.43`. At `tbsi_fuse_search`, RGB and TIR input feature L2 changes averaged `+0.02385` and `-0.00256`; fused feature L2 changed `+0.02056` on a mean baseline of `18.24`, while fused target/background cosine changed only `+0.00008`. These feature summaries are descriptive distances, not a learned class margin.

The two test case studies show the same negative loc9 participation gap: `leftunderbasket` `-0.03787`, `whitesuvturn` `-0.15735`. Their fixed-state GT ranks remained `2 -> 2`; margins changed by `-0.00031` and `+0.00387`, respectively. Fixed-state IoU deltas were only `+0.00010` and `0.00000`. The large sequence-prefix changes in the earlier closed-loop report therefore do not appear as a same-state correction at these already-failed frames; this is consistent with path/state feedback or an earlier divergence, but this diagnostic does not locate that earlier frame.

## Attribution and decision

- **Observed first mismatch:** before the first query-conditioned template MHSA update at location 3, the max-search match statistic already favors background template tokens over foreground tokens in the selected validation events. This is evidence against the current query-match statistic as a target reliability estimator.
- **Not observed:** no evidence that `tbsi_fuse_search` erases a useful target-selective correction. The measured change entering and leaving it is small, and the CENTER rank remains unchanged.
- **Supported:** the current query-conditioned participation candidate is worth retaining as a diagnostic artifact, but it is not admitted to training. Its mean closed-loop gain is not supported by same-state target/background ranking or a reproducible CENTER margin shift.
- **Not established:** this does not prove MHSA is the root cause, nor that any target-conditioned MHSA change will fail. These data do not test training supervision or parameter optimization.
- **Next allowed experiment:** keep the V2 boundary and one variable; first test a role-aware target-vs-context query statistic using the known initial template foreground mask, with the current `q_i` as a control. It must show foreground-over-background participation on held-out train-side events and a same-direction CENTER margin change before any training. Do not adjust `tbsi_fuse_search`, CENTER head, or participation strength in that test.

This diagnosis uses five selected train-side events and two previously observed test case studies. It is mechanism evidence for those events, not a population estimate or an M1 performance result.

## Evidence

- Runner: `tools/run_m1_template_query_stage_diagnostic_20261020.py`.
- Outputs: `output/experiments/M1_TEMPLATE_QUERY_STAGE_DIAGNOSTIC_20261020/{validation_events.json,per_event.csv,summary.json,run.log,sequence_list.txt,test_case_sequence_list.txt}`.
- Prior candidate result: `experiments/reports/m1_template_query_uncertainty_diagnostic_20261020.md`.
