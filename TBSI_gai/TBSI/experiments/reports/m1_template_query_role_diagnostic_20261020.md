# M1 Target-vs-Context Query Diagnostic (2026-10-20)

## Decision

**暂不支持**在当前五事件合同下继续使用前景标记的 target-vs-context query estimator：它没有改善目标 participation 或 CENTER 排名。不要训练该模式或继续调阈值/温度。下一分支应回到第一个错误 CENTER 峰的阶段归因，而不是再换 query 或旁路候选。

## Protocol

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42`, epoch 15. Config SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`; checkpoint SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`.
- Code base HEAD: `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`. The pre-edit dirty `tbsi_layer.py` was preserved at `output/experiments/M1_TEMPLATE_QUERY_ROLE_DIAGNOSTIC_20261020/source_snapshot/tbsi_layer.before.py`, SHA256 `be42aabc41c358c06c44d1c41d91fa354aaed53365ec83cb105e9ad807bcd883`.
- One explicit opt-in mode was added inside `V2TemplateFusion`; default `off` behavior is unchanged. For each template token, the mode computes its max search cosine minus the opposite-role template prototype's max search cosine. Foreground tokens contrast against the context prototype; context tokens contrast against the foreground prototype. RGB/TIR role scores are averaged, then combined with cross-modal agreement using the existing `tau=0.5`, `T=0.15` sigmoid key participation. No training, optimizer, `tbsi_fuse_search`, CENTER head, labels, or tracker state update was changed.
- Paired fixed-state comparison used the same five frozen train-side events from `M1_TEMPLATE_QUERY_STAGE_DIAGNOSTIC_20261020`, the same template initialization, state, frame, crop, and off-arm hard-background cell. Test split was excluded. Validation sequence-list SHA256 `bf238dae12b25699eb08f02519022d234897aca20e70726ca88c5c95b812e5a2`; event manifest SHA256 `005675b5c60e73a3b456aa4523e90285599e67bf0540acaf4dbc800a71be731f`.
- Runtime: 1x NVIDIA GeForce RTX 4090 D, Python 3.8.20, PyTorch `2.0.0+cu118`, CUDA 11.8, FP32. Command: `python tools/run_m1_template_query_role_diagnostic_20261020.py`.
- The first execution stopped in the diagnostic hook because its 1-D foreground mask was not batched. The complete traceback is preserved in `output/experiments/M1_TEMPLATE_QUERY_ROLE_DIAGNOSTIC_20261020/run.log`; the runner now explicitly batches the mask and the paired rerun completed in `rerun.log`.

## Results

| Validation event | GT rank off -> role | CENTER margin delta | loc9 role-query FG - BG | loc9 participation FG - BG |
|---|---:|---:|---:|---:|
| `5runthree:94` | 2 -> 2 | -0.00857 | -0.09080 | -0.05751 |
| `girlplayingphone:82` | 1 -> 1 | -0.00022 | -0.26164 | -0.19948 |
| `blkgirlbike:84` | 5 -> 5 | -0.00908 | -0.29398 | -0.25521 |
| `manaftercar:6` | 3 -> 3 | +0.00180 | -0.16895 | -0.13555 |
| `boyplayingphone366:36` | 2 -> 2 | +0.00531 | -0.18948 | -0.14757 |

Across five events, GT rank improved `0/5`, no top-1 error changed back to GT, mean CENTER GT-hard-background margin delta was `-0.00215`, mean loc9 role-query foreground-background difference was `-0.20097`, and mean participation foreground-background difference was `-0.15907`. The original max-query difference was also negative (`-0.09441`). Thus role conditioning made the measured foreground/background separation more background-favoring rather than correcting it.

## Attribution and Route

- **Refuted:** this initial-template foreground/context prototype contrast, under the current cosine-max estimator and fixed five-event contract, creates target-selective participation or improves CENTER peak ordering.
- **Not established:** the five selected train-side events do not prove that all target-conditioned template signals fail, nor identify MHSA as the system's root cause.
- **Next:** stop query uncertainty variants and bypass module search. Continue first-error localization on independent train-side fixed states: record the target versus strongest-background ranking before and after each relevant MHSA/TBSI stage, before and after `tbsi_fuse_search`, and at CENTER output. Only intervene on the first stage where target/background ordering is lost, with a non-target-position control.

This is mechanism evidence on five selected events, not a population estimate or a tracking-performance claim.

## Artifacts

- Runner: `tools/run_m1_template_query_role_diagnostic_20261020.py`.
- Model gate: `lib/models/layers/tbsi_layer.py`, `target_context_soft`.
- Outputs: `output/experiments/M1_TEMPLATE_QUERY_ROLE_DIAGNOSTIC_20261020/{per_event.csv,summary.json,run.log,rerun.log,sequence_list.txt,source_snapshot/}`.
