# M1 Strip GT-Cell Counterfactual (2026-10-10)

## Decision

Stop the current frozen-M1, CENTER score-only Strip route. The GT-cell intervention shows that selecting the correct cell can improve overlap on many fixed-state failures. A leave-one-sequence-out hard-negative readout finds target evidence in 14/20 sequences, below the fixed 18/20 stability gate, so the precondition for a Strip-only ranking-loss experiment fails. Do not add a ranking loss to this checkpoint or expand Strip training.

This is a train-side diagnostic of one fixed event per sequence. It is not a tracker AUC test, not a causal estimate of Hann-window effects over all frames, and not evidence that all local readout changes fail.

## Experimental Contract

- Parent M1 is the protected epoch-15 checkpoint `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`. The candidate is the frozen-M1 Strip-score epoch-3 checkpoint `40854438c28aa7c22e72e3d649fa952d00811591761119260d8aac917b2d874d`.
- Both arms replay the same trace state, template, image, and crop for each selected frame. The GT cell is used only to select a response cell offline. The same arm's size and offset maps decode raw, Hann-weighted, and GT-selected cells.
- The initial deterministic cohort used SHA ranks 72-95 and 96-119, 24 sequences each, selected by the existing sequence-length eligibility rule without metrics. Eleven of 48 sequence names overlapped prior train-side validation/evaluation lists and were removed before final analysis; no selected event overlaps the earlier M1 fixed-state event list. The remaining 37 sequence cohort yielded 20 eligible events after filtering.
- Event rule: first frame of the first run of three consecutive valid frames with GT coverage of at least 99.9% by the reconstructed crop and M1 IoU below 0.5. At most one event is kept per sequence. A stricter 30-frame IoU-below-0.2 rule produced fewer than four events in the first 23 non-Strip-dev sequences and again in the expanded 47-sequence pool; it was not used for the final result. The event rule was broadened before comparing M1 and Strip; this makes the study exploratory.
- The previous evaluation lists excluded were LCGA faithful dev30, LCGA frozen dev30, Strip dev30, G0 Cplain eval, G0 FFM rescue eval, and DBF G1 train-side eval. Trace rank manifests, exclusions, event frames, and their hashes are recorded in `output/experiments/M1_STRIP_GT_CELL_COUNTERFACTUAL_20261010/oracle_final_hardneg_v2/selection_manifest.json`.
- No optimizer step, training, alpha/LR/initialization change, or checkpoint write occurred. An auxiliary CPU linear probe was fitted on frozen captured features with leave-one-sequence-out evaluation; it did not update either tracker.

## Results

The tracker applies the Hann window at response selection. On the 20 retained events, raw-to-Hann changed the selected cell on three events; all three improved both decoded center error and IoU. Seventeen events retained the same cell. Hann harmed neither metric on any event in this sample.

| Cell policy | Median center error (px) | Mean center error (px) | Median IoU | Mean IoU | Center error < 5 px | IoU >= 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Raw score peak | 26.10 | 32.51 | 0.278 | 0.274 | 1/20 | 0/20 |
| Actual Hann-weighted peak | 22.84 | 29.09 | 0.391 | 0.310 | 1/20 | 0/20 |
| GT-selected cell | 9.25 | 13.07 | 0.520 | 0.543 | 3/20 | 14/20 |

Compared with the actual Hann peak, a strict recomputation from `oracle_final_hardneg_v2/frame_results.csv` finds lower center error on 16/20 events and higher IoU on 16/20 events; both improved strictly on 15/20. Median paired improvement was 8.83 px in center error and 0.193 IoU. This is a substantial response-selection upper bound for overlap, while center localization remains inaccurate for most events: only 3/20 have center error below 5 px at the GT cell. The earlier shorthand “both on 16/20” conflated the two marginal counts; the machine-readable paired counts are authoritative.

M1 and Strip selected identical raw, Hann, and GT cells on all 20 events. Captured features at `tbsi_fuse_search` were identical between arms (maximum absolute difference 0; median relative L2 difference 0). The Strip candidate therefore did not alter the upstream fused representation or change cell choice on this independent event cohort.

The leave-one-sequence-out linear readout on frozen M1 fused features was evaluated against the four highest Hann-weighted M1 response cells outside GT support. Target margin was positive on 14/20 held-out sequences (70%), with median margin +3.26; this shows useful target information on many sequences, but fails the fixed stable-readability rule of at least 18/20 positive-margin sequences. The probe's predicted peak remained a median 3.11 grid cells from the GT center. A supplemental distant-background probe was weaker (7/20 positive margins, median -2.74), so it is not used for the target-versus-distractor decision. The hard-negative probe does not prove that the current CENTER head can use these features or that all relevant distractors are linearly separable.

## Interpretation

The observed Hann changes were helpful, not harmful, in these events. Therefore the mere train/inference difference between raw focal supervision and Hann-weighted online selection does not establish Hann as the error source.

The GT cell can recover IoU on many events, so peak selection contributes to the error. The fused features contain a target-versus-hard-negative signal in 14/20 held-out sequences, but it is not stable across the cohort, and GT-selected regression still has a 9.25 px median center error. Under the fixed 18/20 readability gate, the current score-only Strip should stop; the evidence does not justify a Strip ranking-loss trial. The next research target is how upstream representations and CENTER score/offset/size readouts handle the remaining sequences. This diagnostic does not establish that feature information is absent or decide which upstream change will help.

## Deviations and Limitations

- The stricter sustained-failure event definition yielded too few events, so the fixed event rule was broadened to a three-frame IoU-below-0.5 onset before the M1-versus-Strip comparison. Results are exploratory and specific to this onset definition.
- An initial cohort check found overlap with the Strip dev set; full results were then filtered against all known prior validation/evaluation lists before the registered 20-event summary was produced. The unfiltered intermediate results remain under `oracle/` and are not the reported analysis.
- Two feature-capture attempts failed before producing feature results: the first hook returned its captured array to PyTorch, and the second assumed a three-value crop. Both failures are retained in their partial output directories. A distant-background capture and an intermediate 23-event capture are retained as supplemental artifacts; the final hard-negative capture uses the filtered 20-event manifest and is stored in `oracle_final_hardneg_v2/`.
- Fixed-state replay uses the current working-tree tracker implementation, not a frame-equivalent replay of the protected archived full-test predictions. These measurements cannot be called strict M1 reproduction.
- Only one event per sequence is measured. The linear probe tests a fixed GT support versus four response-selected hard-negative cells, not the behavior of the nonlinear CENTER head. The oracle uses GT at inference only as a diagnostic and has no deployable status.
- No AUC, PR, NPR, OP50, OP75, or FPS was measured. GPU active hours were not independently timed and are `NA`.

## Provenance

- Git HEAD: `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9`.
- Environment: 1x NVIDIA GeForce RTX 4090 D 24 GiB; Python 3.8.20; PyTorch 1.9.0+cu111; CUDA 11.1; FP32.
- Frozen M1 config SHA256: `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`.
- Strip config SHA256: `c639030b45d9ad4f2ae9c4fd586a4539b52d94d521e929b5dd02aa1117fa975d`.
- Analysis scripts: `tools/analyze_strip_gt_cell_counterfactual_20261010.py` SHA256 `612e04a3659d87d1d6429a43c43852b128c7cb96e413c98647759c583948d03c`; `tools/capture_strip_gt_cell_fused_features_20261010.py` SHA256 `3b1becbf34edb276e0f4834231380fcf8f78acd076718e33513da8bbbab805ab`; `tools/finalize_strip_gt_cell_counterfactual_20261010.py` SHA256 `c3c23f44fc62da914d082741edf4de908d6982f0ea21c38e9ce248eed23f1247`; `tools/probe_strip_gt_cell_fused_features_20261010.py` SHA256 `1a539b430e2a59aecd3cddaad80d1dd422fed6f90b38cf1d0d57b0b782505c58`.
- Final hard-negative selection manifest SHA256 `853574f147a39f0d277b8a5b9e5c2bbb301346079537a39bf1fe65afed307e0b`; frame CSV SHA256 `f49d1b4358baadd624bd53366ff626ee42755674fa3ff5ec473e224d41f98564`; frozen fused features SHA256 `453ed889eb5eade906f543cb8028683158a2b22114837ad4b5b66e9b92557a7d`; hard-negative probe summary SHA256 `786016513acbf3ad40455773d4fe995614de9da0105165ce25e90a1f50c307b6`.
- Full traces and machine-readable outputs are under `output/experiments/M1_STRIP_GT_CELL_COUNTERFACTUAL_20261010/`; final hard-negative data are in `oracle_final_hardneg_v2/`.

## Commands

```bash
cd /root/autodl-tmp/TBSI_gai/TBSI
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/autodl-tmp/conda_envs/tbsi/bin/python -u tools/run_m1_search_stage_train_trace_20261009.py --output-dir output/experiments/M1_STRIP_GT_CELL_COUNTERFACTUAL_20261010/trace_rank72 --sequences 24 --rank-offset 72
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/autodl-tmp/conda_envs/tbsi/bin/python -u tools/run_m1_search_stage_train_trace_20261009.py --output-dir output/experiments/M1_STRIP_GT_CELL_COUNTERFACTUAL_20261010/trace_rank96 --sequences 24 --rank-offset 96
TBSI_INFER_FP16=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/autodl-tmp/conda_envs/tbsi/bin/python -u tools/analyze_strip_gt_cell_counterfactual_20261010.py
TBSI_INFER_FP16=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/autodl-tmp/conda_envs/tbsi/bin/python -u tools/capture_strip_gt_cell_fused_features_20261010.py
/root/autodl-tmp/conda_envs/tbsi/bin/python -u tools/finalize_strip_gt_cell_counterfactual_20261010.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/autodl-tmp/conda_envs/tbsi/bin/python -u tools/probe_strip_gt_cell_fused_features_20261010.py
```
