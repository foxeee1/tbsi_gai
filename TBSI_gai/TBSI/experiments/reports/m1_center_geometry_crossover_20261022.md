# M1 CENTER score and geometry crossover (2026-10-22)

## Question and hypothesis

After sparse neck training, CENTER GT-hard-background margin increased slightly while fixed-state IoU decreased. This diagnostic asks whether that mismatch is mainly associated with (a) CENTER response/peak selection, (b) size/offset geometry, or their interaction.

Hypothesis: if center selection is the main limitation, swapping candidate CENTER scores onto parent size/offset maps should reduce IoU on failure-onset states, while forcing the selected cell to the GT center should recover a substantial portion of IoU. If size/offset are the main limitation, candidate geometry should remain worse under the GT-center-cell oracle.

## Contract

- Parent checkpoint: `M1_SPARSE_CROSS_NECK_PARENT_QUICK_20261022`, epoch16, SHA256 `6b176839df86f9552fac7d6a25d83c8ac47dd267e9a0a02e11dc35738ab914de`.
- Candidate checkpoint: `M1_SPARSE_CROSS_NECK_QUICK_MATCHED_RETRY2_20261022`, epoch16, SHA256 `cfb15e4aaf7722570e63d0a41fa3350db8782710ff57f59b6c80d62c73ae848c`.
- Data: same 23 LasHeR official-train calibration events, each with pre-onset and severe-onset frame; 46 fixed states total. Each replay resets to the recorded state and evaluates the same frame/crop contract.
- Intervention: no model changes. Capture each model's CENTER score, size and offset maps, then offline cross-decode the parent/candidate score maps against parent/candidate geometry maps. Also force the decoded center to the nearest GT-center grid cell while retaining each model's geometry map.
- Decoder: exact CENTER rule (argmax of CENTER score multiplied by Hann; gather size/offset at selected cell; map normalized crop box back to image coordinates). IoU is against the fixed GT box.
- Source snapshot: `experiments/snapshots/m1_center_geometry_crossover_20261022.tar.gz`, SHA256 `70a8c730892da20b6f694e95a0dda9a768ec45dd9a6a8d7519598ea24be75a04`.
- Runner: `tools/run_m1_screening_fixed_state_flow_20261022.py`, SHA256 `f33eccbfa52e09df688c18c871288850eb481ee3876c4c10e1fa64fe3d73870c`.
- Analyzer: `tools/analyze_m1_center_geometry_crossover_20261022.py`, SHA256 `e8413930d2cc23488155167d2aff8cbf5d156166edf1d08c79459fb605f7e493`.
- Hardware/software: 1x NVIDIA GeForce RTX 4090D; Python 3.8.20, PyTorch 1.9.0+cu111, CUDA 11.1; FP32 fixed-state inference, one CPU thread.

## Results

Mean IoU across 46 paired states:

| CENTER score source | Size/offset source | Predicted peak | Forced GT-center cell |
|---|---|---:|---:|
| Parent | Parent | 0.47647 | 0.58768 |
| Parent | Candidate | 0.48669 | 0.58850 |
| Candidate | Parent | 0.46151 | 0.58768 |
| Candidate | Candidate | 0.46636 | 0.58850 |

The decoded parent-parent result reproduces the prior parent IoU (`0.47647`), and candidate-candidate reproduces the prior candidate IoU (`0.46636`), validating the offline decoder against the original fixed-state replay.

The paired effects differ by event phase:

| Phase | Parent score + parent geometry | Parent score + candidate geometry | Candidate score + parent geometry | Candidate score + candidate geometry |
|---|---:|---:|---:|---:|
| Pre-onset (n=23) | 0.59970 | 0.61168 | 0.62075 | 0.62897 |
| Severe onset (n=23) | 0.35324 | 0.36171 | 0.30226 | 0.30374 |

With parent scores held fixed, candidate geometry increases mean IoU by `+0.01198` pre-onset and `+0.00846` at onset. With parent geometry held fixed, candidate scores increase pre-onset IoU by `+0.02105` but decrease onset IoU by `-0.05098`. Candidate score maps select the same peak cell as parent on 38/46 states; top-1 falls inside the GT mask on 33/46 states for each model, though candidate mean GT rank is worse (1.674 vs 1.587).

Breaking this down by event phase, parent scores with candidate geometry improve IoU in 15/23 pre-onset states and 14/23 onset states. Candidate scores with parent geometry improve only 3/23 pre-onset states (20 ties), while they worsen 5/23 onset states (18 ties). These are the eight states where parent and candidate select different peak cells: all three pre-onset changes help, and all five severe-onset changes hurt.

The raw score argmax is inside the GT mask in 17/23 pre-onset and 12/23 onset states for both models. After Hann weighting, the parent is inside GT in 20/23 and 13/23 states; the candidate is inside GT in 21/23 and 12/23. At severe onset, mean distance from Hann-selected peak to GT center is `1.586` cells for parent and `1.928` for candidate. The raw-to-Hann peak moves `0.752` cells on average for parent onset maps and only `0.281` for candidate onset maps. Thus the candidate's onset regression is already visible at the score-map/readout boundary; it is not caused by candidate size/offset values.

At severe onset, mean decoded center error is `24.45 px` for parent score+parent geometry and `28.21 px` for candidate score+parent geometry. Swapping only to candidate geometry changes that error to `24.05 px` and `28.27 px`, respectively. Mean relative width error is `0.303` (parent geometry) versus `0.313` (candidate geometry) with parent scores; mean relative height error is `0.147` versus `0.144`. The geometry changes are mixed and small compared with the candidate score map's onset center-error increase.

Forcing the nearest GT-center cell raises parent-geometry IoU from `0.35324` to `0.52662` at severe onset (`+0.17338`), while the candidate-geometry result is `0.52476`. Across both phases, oracle IoU is about `0.588` for either geometry source, indicating little aggregate difference between their size/offset maps under this oracle.

## Interpretation

**Supported: center selection is the larger measured limitation at severe onset in this cohort.** Candidate geometry is mildly favorable when evaluated under parent scores, while candidate scores are specifically unfavorable at severe onset. The GT-center-cell oracle recovers substantially more IoU than swapping geometry does. This explains why a small increase in average target-vs-hard-background margin did not yield better replay: the margin is a peak-amplitude difference, while the box decoder chooses the global Hann-weighted argmax; ranking/location at onset and geometric decoding are separate factors.

**Still unresolved:** the candidate onset score map selects a peak farther from the GT center, and Hann moves it less toward the center than in the parent. The current evidence does not identify whether that comes from altered spatial response shape, target/background competition or an interaction with Hann's center prior. The oracle uses a nearest GT-center grid cell, so its gain is an upper-bound diagnostic rather than an available inference rule. Parent and candidate checkpoints differ throughout training, so map swapping attributes the observed outputs to score-map versus geometry-map roles but does not prove a unique training-mechanism cause.

## Decision and next step

Do not tune sparse top-k or dilation from this result. The next experiment should keep these checkpoints and states fixed and test the score-map/Hann interaction with one reversible readout control: decode candidate CENTER scores once with the learned Hann window and once with an all-ones window, while keeping candidate size/offset maps fixed; run the same paired comparison for parent. This will test whether Hann suppresses useful candidate peaks or whether the onset error is already in the raw score map. Use rank, raw/Hann peak distance, selected-cell geometry error and IoU. This remains a 46-state calibration diagnostic, not a full tracking-performance claim; the locked holdout remains closed.

## Evidence paths

- Parent head maps and replay: `output/experiments/M1_CENTER_GEOMETRY_PARENT_FIXED46_20261022/` (`head_maps.npz` SHA256 `afac5b6cf0c4f9bc0bb004832a7fb6bef0f17a725489c9e72f3688c66ffbada1`).
- Candidate head maps and replay: `output/experiments/M1_CENTER_GEOMETRY_CANDIDATE_FIXED46_20261022/` (`head_maps.npz` SHA256 `40e93ebe9eb67a6548b3f248adb33e51893d2a8994e9ed0ccc7352e51d796a6b`).
- Crossover rows and summary: `output/experiments/M1_CENTER_GEOMETRY_CROSSOVER_FIXED46_20261022/` (`summary.json` SHA256 `2ca9ad48ba9dee06c2d7bb2246305255c612b55fd3cc9288f2379ad291bfd310`).
- Expanded raw/Hann peak and geometry-error analysis: `output/experiments/M1_CENTER_GEOMETRY_CROSSOVER_FIXED46_V2_20261022/` (`summary.json` SHA256 `578c002ab1fe26be7cc403376869428cba28fd20f3fec581c81eefddcc1618cc`). Analyzer SHA256 `29c21e2fb707daeed36e73271bf71cd228c081c2d3743e4a53bbc031722904d9`; source snapshot `experiments/snapshots/m1_center_geometry_crossover_20261022_v2.tar.gz`, SHA256 `7761a6b15b22025a660c51ba57f008399c409bf8bea13c6586d228ac5b8946cb`.
- Exact commands and environment manifest: `output/experiments/M1_CENTER_GEOMETRY_CROSSOVER_FIXED46_V2_20261022/experiment_manifest.json`.
