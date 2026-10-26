# M1 Legacy Screening Asset Cleanup (2026-10-22)

## Scope

Removed superseded M1 train-side cohort membership files and their cohort-specific builders, plus historical per-sequence tracking text outputs. The new `m1_screening_v1_20261022` manifests, event miners, reports, training logs, configs, metrics, and fixed-state diagnostic outputs remain in place.

Across both passes, the total removed was 319 entries and 18,211 files (191,426,723 bytes).

The cleanup removed 170 entries containing 18,062 files (191,240,529 bytes):

- 12 legacy diagnostic cohort membership/manifest groups, without deleting their reports or diagnostic outputs;
- 7 superseded cohort analysis/build scripts and available Python 3.8 bytecode;
- non-anchor directories under `output/test/tracking_results/tbsi_track/`.

A second pass removed 149 superseded experiment-level sequence/sample manifests and one old precision-screen selector source (186,194 bytes). It excluded every `M1_SCREENING_*` output and all files under the current `m1_screening_v1_20261022` protocol. Its per-file SHA256 inventory is `experiments/protocols/m1_screening_v1_20261022/legacy_sampling_cleanup_20261022.json`.

The retained tracking directories are the protected M1 result and the existing v1.0.1 4090D equivalent-run result. The v1.0.2 and v1.0.3 result directories were absent before this cleanup; the v1.0.3 checkpoint/result had already been removed in the registered 2026-10-20 cleanup. No training checkpoints were selected by this cleanup. The M1 config and checkpoint SHA256 values were verified before and after deletion.

## Records and reproducibility

Experiment records, reports, configs, logs, scalar summaries, and the newly frozen protocol remain. Historical registry paths to deleted tracking outputs or sampling lists are retained as provenance and should be read as historical references, not as existing files. The deletion inventories record each path, byte count, and SHA256 in `experiments/protocols/m1_screening_v1_20261022/{legacy_asset_cleanup_20261022.json,legacy_sampling_cleanup_20261022.json}`.

This maintenance action does not change any model, training, or evaluation code behavior. It preserves the fresh 176/48/24/24 role split, the 244-sequence historical evaluation manifest, and the four retrospective paper event records.
