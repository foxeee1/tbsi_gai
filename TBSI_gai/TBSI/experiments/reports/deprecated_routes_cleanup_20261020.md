# Deprecated Routes Cleanup (2026-10-20)

## Scope

Removed non-report artifacts associated with the historical v1.0.0 3090 run and the stopped CFVI, TAFH, FMR, HF-Relation, R2, DBF, G0, TAM, LCGA, MCCA, and Bridge Token Pruner directions.

- Removed the `experiments/proxy/` sequence lists and manifests.
- Cleaned 75 matching experiment output directories, retaining any Markdown reports found inside them.
- Removed 59 matching per-sequence tracking result directories.
- Removed six abandoned layer modules and the matching historical configs for G0, TAM, Bridge Token Pruner, and v1.0.0.
- Removed loose CFVI, HF-Relation, and R2 launcher logs and P0 result files.

## Preserved

- All reports in `experiments/reports/`, including the reports for the stopped directions.
- Markdown reports found inside matching experiment output directories.
- The DBF G1 worker-4 run output and tracking result, which remain registered evidence.
- M1 and the 4090D equivalent reproduction group, including their configs, checkpoints, and tracking results.
- The experiment asset database and its prior rows.

The removed source modules and configs were snapshotted outside the repository at `/tmp/autodl-deprecated-code-configs-precleanup-20261020.tar.gz`. The protected M1 config and checkpoint SHA256 values were rechecked after cleanup and match their recorded values.

This cleanup removes historical machine-readable artifacts referenced by some earlier registry rows. Their experiment reports remain as the retained record; the deleted artifact paths should be treated as historical references, not present files.

## Large Artifact Cleanup (2026-10-20)

Removed the 15 checkpoint files explicitly selected by the user, all `selected_features.npz` files under `output/experiments/` (27 files), `fusion_mechanism.jsonl`, and 10 corresponding per-sequence tracking result directories. Experiment configs, reports, logs, summaries, and test records were retained. This includes the v1.0.3 seed3407 checkpoint and tracking outputs; its retained report/registry metrics remain historical records and the checkpoint is no longer available for inference or reproduction.

The M1 protected checkpoint was not part of this cleanup and its recorded SHA256 remains unchanged.
