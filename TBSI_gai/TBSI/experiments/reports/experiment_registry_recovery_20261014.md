# Experiment Registry Recovery (2026-10-14)

While updating the machine-readable asset registry, a pre-existing row with an inconsistent number of CSV fields caused the in-place writer to stop after truncating the file. The Markdown asset database remained intact and is the repository's declared authoritative source.

The registry was rebuilt as a valid 23-column CSV from the 274 unique experiment IDs present in the Markdown database. Each reconstructed row preserves its full source Markdown line in `Notes`; fields that could not be mapped reliably are `NA`. Valid rows from the truncated CSV and the bounded predictor record were retained where needed. The truncated four-line state is preserved at `experiments/snapshots/experiment_asset_registry_truncated_20261014.csv`.

Validation confirmed 274 unique IDs, 23 columns per row, and no duplicate IDs. The rebuilt table preserves the source evidence and conclusions but may not reproduce the exact former CSV cell mapping for every historical record. The Markdown database remains authoritative.
