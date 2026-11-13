# Paper Core: Mass-Redist

This checkout contains the recent implementation associated with the paper
revision's **Mass-Redist** method for RGB-T tracking. The feature is disabled
by default, so the historical TBSI path keeps its identity behavior unless a
configuration explicitly enables it.

## Included implementation

- `lib/models/layers/tbsi_layer.py`: feature-gated cross-modal fusion and
  receiver-side mass allocation.
- `lib/models/layers/low_rank_joint_mass_reallocator.py` and
  `lib/models/layers/expected_score_mass_reallocator.py`: low-rank scoring and
  expected-score budget allocation used by the latest paper path.
- `lib/models/layers/early_conditioned_msha_gate.py` and
  `lib/models/layers/early_gate_search_context.py`: search-conditioned source
  Value gating used as the fixed parent path.
- `lib/models/tbsi_track/`, `lib/config/tbsi_track/config.py`: model wiring and
  explicit configuration switches.
- `paper_revision_20261102/method3_mass_redist.tex`: the corresponding method
  description and equations.

The other small layer modules imported by `tbsi_layer.py` are included because
Python imports the host module eagerly; they remain independently gated and
are not enabled by the paper configuration.

## Enabling the method

Set these fields in a new experiment configuration:

```yaml
MODEL:
  BACKBONE:
    EARLY_CONDITIONED_CROSS_GATE: true
    EARLY_GATE_SEARCH_CONTEXT_MODE: cross
    EARLY_GATE_VALUE_MODULATION: true
    MASS_REDIST_ENABLE: true
    MASS_REDIST_MODE: lr_expected
```

Use a fresh run directory, a public pretrained checkpoint, and a new run
identity. Do not use this export as a claim that the method is a validated
multi-seed result; the accompanying reports define the current evidence
boundary.
