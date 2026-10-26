# S1 state contract sequence analysis

## Question

Can the existing selective state gate distinguish remaining162 sequences where S1 state feedback is harmful from sequences where it helps?

## Data and construction

- Merged the official per-sequence results from M1, S1, S1 output-only, and S1 selective-commit.
- Joined state commit coverage and mean risk probability from the selective-commit audit.
- Used the four-way attribution table to define:
  - `output_effect = B - A`: S1 output with M1 state minus M1.
  - `state_harm = C - B`: M1 output with S1 state minus S1 output with M1 state.
- Cohorts and metrics are inherited from the fixed 244-sequence experiment; no test threshold was fitted.

## Results

| Cohort | n | Mean state harm C-B | Spearman(coverage, state harm) | Spearman(risk, state harm) |
|---|---:|---:|---:|---:|
| low-AUC62 | 62 | +3.8164 | -0.123 | +0.132 |
| normal20 | 20 | +0.0983 | -0.138 | -0.015 |
| remaining162 | 162 | **-0.8658** | **-0.001** | **-0.015** |

The current gate variables have essentially no sequence-level ranking ability for remaining162 state harm. High coverage does not imply greater damage, and low coverage does not imply safety. The largest remaining162 state harms include `umbrellawillopen` (-57.89), `bluegirlbiketurn` (-32.69), `whiteridingbike` (-26.48), and `ab_whiteboywithbluebag` (-14.26) AUC points. Their commit coverage ranges from 0.012 to 0.181, so the damage is not explained by intervention frequency.

## Interpretation

The current selective-commit failure is not a threshold calibration problem. The fixed-state risk target and its visible inputs do not encode whether the current crop remains reachable after a state update. This also explains why hysteresis and partial alpha changes did not solve `boyss`-type failures.

The next useful state experiment must add a state-reachability observable, such as crop-edge distance, predicted center displacement relative to the previous safe state, and a recoverability signal over the next frame. A gate based only on peak margin, entropy, or same-state utility should be stopped.

## Evidence

- `output/experiments/M1_S1_STATE_CONTRACT_ANALYSIS_20261016/merged_sequence_state_analysis.csv`
- Source selective audit: `output/experiments/M1_S1_SELECTIVE_STATE_COMMIT_FULL244_20261015/full244/state_commit_by_sequence.csv`
- Source four-way attribution: `output/experiments/M1_S1_STATE_ONLY_FULL244_20261016/abcd_per_sequence.csv`
- Source report: `experiments/reports/m1_s1_abcd_state_output_attribution_20261016.md`

Status: `DIAGNOSTIC-COMPLETE / CURRENT-GATE-SIGNAL-REJECTED`.
