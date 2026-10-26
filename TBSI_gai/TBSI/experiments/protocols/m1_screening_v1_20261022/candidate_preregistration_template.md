# M1 Candidate Pre-registration

Copy this file to a new experiment ID before changing model behavior.

## Question

- One failure class (A/B/C/D):
- One falsifiable hypothesis:
- Evidence that motivated it, with asset paths:
- What result would disprove it:

## Frozen comparison

- Direct parent ID and checkpoint SHA256:
- Code commit plus dirty-source snapshot SHA256:
- Candidate config and output directory:
- Single changed variable:
- No-op/control condition:
- Non-target-position control:
- Train manifest and SHA256:
- Evaluation manifest and SHA256:
- Seed list:
- Resolution/backbone/precision/optimizer/batch:
- Full-training optimizer-step budget:
- Proxy optimizer-step budget and ratio:
- Warmup/decay fractions of total steps:
- Selected checkpoint rule:
- Evaluator and metric command:

## Predicted information flow

- Expected change at module input:
- Expected target versus hard-background margin/rank change:
- Expected CENTER score/rank/decoded box change:
- Expected sequence-level change:
- Expected non-target control result:

## Admission gates

- Disabled and zero-gate parity:
- Shape/save-load contract:
- Intended parameter gradient and frozen-parent drift:
- Fixed-sample fit result:
- Fixed-state causal result:
- Stop condition:

## Result and attribution

- Actual optimizer steps and schedule:
- Mechanism metrics:
- Full-244 metrics and per-sequence deltas:
- Attribute and A/B/C/D subgroup outcomes:
- Proxy/full agreement:
- Failure class: compute/gradient contract, data support, supervision/optimization, representation/interface, closed-loop/evaluation:
- What is excluded, what remains unresolved:
- Decision: continue / revise one variable / stop:
- Complete command, environment, output and registry ID:
