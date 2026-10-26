# M1 template route probe v1

- ID: `TEMPLATE-ROUTE-PROBE-V1-20260928`
- Parent: `TBSI-E-MHSA-FULL-20260927`
- Config: `TBSI_E_MHSA_FULL_4090D_seed42`
- Scope: frozen-weight probe; 24 fixed LasHeR train proxy sequences, 2 deterministic events per sequence, 5 conditions, offset 0
- Conditions: `F`, `RT_down`, `TR_down`, `Both_down`, `Both_up`, with bias magnitude `ln(2)`
- Hardware/software: 1 x RTX 4090D; Python 3.8.20; PyTorch 1.9.0+cu111; CUDA 11.1; FP32 inference

The corrected smoke run passed with default-vs-explicit-F box parity `0.0` on 4 events.
The full probe completed with 48 events and 240 condition rows. Full parity remained
`0.0`.

| Condition | Mean event IoU delta vs F | Median | Positive events | Negative events |
|---|---:|---:|---:|---:|
| `RT_down` | -0.000070 | 0.000000 | 21 | 22 |
| `TR_down` | +0.000227 | +0.000092 | 27 | 16 |
| `Both_down` | +0.000144 | 0.000000 | 23 | 20 |
| `Both_up` | -0.000147 | -0.000000 | 21 | 22 |

Sequence-level aggregation gives `RT_down`: 11/24 positive sequences,
`TR_down`: 17/24, `Both_down`: 15/24, and `Both_up`: 10/24. The route intervention
changed attention mass consistently, but the localization response was small and not
uniform enough to support a strong static-routing claim.

Mean cross-modal attention mass by layer (`F` -> `RT_down` / `TR_down` / `Both_up`):

| Layer index | RGB query -> TIR key | TIR query -> RGB key |
|---:|---:|---:|
| 0 | 0.3237 -> 0.2271 / 0.3237 / 0.4335 | 0.7846 -> 0.7846 / 0.6962 / 0.8568 |
| 1 | 0.3451 -> 0.2444 / 0.3451 / 0.4563 | 0.7225 -> 0.7225 / 0.6224 / 0.8088 |
| 2 | 0.3457 -> 0.2372 / 0.3457 / 0.4658 | 0.6499 -> 0.6499 / 0.5253 / 0.7624 |

Interpretation: directionality is detectable, with `TR_down` more favorable than
`RT_down` in this development probe, but the effect is too small and heterogeneous to
justify a learned shared/private module yet. This is a functional sensitivity result,
not evidence that HMoE or M3 will improve tracking.

Status: `DIAGNOSTIC-COMPLETE`.
