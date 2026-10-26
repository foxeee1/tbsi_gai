# SJI extreme-event visual check

Scope was fixed to `SJI-00005`, `SJI-00014`, and `SJI-00030`, with no event reselection, alpha tuning, or model rerun. The check used the existing v1 event rows and offline-rendered contact sheets.

| Event | Sequence/frame | v1 IoU(00/10/01/11) | Visual audit |
|---|---|---|---|
| `SJI-00005` | `1handsth`, 60 | 0.8691 / 0.8714 / 0.8547 / 0.0000 | Large, crowded RGB scene; copied person is visibly displaced from GT and the destination is disjoint. No obvious coordinate overlap bug; the zero-IoU result is an extreme crowded-scene case and should not represent the full set. |
| `SJI-00014` | `AQmanfromdarktrees`, 60 | 0.8259 / 0.8152 / 0.5276 / 0.4491 | Low-contrast night scene with a small target and nearby copy. Paste and GT are separated; the event is visually difficult and the effect is moderate rather than a complete lock. |
| `SJI-00030` | `boydown`, 12 | 0.8022 / 0.8196 / 0.7908 / 0.0000 | Low-contrast night scene with a small target and nearby copy. Paste and GT are separated; no obvious geometry failure, but the zero-IoU event is highly sensitive to the synthetic copy. |

Artifacts: `output/experiments/synthetic_joint_interference_v1/batch/extreme_event_inspection/`.

This check is an artifact/geometry audit only. It does not validate a real-world failure mechanism and does not reopen training.
