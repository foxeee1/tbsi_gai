# C5 Correspondence Analysis

## Contract

- Parent: B3 dense V2 fusion; diagnostic checkpoint is the existing S1 epoch-3 checkpoint with dense `v2` configuration, explicitly a proxy because an independent B3 dense checkpoint is unavailable.
- Frozen RPP-v2 test manifest: 60 sequences.
- FP32 tracker inference, serial scheduling, maximum 80 frames per sequence.
- No training and no change to C1-C4.
- Compared cosine row-wise matching against uniform-marginal log-domain Sinkhorn matching on the 64 RGB and 64 TIR template tokens.

## Results

| Layer/direction | Cosine precision | Sinkhorn precision | Delta | Cosine distance | Sinkhorn distance | False-rate delta |
|---|---:|---:|---:|---:|---:|---:|
| L3 RGB->TIR | 0.1920 | 0.2164 | +0.0244 | 0.7695 | 0.3769 | -0.1084 |
| L3 TIR->RGB | 0.1850 | 0.2130 | +0.0280 | 1.0348 | 0.3935 | -0.1304 |
| L6 RGB->TIR | 0.2129 | 0.2237 | +0.0108 | 0.5815 | 0.2562 | -0.0436 |
| L6 TIR->RGB | 0.2107 | 0.2227 | +0.0120 | 0.7093 | 0.2744 | -0.0532 |
| L9 RGB->TIR | 0.2102 | 0.2259 | +0.0157 | 0.5982 | 0.2541 | -0.0652 |
| L9 TIR->RGB | 0.2136 | 0.2255 | +0.0119 | 0.6675 | 0.2718 | -0.0512 |

Top-5 and top-10 target recall were already close to saturation for cosine and became nearly 1.0 with Sinkhorn, so they are not discriminative here. The strongest effect is lower overall match distance and fewer target-to-background false matches, especially in early layers.

## Decision

C5 provides evidence that a global correspondence constraint changes the matching structure, particularly at early layers. However, the target-match precision gain is only `+1.1` to `+2.8` percentage points, below the preregistered `>5%` promotion gate. Therefore do not launch C6 correspondence-guided fusion training yet. The result is promising as a mechanism diagnostic but insufficient to justify a new trainable module.

The next justified action is either to stop Innovation 1, or define a stronger correspondence metric/label contract before any C6 run. Do not tune Sinkhorn temperature or add trainable projection solely to chase this proxy gain.

## Artifacts

- `tools/run_c5_correspondence_analysis.sh`
- `tools/analyze_c5_correspondence.py`
- `output/experiments/C5_correspondence_analysis/logs/summary.json`
- `output/experiments/C5_correspondence_analysis/logs/correspondence.jsonl`

Canonical baseline SHA256 remains `4b9340f5ffaaeb8f2ab9af7887511e6401dbb1fb83d696538251a2d8fadd24b`; its diff is empty.
