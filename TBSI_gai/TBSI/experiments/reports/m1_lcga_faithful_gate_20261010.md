# M1 official LCGA bounded gate (2026-10-10)

## Decision

**Stop before matched short training.** The pinned official LCGA operator passed semantic and numerical gates, but its two-modality search refiner exceeded the prespecified 10% closed-loop FPS cost in both execution orders. This is an efficiency stop for this insertion/implementation, not evidence that curvature guidance cannot improve tracking. The new dev30 was not evaluated; no faithful-LCGA checkpoint was trained, and no full training or full test was started.

## Identity and scope

- Parent: protected `TBSI_E_MHSA_FULL_4090D_seed42` epoch 15, SHA256 `1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef`; protected YAML SHA256 `fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1e`.
- Official CGA commit `8941ace6065fdb9bddb7b56e6c00152f67defd78`; `basicsr/archs/cga_arch.py` SHA256 `dd189242247a1edf9439f810a61acdbc595e7900546d7b3aef05a2d9d1ed2ad9`. The five relevant definitions were ported without AST differences. `idx=0, rs_id=0` does not shift; tested `idx=2` activates shift and mask.
- New independent config `experiments/tbsi_track/M1_LCGA_FAITHFUL_PROXY_seed42.yaml`, SHA256 `a9de1cdf162a1b08fa35801eeaf228f736948a16e14f5655621b477a95089d6d`; feature gate `TYPE=OFFICIAL_LCGA`. The old A1 remains **LCGA-style-unmasked-shift**, not official LCGA.
- Final source snapshot `output/experiments/M1_LCGA_FAITHFUL_GATE_20261010/provenance/source_gate_final_20261010.tar.gz`, SHA256 `ef25aaa07883b3c87e430cfa57e5beb079a7bb18c94177bc28572f8972af98e5`; pre-port snapshot SHA256 `1fc695b1887bc863c06e45191ac98650c790f96f059a8bc9df58b871a06ff4f2`. Git HEAD `09f3d57c0761b01de1fa3aa29ff296c20ea5e9e9` plus these uncommitted source snapshots.
- Hardware: one NVIDIA GeForce RTX 4090 D 24 GiB; Python 3.8.20, PyTorch 1.9.0+cu111, CUDA 11.1; FP32, single tracker process, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. Actual GPU hours not separately measured (NA).

## Gates

1. Operator: official-vs-port output, loss, input gradient and parameter gradient maximum differences were **0** on `[2,256,768]` for both unshifted and shifted/masked variants, CPU and CUDA. The adapter has 4,755,386 trainable parameters. With nonzero outer and curvature gates, disabling curvature changed output; all four curvature-convolution gradients were positive. Evidence: `operator_parity_cpu.json`, `operator_parity_cuda.json`.
2. Full model: identical M1 weights in disabled and enabled models; only 88 refiner state keys were absent from M1. At zero outer alpha, `pred_boxes`, `score_map`, `size_map` and `offset_map` maximum differences were **0**. One real old-train batch of 16 was reused for four diagnostic optimizer steps, not scheduled training. Outer gate gradient appeared immediately; internal curvature-alpha gradient at step 2; RGB first curvature-convolution gradient at step 3 (`2.90e-14`) and step 4 (`3.69e-14`). This proves graph connectivity, not practically useful curvature learning. Refiner and fusion LR both `1e-4`. Evidence: `full_model_batch_gate.json`.
3. Closed-loop latency: fixed first three sequences from the **old** train-side dev list (`2ndcarcome`, `3whitemen`, `4runeight`; 806 tracked frames per mode). Manifest SHA256 `37ad5c642c246d49dbdf7d05bcaab8551bff0a0c75aaa92aa57b5570779c33f2`. Disabled mode skipped the entire refiner; enabled mode computed both official branches with outer alpha zero. Predictions were framewise identical (`max box delta 0`). Disabled-first: **68.53 vs 60.36 FPS**, **11.92%** drop. Enabled-first: **72.22 vs 58.37 FPS**, **19.17%** drop. The order sensitivity changes the magnitude but not the stop decision. Evidence: `latency_gate.json`, `latency_gate_reverse.json`.

## Interpretation and next modification

The operator is faithful to the pinned source; integration preserves M1 at zero gate. However, the present independent RGB/TIR refiner costs too much before any performance evidence, and the real-batch curvature gradient is extremely weak under double-zero gating. No claim about center error, target-background margin, AUC improvement or causal repair is supported. Do not train this candidate or tune it on the unused dev30. A future version would require a **new method identity and budget** with a specific latency-reduction hypothesis and a fresh predeclared gate; the frozen dev30 remains untouched (manifest SHA256 `4739caca72bb3f8d2ec6dce71a7e1abb9be4c7cac6a8ee354e584a6378bab6d2`).

Commands from repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/autodl-tmp/conda_envs/tbsi/bin/python tools/verify_lcga_official_gate_20261010.py --device cuda --output output/experiments/M1_LCGA_FAITHFUL_GATE_20261010/operator_parity_cuda.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/autodl-tmp/conda_envs/tbsi/bin/python tools/test_lcga_faithful_model_20261010.py
TBSI_INFER_FP16=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/autodl-tmp/conda_envs/tbsi/bin/python tools/benchmark_lcga_faithful_20261010.py
TBSI_INFER_FP16=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/autodl-tmp/conda_envs/tbsi/bin/python tools/benchmark_lcga_faithful_20261010.py --reverse
```
