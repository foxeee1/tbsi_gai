# M1 Candidate Selector OOF

固定 fused/RGB-only/TIR-only 候选，检验推理可见候选几何是否能在 sequence leave-one-out 下选择优于 fused 的候选。只在固定 crop/state 上评估，不修改 state commit。

结果：geometry-only selector mean IoU `0.62091`，相对 fused `-0.02291`，sequence CI `[-0.03508,-0.00279]`；oracle `+0.04396`。当前 selector 被否定。
