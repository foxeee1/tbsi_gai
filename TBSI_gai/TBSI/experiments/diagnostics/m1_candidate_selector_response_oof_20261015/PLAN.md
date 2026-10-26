# M1 Candidate Selector Response OOF

固定三候选，增加 peak score、top-2 margin、熵、peak cell 和候选间 response 差异，sequence-LOSO 评估 selector。

结果：mean IoU `0.60609`，相对 fused `-0.03774`，95% CI `[-0.07081,-0.01292]`；selector 被排除。下一项转做 center/crop 可达性审计。
