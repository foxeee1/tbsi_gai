---
name: tbsi-temporal-token-v2-design
description: TemporalTokenLayerV2 multi-granularity gate + quality-aware cross-modal design
metadata:
  type: reference
---

# 时序令牌 v2 设计方案 (2026-07-16 保留)

## 设计背景

基于 v1 的三集交叉验证 (MiniA/B/C) 的分析结论：
- v1 是三集平均 **-0.48 AUC 退化**
- 根因：伪时序 (每帧独立，训练无传播) + 单标量门控 + 跨模态无交互
- 但 LI(低光照) 和 AIV(表观变化) 在 MiniB/MiniC 上**稳定提升** (+0.8~2.5 AUC)

核心判断：时序令牌方向正确，但算子设计过于粗糙。

## 架构设计

```
时序令牌 v2 = Token-centric attention + 多粒度门控 + 质量接口 + 跨模态交互
```

| 组件 | v1 | v2 |
|------|----|----|
| 门控 | 1 个 scalar (控制 4×256 维) | K 个 per-token gate + feat-ctx 调制 + quality 调制 |
| Quality 接口 | 无 | `Linear(2, K)` → bias add → gate logit<br>None 时自动 0 (可插拔) |
| 模态交互 | 双流完全独立 | 2K token → self-attn → split (O(64) FLOPs, +2%) |
| MLP | 已修复 | 保留 |
| 梯度裁剪 | 0.1 | 0.3 |
| 参数量 | 528K | ~550K |
| FLOPs | 143M | ~146M |

## 质量接口设计哲学

论文故事线：**质量感知融合 → 质量门控时序记忆 → 时序增强融合** (闭环)

```
DaFusion 输出 quality_hint (B,2) ──→ 时序令牌 gate_logit += proj(quality_hint)
                                              │
                                              ▼
                                    质量好→gate小→保持记忆
                                    质量差→gate大→快速更新
                                              │
                                              ▼
                                   TC3 用更新后的 token 调制融合特征
                                              │
                                              ▼
                                    融合输出再次增强 → 检测头
```

- `QUALITY_GATE: False` (standalone): quality_proj 零初始化 → 输出 0 → 不影响门控
- `QUALITY_GATE: True` (dual): quality_hint 从融合模块传入 → 调制门控

## 消融序列 (待跑)

| 实验 | 配置 | 预期 |
|:----:|:----:|:----:|
| A | 完美基线小数据 | 基准 (64.26) |
| B | 完美基线+融合 | 融合独立贡献 |
| C | 完美基线+时序令牌v2 | +0.3~0.5 AUC |
| D | 完美基线+融合+时序令牌v2 | >> B+C-A (协同) |

## 文件清单

- Code: `lib/models/layers/temporal_token_v2.py` (新模块，需集成到 tbsi_track.py)
- Config: `experiments/tbsi_track/基线+时序令牌v2.yaml`
- Config: `experiments/tbsi_track/基线+融合+时序令牌v2.yaml`

## 待办 (集成到主模型)

在 `tbsi_track.py` 中需要：
1. `build_tbsi_track` 中判断 `TEMPORAL_TOKENS_V2` → 导入 `TemporalTokenBlockV2`
2. `forward_head` 中从 `forward_fusion_only` 拿到 fused_feat 后提取 quality_hint
3. `forward_fusion_only` 中 DaFusion 的 `forward()` 返回 `(fused, quality_hint)` 而不是只返回 fused
4. `TemporalTokenBlockV2` 的 forward 接收 `quality_hint` 参数
