"""
TBSI 三级训练测试基准 (Benchmark)
==================================

三级渐进式验证流水线:
  1. Level 1 (Micro) — 快速验证损失收敛和代码正确性 (~20-30 min)
  2. Level 2 (Mini)  — 验证 SR/PR/NPR 指标趋势 (~3-5 hours)
  3. Level 3 (Full)  — 全量训练测试得到最终结果 (~12-15 hours)

核心原则:
  - 每次修改不超过 2 个文件
  - 严格归因：每次修改与指标变化一一对应
  - 前一级指标达标方可进入下一级

快速开始:
  python benchmark/bm_run.py --init                  # 初始化基准
  python benchmark/bm_run.py --run --desc "修改说明"  # 运行迭代
  python benchmark/bm_run.py --status                # 查看状态
"""
