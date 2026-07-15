"""
MiniLasHeR: 对照基准比较工具

比较两次实验的基准数据，判断修改是否有效。
重点关注梯度流变化和验证指标差异。

用法:
  python tools/compare_benchmark.py --baseline output/benchmark/reference_benchmark.json \\
      --experiment output/benchmark/exp_fix_detach_benchmark.json

  # 只比较特定模块组
  python tools/compare_benchmark.py --baseline ref.json --experiment exp.json \\
      --filter da_fusion,degradation_mod
"""

import argparse
import json
import sys


def compare_grad(group_name: str, base_grad: dict, exp_grad: dict, filter_key: str = ""):
    """Compare gradient norms between two experiments for a module group."""
    if not base_grad or not exp_grad:
        return

    print(f"\n  ── {group_name} ──")
    print(f"  {'Metric':<15} {'Baseline':>12} {'Experiment':>12} {'Δ':>12} {'':>4}")
    print(f"  {'-'*55}")

    for key in ['mean', 'max', 'min', 'final']:
        b_val = base_grad.get(key, 0)
        e_val = exp_grad.get(key, 0)
        delta = e_val - b_val
        # Flag significant changes
        flag = ""
        if abs(delta) > abs(b_val) * 0.5 and abs(b_val) > 1e-8:
            flag = " 💥"
        elif abs(delta) < 1e-8 and abs(b_val) > 1e-6:
            flag = " ⚠️"
        elif e_val > b_val and b_val > 0:
            flag = "  ↑"
        elif e_val < b_val and e_val > 0:
            flag = "  ↓"
        print(f"  {key:<15} {b_val:>12.6f} {e_val:>12.6f} {delta:>+12.6f}{flag}")


def compare_loss(base_summary: dict, exp_summary: dict):
    """Compare epoch-level loss trends."""
    if not base_summary or not exp_summary:
        return

    print(f"\n  ── Loss Trend ──")
    all_epochs = sorted(set(list(base_summary.keys()) + list(exp_summary.keys())))
    print(f"  {'Epoch':<10} {'Base Loss':>12} {'Exp Loss':>12} {'Δ':>12}")
    print(f"  {'-'*46}")

    for ep in all_epochs:
        b_loss = base_summary.get(ep, {}).get('Loss/total_mean', None)
        e_loss = exp_summary.get(ep, {}).get('Loss/total_mean', None)
        if b_loss is None and e_loss is None:
            continue
        delta = (e_loss - b_loss) if (e_loss is not None and b_loss is not None) else 0
        b_str = f"{b_loss:.6f}" if b_loss is not None else "N/A"
        e_str = f"{e_loss:.6f}" if e_loss is not None else "N/A"
        d_str = f"{delta:+.6f}" if (e_loss is not None and b_loss is not None) else "N/A"
        flag = " 🔴" if (e_loss is not None and b_loss is not None and e_loss > b_loss * 1.05) else ""
        flag = " ✅" if (e_loss is not None and b_loss is not None and e_loss < b_loss * 0.95) else flag
        print(f"  {ep:<10} {b_str:>12} {e_str:>12} {d_str:>12}{flag}")


def compare_eval(base_eval: dict, exp_eval: dict):
    """Compare evaluation metrics."""
    if not base_eval or not exp_eval:
        return

    print(f"\n  ── Evaluation Metrics ──")
    all_metrics = sorted(set(list(base_eval.keys()) + list(exp_eval.keys())))
    print(f"  {'Metric':<20} {'Baseline':>10} {'Experiment':>10} {'Δ':>10}")
    print(f"  {'-'*50}")

    for m in all_metrics:
        b_val = base_eval.get(m, None)
        e_val = exp_eval.get(m, None)
        flag = ""
        if b_val is not None and e_val is not None:
            if isinstance(b_val, (int, float)) and isinstance(e_val, (int, float)):
                delta = e_val - b_val
                if delta > 0.5:
                    flag = " ✅"
                elif delta < -0.5:
                    flag = " 🔴"
                print(f"  {m:<20} {b_val:>10.4f} {e_val:>10.4f} {delta:>+10.4f}{flag}")
                continue
        b_str = f"{b_val}" if b_val is not None else "N/A"
        e_str = f"{e_val}" if e_val is not None else "N/A"
        print(f"  {m:<20} {b_str:>10} {e_str:>10}")


def compare_health(base_health: dict, exp_health: dict):
    """Compare health check results."""
    print(f"\n  ── Health Check ──")
    base_ok = base_health.get('healthy', False) if base_health else False
    exp_ok = exp_health.get('healthy', False) if exp_health else False
    print(f"  Baseline: {'✅ Healthy' if base_ok else '⚠️ Issues'}")
    print(f"  Experiment: {'✅ Healthy' if exp_ok else '⚠️ Issues'}")

    # Compare per-check status
    base_checks = base_health.get('checks', {}) if base_health else {}
    exp_checks = exp_health.get('checks', {}) if exp_health else {}
    all_checks = sorted(set(list(base_checks.keys()) + list(exp_checks.keys())))
    for c in all_checks:
        b_status = base_checks.get(c, {}).get('status', 'N/A')
        e_status = exp_checks.get(c, {}).get('status', 'N/A')
        if b_status != e_status:
            print(f"  ⚠️ {c}: {b_status} → {e_status}")


def main():
    parser = argparse.ArgumentParser(description='Compare MiniLasHeR benchmarks')
    parser.add_argument('--baseline', required=True, help='Baseline benchmark JSON')
    parser.add_argument('--experiment', required=True, help='Experiment benchmark JSON')
    parser.add_argument('--filter', default='', help='Comma-separated gradient group filter')
    args = parser.parse_args()

    with open(args.baseline) as f:
        baseline = json.load(f)
    with open(args.experiment) as f:
        experiment = json.load(f)

    filters = [s.strip() for s in args.filter.split(',') if s.strip()] if args.filter else []

    print(f"\n{'='*60}")
    print(f"  Benchmark Comparison")
    print(f"{'='*60}")
    print(f"  Baseline:   {args.baseline}")
    print(f"    config:   {baseline.get('meta', {}).get('config', 'N/A')}")
    print(f"    time:     {baseline.get('meta', {}).get('timestamp', 'N/A')}")
    print(f"  Experiment: {args.experiment}")
    print(f"    config:   {experiment.get('meta', {}).get('config', 'N/A')}")
    print(f"    time:     {experiment.get('meta', {}).get('timestamp', 'N/A')}")
    print(f"{'='*60}")

    # Compare gradient summary
    base_grad = baseline.get('gradient_summary', {})
    exp_grad = experiment.get('gradient_summary', {})

    groups_to_compare = ['total_norm', 'backbone_mean', 'head_mean',
                         'tbsi_layer_mean', 'da_fusion_mean',
                         'degradation_mod_mean', 'temporal_token_mean']

    if filters:
        groups_to_compare = [g for g in groups_to_compare if any(f in g for f in filters)]

    for group in groups_to_compare:
        if group in base_grad or group in exp_grad:
            compare_grad(group, base_grad.get(group, {}), exp_grad.get(group, {}))
        else:
            print(f"\n  [{group}] Not found in either benchmark")

    # Compare loss
    print(f"\n{'='*60}")
    print(f"  LOSS COMPARISON")
    print(f"{'='*60}")
    compare_loss(baseline.get('loss_summary', {}), experiment.get('loss_summary', {}))

    # Compare evaluation
    print(f"\n{'='*60}")
    print(f"  EVALUATION COMPARISON")
    print(f"{'='*60}")
    compare_eval(baseline.get('eval_metrics', {}), experiment.get('eval_metrics', {}))

    # Compare health
    print(f"\n{'='*60}")
    print(f"  HEALTH CHECK COMPARISON")
    print(f"{'='*60}")
    compare_health(baseline.get('health', {}), experiment.get('health', {}))

    # Final verdict
    print(f"\n{'='*60}")
    b_auc = baseline.get('eval_metrics', {}).get('AUC', None)
    e_auc = experiment.get('eval_metrics', {}).get('AUC', None)
    if b_auc is not None and e_auc is not None:
        delta = e_auc - b_auc
        if delta > 1.0:
            verdict = "✅ 方向明确正确 (AUC > +1%)"
        elif delta > 0.3:
            verdict = "🟡 信号正向 (AUC +0.3%~1%)"
        elif delta > -0.3:
            verdict = "⚠️ 几乎没有效果 (|Δ| < 0.3%)"
        else:
            verdict = "🔴 有问题 (AUC 下降)"
        print(f"  AUC: {b_auc:.2f} → {e_auc:.2f} ({delta:+.2f})")
        print(f"  判断: {verdict}")

    b_health = baseline.get('health', {}).get('healthy', False)
    e_health = experiment.get('health', {}).get('healthy', False)
    if not b_health and e_health:
        print(f"  ✅ 实验修复了基线中的健康检查问题")
    elif b_health and not e_health:
        print(f"  ⚠️ 实验引入了新的健康检查问题")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
