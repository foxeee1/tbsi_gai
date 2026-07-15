"""
MiniLasHeR: 对照基准采集脚本

功能:
  1. 运行 MiniLasHeR 训练（使用现有 train.py）
  2. 训练完毕后解析 diagnostics CSV + 训练日志
  3. 运行 MiniLasHeR 测试集评估
  4. 汇总为 JSON 基准文件，含模块健康度检查

用法:
  python tools/run_mini_benchmark.py --config vitb_256_tbsi_32x4_4e4_miniLasHeR_15ep --save_prefix reference
  python tools/run_mini_benchmark.py --config vitb_256_tbsi_32x4_4e4_miniLasHeR_15ep --save_prefix exp_fix_detach

输出:
  output/benchmark/{save_prefix}_benchmark.json
"""

import argparse
import csv
import json
import os
import os.path as osp
import re
import sys
import time

import numpy as np


def parse_loss_from_log(log_file: str) -> list:
    """Parse training log to extract per-iteration loss values."""
    if not osp.exists(log_file):
        print(f"  [WARN] Log not found: {log_file}")
        return []

    records = []
    pat = re.compile(
        r'\[(train|val):\s*(\d+),\s*(\d+)\s*/\s*(\d+)\].*?'
        r'Loss/total:\s*([\d.]+).*?'
        r'Loss/giou:\s*([\d.]+).*?'
        r'Loss/l1:\s*([\d.]+).*?'
        r'Loss/location:\s*([\d.]+).*?'
        r'IoU:\s*([\d.]+)'
    )
    with open(log_file) as f:
        for line in f:
            m = pat.search(line)
            if m:
                records.append({
                    'mode': m.group(1), 'epoch': int(m.group(2)),
                    'iter': int(m.group(3)),
                    'Loss/total': float(m.group(5)),
                    'Loss/giou': float(m.group(6)),
                    'Loss/l1': float(m.group(7)),
                    'Loss/location': float(m.group(8)),
                    'IoU': float(m.group(9)),
                })
    return records


def parse_diag_csv(csv_path: str) -> list:
    """Parse a diagnostics CSV into a list of dicts."""
    if not osp.exists(csv_path):
        return []
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = {}
            for k, v in row.items():
                k = k.strip()
                try:
                    clean[k] = int(v) if k in ('step', 'epoch', 'total_params', 'zero_grad_count') else float(v)
                except ValueError:
                    clean[k] = v
            rows.append(clean)
    return rows


def summarize_records(records: list) -> dict:
    """Numeric summary of a list of dicts."""
    if not records:
        return {}
    keys = records[0].keys()
    num_keys = [k for k in keys if k not in ('step', 'epoch')]
    out = {}
    for k in num_keys:
        vals = [r.get(k, None) for r in records if isinstance(r.get(k), (int, float))]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        out[k] = {
            'mean': round(float(np.mean(vals)), 6),
            'std': round(float(np.std(vals)), 6),
            'min': round(float(np.min(vals)), 6),
            'max': round(float(np.max(vals)), 6),
            'first': round(float(vals[0]), 6),
            'final': round(float(vals[-1]), 6),
        }
    return out


def summarize_loss_by_epoch(records: list) -> dict:
    """Aggregate loss records into per-epoch stats."""
    by_epoch = {}
    for r in records:
        by_epoch.setdefault(r['epoch'], []).append(r)
    summary = {}
    for ep in sorted(by_epoch):
        losses = by_epoch[ep]
        total = [l.get('Loss/total', 0) for l in losses]
        summary[f'epoch_{ep}'] = {
            'Loss/total_mean': round(float(np.mean(total)), 6),
            'Loss/total_min': round(float(np.min(total)), 6),
            'Loss/total_final': round(float(total[-1]), 6),
            'num_records': len(losses),
        }
    return summary


def run_training(config_name: str, output_dir: str) -> float:
    """Run training via tracking/train.py, return elapsed seconds."""
    cmd = (
        f"unset OMP_NUM_THREADS && "
        f"python tracking/train.py --script tbsi_track "
        f"--config {config_name} --save_dir {output_dir} --mode single"
    )
    print(f"\n{'='*60}")
    print(f"  TRAINING: {config_name}")
    print(f"  CMD: {cmd}")
    print(f"{'='*60}\n")

    t0 = time.time()
    ret = os.system(cmd)
    elapsed = time.time() - t0

    if ret != 0:
        print(f"\n  [WARN] Training exit code = {ret}")
    print(f"  Training took {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return elapsed


def run_evaluation(config_name: str, output_dir: str) -> dict:
    """Run evaluation on mini_lasher_test, return parsed metrics."""
    print(f"\n{'='*60}")
    print(f"  EVALUATION: {config_name} on mini_lasher_test")
    print(f"{'='*60}\n")

    # Test
    test_cmd = (
        f"python tracking/test.py tbsi_track {config_name} "
        f"--dataset_name mini_lasher_test --threads 0 --num_gpus 1"
    )
    t0 = time.time()
    os.system(test_cmd)
    print(f"  Test took {time.time()-t0:.0f}s")

    # Analysis
    analysis_cmd = (
        f"python tracking/analysis_results.py "
        f"--tracker_name tbsi_track --tracker_param {config_name} "
        f"--dataset_name mini_lasher_test"
    )
    os.system(analysis_cmd)

    # Collect result files
    results = {}
    results_root = osp.join(output_dir, 'test', 'tbsi_track', config_name)
    if osp.isdir(results_root):
        for d in os.listdir(results_root):
            dpath = osp.join(results_root, d)
            if not osp.isdir(dpath):
                continue
            for fname in os.listdir(dpath):
                if fname.endswith('.json'):
                    try:
                        with open(osp.join(dpath, fname)) as f:
                            data = json.load(f)
                            if isinstance(data, dict):
                                results.update(data)
                    except Exception:
                        pass
    return results


def health_check(grad_summary: dict, loss_summary: dict) -> dict:
    """Run basic health checks and return results."""
    checks = {}
    issues = []

    # 1. Gradient health
    groups = ['da_fusion', 'degradation_mod', 'temporal_token', 'tbsi_layer', 'head']
    for g in groups:
        key = f'{g}_mean'
        if key in grad_summary:
            final_val = grad_summary[key].get('final', -1)
            max_val = grad_summary[key].get('max', -1)
            if final_val > 1e-8:
                checks[f'{g}_grad'] = {'status': 'PASS', 'final_grad': final_val}
            elif final_val < 0:
                checks[f'{g}_grad'] = {'status': 'N/A', 'final_grad': final_val}
            else:
                checks[f'{g}_grad'] = {'status': 'FAIL', 'final_grad': final_val,
                                        'issue': f'{g} gradient near zero'}
                issues.append(f'{g}_grad')

    # 2. Loss trend
    if len(loss_summary) >= 2:
        first_ep = list(loss_summary.keys())[0]
        last_ep = list(loss_summary.keys())[-1]
        first_loss = loss_summary[first_ep]['Loss/total_mean']
        last_loss = loss_summary[last_ep]['Loss/total_mean']
        if last_loss < first_loss * 0.85:
            checks['loss_trend'] = {'status': 'PASS', 'trend': f'{first_loss:.4f}→{last_loss:.4f}'}
        else:
            checks['loss_trend'] = {'status': 'WARN', 'trend': f'{first_loss:.4f}→{last_loss:.4f}',
                                     'issue': 'loss not decreasing enough'}
            issues.append('loss_trend')

    # 3. Total gradient health
    if 'total_norm' in grad_summary:
        final_norm = grad_summary['total_norm'].get('final', 0)
        if 1e-8 < final_norm < 100:
            checks['total_grad'] = {'status': 'PASS', 'norm': final_norm}
        elif final_norm >= 100:
            checks['total_grad'] = {'status': 'WARN', 'norm': final_norm, 'issue': 'gradient explosion'}
            issues.append('grad_explosion')
        elif final_norm <= 1e-8:
            checks['total_grad'] = {'status': 'FAIL', 'norm': final_norm, 'issue': 'gradient vanishing'}
            issues.append('grad_vanishing')

    all_pass = not any(c.get('status') == 'FAIL' for c in checks.values())
    return {
        'summary': '✅ All checks pass' if all_pass else f'⚠️ Issues: {issues}',
        'healthy': all_pass,
        'checks': checks,
        'issues': issues,
    }


def collect_benchmark(config_name: str, save_prefix: str, output_dir: str = './output'):
    """Run training + parse diagnostics + evaluate → save benchmark JSON."""

    benchmark_dir = osp.join(output_dir, 'benchmark')
    os.makedirs(benchmark_dir, exist_ok=True)
    out_path = osp.join(benchmark_dir, f'{save_prefix}_benchmark.json')

    if osp.exists(out_path):
        print(f"  [SKIP] {out_path} exists. Delete to re-run.")
        with open(out_path) as f:
            return json.load(f)

    # ---- Step 1: Training ----
    elapsed = run_training(config_name, output_dir)

    # ---- Step 2: Parse diagnostics ----
    log_dir = osp.join(output_dir, 'logs')
    diag = {}
    for fname in ['gradient_flow.csv', 'loss_components.csv', 'internal_signals.csv']:
        rows = parse_diag_csv(osp.join(log_dir, fname))
        diag[fname.replace('.csv', '')] = {
            'num_records': len(rows),
            'first': rows[0] if rows else None,
            'last': rows[-1] if rows else None,
            'summary': summarize_records(rows),
        }

    # ---- Step 3: Parse loss log ----
    log_file = osp.join(log_dir, f'tbsi_track-{config_name}.log')
    loss_history = parse_loss_from_log(log_file)
    loss_summary = summarize_loss_by_epoch(loss_history)

    # ---- Step 4: Health checks ----
    grad_summary = diag.get('gradient_flow', {}).get('summary', {})
    health = health_check(grad_summary, loss_summary)

    # ---- Step 5: Evaluation ----
    eval_metrics = run_evaluation(config_name, output_dir)

    # ---- Step 6: Assemble benchmark ----
    benchmark = {
        'meta': {
            'config': config_name,
            'prefix': save_prefix,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'train_elapsed_s': round(elapsed, 1),
            'dataset': 'MiniLasHeR (30 seq, ~22K frames)',
        },
        'gradient_summary': grad_summary,
        'loss_summary': loss_summary,
        'loss_history': loss_history[-500:],  # keep tail to limit file size
        'eval_metrics': eval_metrics,
        'diagnostics': {
            k: {'num_records': v['num_records'], 'summary': v['summary']}
            for k, v in diag.items()
        },
        'health': health,
    }

    with open(out_path, 'w') as f:
        json.dump(benchmark, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  BENCHMARK SAVED: {out_path}")
    print(f"  Health: {health['summary']}")
    if eval_metrics:
        auc = eval_metrics.get('AUC', 'N/A')
        pr = eval_metrics.get('Precision', 'N/A')
        print(f"  Eval: AUC={auc}, PR={pr}")
    print(f"{'='*60}\n")

    return benchmark


def main():
    parser = argparse.ArgumentParser(description='MiniLasHeR Benchmark Collection')
    parser.add_argument('--config', default='vitb_256_tbsi_32x4_4e4_miniLasHeR_15ep')
    parser.add_argument('--save_prefix', default='reference')
    parser.add_argument('--output_dir', default='./output')
    args = parser.parse_args()
    collect_benchmark(args.config, args.save_prefix, args.output_dir)


if __name__ == '__main__':
    main()
