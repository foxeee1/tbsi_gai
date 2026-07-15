"""
综合评测：全指标 + 分场景 + 多实验对比

从 eval_data.pkl 中提取所有指标，并输出到 JSON + 表格。

用法:
  python tools/comprehensive_eval.py 完美基线小数据 buggy初始 fix_B2_dafusion
  python tools/comprehensive_eval.py --compare buggy初始 fix_B2_dafusion
"""

import argparse
import json
import os
import os.path as osp
import pickle
import sys

import numpy as np
import torch

# ===== MiniLasHeR 测试集的 5 类场景定义 =====
# (与 mini_lasher.py 中的 MINI_TEST_SEQUENCES 保持一致)
MINI_TEST_SEQUENCES = [
    # Normal (6)
    "1boycoming", "2girl", "baggirl", "basketboy", "blackboy", "bike",
    # Illumination (6)
    "belowdarkgirl", "boyfromdark", "darkgirl", "girlfromlight_quezhen",
    "carlightcome2", "boyinsnowfield3",
    # ThermalCross (6)
    "ab_pingpongball2", "ab_bolstershaking", "ab_blkskirtgirl",
    "ab_rightlowerredcup_quezhen", "ab_whiteboywithbluebag", "ab_girlcrossroad",
    # Occlusion (6)
    "boy2trees", "boyaftertree", "carbehindtrees", "boy2buildings",
    "girlafterglassdoor", "ab_bikeoccluded",
    # FastMotion (6)
    "10runone", "boyruninsnow", "mototurneast", "bikeboyturntimes",
    "runningcameragirl", "boytakingbasketballfollowing",
]

CATEGORIES = {
    'Normal': MINI_TEST_SEQUENCES[0:6],
    'Illumination': MINI_TEST_SEQUENCES[6:12],
    'ThermalCross': MINI_TEST_SEQUENCES[12:18],
    'Occlusion': MINI_TEST_SEQUENCES[18:24],
    'FastMotion': MINI_TEST_SEQUENCES[24:30],
}


def compute_metrics(eval_data: dict) -> dict:
    """从 eval_data 计算所有指标的全量字典。"""
    overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])  # [N_seq, N_trk, N_thresh]
    center = torch.tensor(eval_data['ave_success_rate_plot_center'])
    center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])
    seq_names = eval_data['sequences']
    valid = torch.tensor(eval_data['valid_sequence']).bool()

    # 每序列的 overlap 均值
    avg_overlap_seq = overlap.mean(dim=-1)  # [N_seq, N_trk]

    # 全部指标 (对所有 valid 序列平均)
    n_trk = overlap.shape[1]
    results = {}
    for trk_idx in range(n_trk):
        # 筛选 valid 序列
        valid_overlap = overlap[valid, trk_idx]  # [N_valid, N_thresh]
        valid_center = center[valid, trk_idx]
        valid_center_norm = center_norm[valid, trk_idx]
        valid_avg_overlap = avg_overlap_seq[valid, trk_idx]

        # AUC = 各阈值成功率均值
        auc = valid_overlap.mean(dim=0).mean().item() * 100

        # OP50/OP75 = 对应阈值处的成功率
        thresh = eval_data['threshold_set_overlap']
        op50_thresh = 0.50
        op75_thresh = 0.75
        op50_idx = min(range(len(thresh)), key=lambda i: abs(thresh[i] - op50_thresh))
        op75_idx = min(range(len(thresh)), key=lambda i: abs(thresh[i] - op75_thresh))
        op50 = valid_overlap[:, op50_idx].mean().item() * 100
        op75 = valid_overlap[:, op75_idx].mean().item() * 100

        # Precision @ 20px
        center_thresh = eval_data['threshold_set_center']
        prec_idx = min(range(len(center_thresh)), key=lambda i: abs(center_thresh[i] - 20))
        precision = valid_center[:, prec_idx].mean().item() * 100

        # Norm Precision @ 0.20
        norm_thresh = eval_data['threshold_set_center_norm']
        nprec_idx = min(range(len(norm_thresh)), key=lambda i: abs(norm_thresh[i] - 0.20))
        norm_precision = valid_center_norm[:, nprec_idx].mean().item() * 100

        results = {
            'AUC': round(auc, 2),
            'OP50': round(op50, 2),
            'OP75': round(op75, 2),
            'Precision': round(precision, 2),
            'Norm_Precision': round(norm_precision, 2),
        }

    # ===== 分场景指标 =====
    seq_list = [seq_names[i] for i, v in enumerate(valid) if v]
    cat_results = {}
    for cat_name, cat_seqs in CATEGORIES.items():
        cat_indices = [i for i, s in enumerate(seq_list) if s in cat_seqs]
        if not cat_indices:
            cat_results[cat_name] = {'AUC': 'N/A', 'count': 0}
            continue
        cat_overlap = overlap[valid, 0][cat_indices]  # [N_cat, N_thresh]
        cat_auc = cat_overlap.mean(dim=0).mean().item() * 100

        cat_center = center[valid, 0][cat_indices]
        cat_prec = cat_center[:, prec_idx].mean().item() * 100

        cat_center_norm = center_norm[valid, 0][cat_indices]
        cat_nprec = cat_center_norm[:, nprec_idx].mean().item() * 100

        cat_results[cat_name] = {
            'AUC': round(cat_auc, 2),
            'Precision': round(cat_prec, 2),
            'Norm_Precision': round(cat_nprec, 2),
            'count': len(cat_indices),
        }

    results['per_category'] = cat_results
    return results


def load_experiment(exp_name: str, base_dir: str = 'output/experiments') -> dict:
    """加载一个实验的 eval_data，返回所有指标。"""
    search_paths = [
        osp.join(base_dir, exp_name, 'test_analysis'),
        osp.join(base_dir, exp_name, 'test_results'),
        osp.join('output/test/result_plots', exp_name),
        osp.join('output/test/result_plots', 'mini_lasher_test'),
    ]
    for p in search_paths:
        ep = osp.join(p, 'eval_data.pkl')
        if osp.exists(ep):
            with open(ep, 'rb') as f:
                eval_data = pickle.load(f)
            metrics = compute_metrics(eval_data)
            metrics['_source'] = ep
            return metrics
    return None


def format_table_row(name: str, metrics: dict) -> str:
    auc = metrics.get('AUC', 'N/A')
    op50 = metrics.get('OP50', 'N/A')
    op75 = metrics.get('OP75', 'N/A')
    prec = metrics.get('Precision', 'N/A')
    nprec = metrics.get('Norm_Precision', 'N/A')
    return f"  {name:<30} {auc:>8} {op50:>8} {op75:>8} {prec:>8} {nprec:>8}"


def format_category_row(name: str, cat_metrics: dict, indent: str = "    ") -> str:
    auc = cat_metrics.get('AUC', 'N/A')
    prec = cat_metrics.get('Precision', 'N/A')
    nprec = cat_metrics.get('Norm_Precision', 'N/A')
    count = cat_metrics.get('count', '?')
    return f"  {indent}{name:<20} {auc:>8} {prec:>8} {nprec:>8}  (n={count})"


def main():
    parser = argparse.ArgumentParser(description='Comprehensive evaluation')
    parser.add_argument('experiments', nargs='+', help='Experiment names')
    parser.add_argument('--base_dir', default='output/experiments')
    parser.add_argument('--output', '-o', default=None, help='Save JSON to file')
    parser.add_argument('--compare', action='store_true',
                        help='Compare experiments side-by-side')
    args = parser.parse_args()

    all_results = {}
    for exp_name in args.experiments:
        metrics = load_experiment(exp_name, args.base_dir)
        if metrics is None:
            print(f"  ❌ {exp_name}: eval_data.pkl not found")
            continue
        all_results[exp_name] = metrics

    # ===== 输出 =====
    print(f"\n{'='*70}")
    print(f"  综合评测报告")
    print(f"{'='*70}\n")

    if args.compare and len(all_results) >= 2:
        # 对比模式：并列表格
        exp_names = list(all_results.keys())
        n_exp = len(exp_names)
        print(f"  全局指标对比:")
        header = f"  {'实验':<25}"
        for _ in exp_names:
            header += f" {'AUC':>7} {'OP50':>7} {'OP75':>7} {'PR':>7} {'NPR':>7}"
        print(header)
        print(f"  {'-'*25}{'-'*38*n_exp}")

        for exp in exp_names:
            m = all_results[exp]
            auc = m.get('AUC', 'N/A')
            op50 = m.get('OP50', 'N/A')
            op75 = m.get('OP75', 'N/A')
            prec = m.get('Precision', 'N/A')
            nprec = m.get('Norm_Precision', 'N/A')
            row = f"  {exp:<25}"
            row += f" {auc:>7} {op50:>7} {op75:>7} {prec:>7} {nprec:>7}"
            print(row)
            if exp != exp_names[-1]:
                delta = f"  {'↓'*25}"
                print(delta)

        print()

        # 分场景对比
        print(f"  分场景 AUC 对比:")
        cat_header = f"  {'场景':<20}"
        for _ in exp_names:
            cat_header += f" {'AUC':>7} {'PR':>7} {'NPR':>7}"
        print(cat_header)
        print(f"  {'-'*20}{'-'*24*n_exp}")

        for cat_name in CATEGORIES:
            row = f"  {cat_name:<20}"
            for exp in exp_names:
                cat_m = all_results[exp].get('per_category', {}).get(cat_name, {})
                auc = cat_m.get('AUC', 'N/A')
                prec = cat_m.get('Precision', 'N/A')
                nprec = cat_m.get('Norm_Precision', 'N/A')
                row += f" {str(auc):>7} {str(prec):>7} {str(nprec):>7}"
            print(row)

    else:
        # 逐个打印
        for exp_name, metrics in all_results.items():
            print(f"  ── {exp_name} ──")
            print(f"  AUC={metrics['AUC']}, OP50={metrics['OP50']}, "
                  f"OP75={metrics['OP75']}, PR={metrics['Precision']}, "
                  f"NPR={metrics['Norm_Precision']}")
            if 'per_category' in metrics:
                print()
                for cat_name, cat_m in metrics['per_category'].items():
                    print(f"    {cat_name:<15} AUC={cat_m['AUC']:>6.2f}  "
                          f"PR={cat_m['Precision']:>6.2f}  "
                          f"NPR={cat_m['Norm_Precision']:>6.2f}  "
                          f"(n={cat_m['count']})")
            print()

    # 保存 JSON
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  已保存到: {args.output}")

    # 格式化的 JSON 输出给 pipeline 解析用
    print(f"\n  {'='*70}")
    print(f"  JSON 摘要:")
    summary = {}
    for exp in all_results:
        m = all_results[exp]
        summary[exp] = {
            'AUC': m.get('AUC'),
            'OP50': m.get('OP50'),
            'OP75': m.get('OP75'),
            'Precision': m.get('Precision'),
            'Norm_Precision': m.get('Norm_Precision'),
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
