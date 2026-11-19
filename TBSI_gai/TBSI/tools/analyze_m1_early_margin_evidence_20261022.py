#!/usr/bin/env python3
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'output/experiments/M1_EARLY_MARGIN_FIXED_STATE_EVIDENCE_20261022/retry'
MANIFEST = ROOT / 'output/experiments/M1_PRECISION_SCREEN_MANIFEST_20261020/manifest.json'

def read_trace(root, name):
    p = root / name / 'frames.jsonl'
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return rows[-1]

def main():
    events = {x['sequence']: x for x in json.loads(MANIFEST.read_text())['selection']}
    rows = []
    for name, event in sorted(events.items()):
        cand = read_trace(EXP / 'candidate_trace', name)
        ctrl = read_trace(EXP / 'control_trace', name)
        cs, ps = cand['early_gate_stats'], ctrl['early_gate_stats']
        for i, (c, p) in enumerate(zip(cs, ps)):
            rows.append({
                'sequence': name, 'role': event.get('role', 'unknown'), 'tbsi_index': i,
                'rgb_margin': c['rgb_margin_mean'], 'tir_margin': c['tir_margin_mean'],
                'rgb_gate_candidate': c['gate_rgb_mean'], 'tir_gate_candidate': c['gate_tir_mean'],
                'rgb_gate_control': p['gate_rgb_mean'], 'tir_gate_control': p['gate_tir_mean'],
                'rgb_gate_delta': c['gate_rgb_mean'] - p['gate_rgb_mean'],
                'tir_gate_delta': c['gate_tir_mean'] - p['gate_tir_mean'],
                'center_margin_delta': cand['peak_diagnostics']['target_background_margin_raw'] - ctrl['peak_diagnostics']['target_background_margin_raw'],
                'center_rank_delta': cand['peak_diagnostics']['gt_rank_raw'] - ctrl['peak_diagnostics']['gt_rank_raw'],
            })
    fields = list(rows[0])
    with (EXP / 'evidence_per_location.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    arr = lambda key: np.asarray([r[key] for r in rows], dtype=float)
    summary = {
        'events': len(events), 'locations': len(rows),
        'rgb_margin_mean': float(arr('rgb_margin').mean()),
        'tir_margin_mean': float(arr('tir_margin').mean()),
        'rgb_gate_delta_abs_mean': float(np.abs(arr('rgb_gate_delta')).mean()),
        'tir_gate_delta_abs_mean': float(np.abs(arr('tir_gate_delta')).mean()),
        'rgb_gate_delta_mean': float(arr('rgb_gate_delta').mean()),
        'tir_gate_delta_mean': float(arr('tir_gate_delta').mean()),
        'center_margin_delta_mean': float(arr('center_margin_delta').mean()),
        'center_rank_delta_mean': float(arr('center_rank_delta').mean()),
        'center_rank_improved_events': int(sum(arr('center_rank_delta') < 0)),
        'center_rank_worsened_events': int(sum(arr('center_rank_delta') > 0)),
        'evidence_to_gate_corr_rgb': float(np.corrcoef(arr('rgb_margin'), arr('rgb_gate_delta'))[0,1]),
        'evidence_to_gate_corr_tir': float(np.corrcoef(arr('tir_margin'), arr('tir_gate_delta'))[0,1]),
    }
    (EXP / 'evidence_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__': main()
