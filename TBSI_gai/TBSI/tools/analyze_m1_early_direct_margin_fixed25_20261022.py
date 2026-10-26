#!/usr/bin/env python3
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'output/experiments/M1_EARLY_DIRECT_MARGIN_FIXED25_20261022'
MANIFEST = ROOT / 'output/experiments/M1_PRECISION_SCREEN_MANIFEST_20261020/manifest.json'

def trace(root, name):
    p = root / name / 'frames.jsonl'
    return json.loads(p.read_text().splitlines()[-1])

def main():
    events = {x['sequence']: x for x in json.loads(MANIFEST.read_text())['selection']}
    rows=[]
    for name,event in sorted(events.items()):
        c=trace(EXP/'candidate_trace',name); z=trace(EXP/'control_trace',name)
        for i,(a,b) in enumerate(zip(c['early_gate_stats'],z['early_gate_stats'])):
            rows.append({'sequence':name,'role':event.get('role','unknown'),'tbsi_index':i,
                'rgb_margin':a['rgb_margin_mean'],'tir_margin':a['tir_margin_mean'],
                'rgb_gate_delta':a['gate_rgb_mean']-b['gate_rgb_mean'],
                'tir_gate_delta':a['gate_tir_mean']-b['gate_tir_mean'],
                'center_margin_delta':c['peak_diagnostics']['target_background_margin_raw']-z['peak_diagnostics']['target_background_margin_raw'],
                'center_rank_delta':c['peak_diagnostics']['gt_rank_raw']-z['peak_diagnostics']['gt_rank_raw']})
    with (EXP/'evidence_per_location.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    a=lambda k:np.asarray([r[k] for r in rows],dtype=float)
    s={'events':len(events),'locations':len(rows),
       'rgb_margin_mean':float(a('rgb_margin').mean()),'tir_margin_mean':float(a('tir_margin').mean()),
       'rgb_gate_delta_abs_mean':float(np.abs(a('rgb_gate_delta')).mean()),'tir_gate_delta_abs_mean':float(np.abs(a('tir_gate_delta')).mean()),
       'rgb_gate_delta_mean':float(a('rgb_gate_delta').mean()),'tir_gate_delta_mean':float(a('tir_gate_delta').mean()),
       'center_margin_delta_mean':float(a('center_margin_delta').mean()),'center_rank_delta_mean':float(a('center_rank_delta').mean()),
       'center_rank_improved_events':int(sum(a('center_rank_delta')<0)),'center_rank_worsened_events':int(sum(a('center_rank_delta')>0)),
       'evidence_to_gate_corr_rgb':float(np.corrcoef(a('rgb_margin'),a('rgb_gate_delta'))[0,1]),
       'evidence_to_gate_corr_tir':float(np.corrcoef(a('tir_margin'),a('tir_gate_delta'))[0,1])}
    (EXP/'evidence_summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
if __name__=='__main__':main()
