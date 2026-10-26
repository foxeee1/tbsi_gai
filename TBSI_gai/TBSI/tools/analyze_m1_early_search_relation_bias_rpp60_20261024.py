#!/usr/bin/env python3
import csv, json, os
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'output/experiments/M1_EARLY_SEARCH_RELATION_BIAS_QUICK3_20261024'
MANIFEST = EXP / 'rpp_v2_60_manifest.txt'
RESULTS = ROOT / 'output/test/tracking_results/tbsi_track/M1_EARLY_SEARCH_RELATION_BIAS_QUICK3_20261024_relation_bias_rpp60'
os.environ['TBSI_RPP_TEST_LIST'] = str(MANIFEST)
import sys
sys.path.insert(0, str(ROOT))
from lib.test.evaluation import get_dataset
from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.test.utils.load_text import load_text

def eval_one(boxes, seq):
    gt = torch.as_tensor(seq.ground_truth_rect, dtype=torch.float64)
    visible = torch.as_tensor(seq.target_visible, dtype=torch.uint8) if seq.target_visible is not None else None
    overlap, center, center_norm, _ = calc_seq_err_robust(torch.as_tensor(boxes, dtype=torch.float64), gt, seq.dataset, visible)
    th = torch.arange(0., 1.0001, .05, dtype=torch.float64)
    success = (overlap[:, None] > th[None, :]).double().mean(0)
    return {'auc': float(success.mean()*100), 'op50': float(success[10]*100), 'op75': float(success[15]*100), 'precision': float((center <= 20).double().mean()*100), 'norm_precision': float((center_norm <= .2).double().mean()*100)}

def main():
    expected = [x.strip() for x in MANIFEST.read_text().splitlines() if x.strip()]
    seqs = get_dataset('rpp_lasher_test')
    if [s.name for s in seqs] != expected: raise RuntimeError('RPP manifest order mismatch')
    rows=[]
    for seq in seqs:
        p=RESULTS/(seq.name+'.txt')
        vals=eval_one(np.asarray(load_text(str(p), delimiter=('\t',','), dtype=np.float64)), seq)
        rows.append({'sequence':seq.name, **vals, 'frames':len(seq.ground_truth_rect)})
    keys=('auc','op50','op75','precision','norm_precision')
    summary={'sequence_count':len(rows), 'candidate_macro_metrics':{k:float(np.mean([r[k] for r in rows])) for k in keys}, 'per_sequence':rows}
    (EXP/'rpp60_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (EXP/'rpp60_per_sequence.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps({'sequence_count':summary['sequence_count'], 'candidate_macro_metrics':summary['candidate_macro_metrics']}, indent=2))
if __name__=='__main__': main()
