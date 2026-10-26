#!/usr/bin/env python3
"""Machine gate for G0-Expand; permits at most the Cplain planning step."""
import argparse,json
from pathlib import Path
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--summary',required=True); ap.add_argument('--parity',required=True); ap.add_argument('--output',required=True); args=ap.parse_args(); d=json.loads(Path(args.summary).read_text()); parity=json.loads(Path(args.parity).read_text()); checks={'fresh_export_parity':bool(parity.get('passed')) and parity.get('max_abs_box_diff')==0.0}
 for seed in map(str,d['seeds']):
  s=d[seed]; checks[seed]={'source_positive':s['L_E']['mean_delta_vs_L_zero']>0,'source_threshold':s['L_E']['mean_delta_vs_L_zero']>=0.03,'beats_lcopy':s['L_E']['mean_delta_vs_L_zero']>s['L_Lcopy']['mean_delta_vs_L_zero'],'beats_spatial':s['L_E']['mean_delta_vs_L_zero']>s['L_E_spatial']['mean_delta_vs_L_zero'],'beats_event':s['L_E']['mean_delta_vs_L_zero']>s['L_E_event']['mean_delta_vs_L_zero'],'rank_direction':s['L_E']['mean_best_rank']<s['L_zero']['mean_best_rank']}
 passed=checks['fresh_export_parity'] and all(all(v for v in x.values()) for k,x in checks.items() if k != 'fresh_export_parity'); decision='CONDITIONAL_PASS_CPLAIN_ONLY' if passed else 'STOP_OR_INCONCLUSIVE_NO_TRAIN'
 out={'decision':decision,'checks':checks,'fresh_parity':parity,'training_authorized':False,'next_allowed':'Cplain frozen-M1 closed-loop diagnostic only' if passed else 'no module training','evidence':str(Path(args.summary).resolve())}; Path(args.output).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2)); raise SystemExit(0 if passed else 2)
if __name__=='__main__': main()
