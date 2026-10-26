"""Sequence-held-out utility/no-op gate screen for existing M1 fixed states."""
import argparse, csv, json
from pathlib import Path
import numpy as np


def fit(x, y, alpha=10.0):
    mu=x.mean(0); sd=x.std(0); sd[sd<1e-6]=1.; z=np.clip((x-mu)/sd,-8,8.)
    return mu, sd, float(y.mean()), np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]), z.T@(y-y.mean()))


def pred(m, x):
    mu,sd,b,w=m; return b+np.clip((x-mu)/sd,-8,8.)@w


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output-dir',required=True); args=ap.parse_args()
    rows=list(csv.DictReader(open(args.input))); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    feature_sets={
      'rank_margin':['base_gt_rank','base_margin'],
      'signals':['base_gt_rank','base_margin','energy_gap_rgb_minus_tir','template_support_gap_rgb_minus_tir','template_psr_gap_rgb_minus_tir','conflict'],
    }
    seqs=sorted(set(r['sequence'] for r in rows)); y_rgb=np.array([float(r['rgb_utility']) for r in rows]); y_tir=np.array([float(r['tir_utility']) for r in rows])
    xall={k:np.array([[float(r[f]) for f in fs] for r in rows],float) for k,fs in feature_sets.items()}
    records=[]; aggregate={}
    for model, x in xall.items():
      for threshold in [0.0,0.0025,0.005,0.01,0.02]:
        selected=[]; oracle=[]; harmful=[]; covered=[]; pred_rows=[]
        for s in seqs:
          tr=np.array([r['sequence']!=s for r in rows]); te=~tr
          mr=fit(x[tr],y_rgb[tr]); mt=fit(x[tr],y_tir[tr])
          pr,pt=pred(mr,x[te]),pred(mt,x[te]); actual_r,actual_t=y_rgb[te],y_tir[te]
          for j in range(te.sum()):
            choices=[0.,float(pr[j]),float(pt[j])]; k=int(np.argmax(choices)); chosen_pred=choices[k]
            use=chosen_pred>threshold
            actual=choices_actual=[0.,float(actual_r[j]),float(actual_t[j])]
            val=actual[k] if use else 0.
            selected.append(val); oracle.append(max(actual)); covered.append(use); harmful.append(use and val < -1e-4)
            pred_rows.append({'sequence':s,'threshold':threshold,'selected_action':('noop' if not use else ('rgb' if k==1 else 'tir')),'selected_utility':val,'oracle_utility':max(actual),'predicted_utility':chosen_pred})
        key=model+'@'+str(threshold)
        by_seq={}
        for rr in pred_rows: by_seq.setdefault(rr['sequence'],[]).append(rr)
        seq_selected=[np.mean([rr['selected_utility'] for rr in vv]) for vv in by_seq.values()]
        seq_oracle=[np.mean([rr['oracle_utility'] for rr in vv]) for vv in by_seq.values()]
        aggregate[key]={'model':model,'threshold':threshold,'n_states':len(selected),'n_sequences':len(by_seq),'selected_mean_utility':float(np.mean(selected)),'sequence_macro_selected_utility':float(np.mean(seq_selected)),'oracle_mean_utility':float(np.mean(oracle)),'sequence_macro_oracle_utility':float(np.mean(seq_oracle)),'oracle_fraction':float(np.mean(selected)/max(np.mean(oracle),1e-12)),'sequence_macro_oracle_fraction':float(np.mean(seq_selected)/max(np.mean(seq_oracle),1e-12)),'coverage':float(np.mean(covered)),'harmful_rate_all':float(np.mean(harmful)),'harmful_rate_covered':float(np.sum(harmful)/max(np.sum(covered),1))}
        records.extend(pred_rows)
    json.dump({'id':'M1-UTILITY-NOOP-GATE-SCREEN-20261014','input':args.input,'sequence_holdout':True,'models':aggregate,'limitations':['fixed-state target attenuation utility proxy','not a trained closed-loop gate','GT defines offline utility only']},open(out/'summary.json','w'),indent=2)
    with open(out/'per_state_selection.csv','w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    print(json.dumps(aggregate,indent=2))

if __name__=='__main__': main()
