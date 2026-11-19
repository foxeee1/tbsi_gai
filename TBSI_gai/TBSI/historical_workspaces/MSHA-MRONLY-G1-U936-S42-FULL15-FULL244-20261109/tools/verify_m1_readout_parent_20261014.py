#!/usr/bin/env python3
from __future__ import print_function
import csv
import json
import os
import subprocess
import time
from pathlib import Path

from run_m1_readout_structures_proxy_20261014 import (
    ROOT, PYTHON, CONFIG, MANIFEST, M1_RESULTS, OLD_RESULTS, metrics, summarize,
)

def main():
    out = ROOT / 'output/experiments/M1_READOUT_STRUCTURES_PROXY_20261014'
    parent_dir = ROOT / 'output/test/tracking_results/tbsi_track' / (CONFIG + '_1024')
    ckpt = ROOT / 'output/experiments' / CONFIG / 'checkpoints/TBSITrack_ep0015.pth.tar'
    if parent_dir.exists():
        raise RuntimeError('refusing existing parent output {}'.format(parent_dir))
    env = os.environ.copy()
    for key in ('TBSI_BOUNDED_READOUT_PREDICTOR','TBSI_READOUT_POLICY','TBSI_READOUT_AUDIT_DIR'):
        env.pop(key, None)
    env.update({'TBSI_RPP_TEST_LIST':str(MANIFEST),'TBSI_INFER_FP16':'0',
                'TBSI_CHECKPOINT_OVERRIDE':str(ckpt),'OMP_NUM_THREADS':'1',
                'MKL_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'})
    cmd=[PYTHON,'-u','tracking/test.py','tbsi_track',CONFIG,'--runid','1024',
         '--dataset_name','rpp_lasher_test','--threads','0','--num_gpus','1']
    log=out/'M1_CURRENT_PARENT_1024.log'
    print('START current-code M1 parent run1024',flush=True)
    with log.open('w') as stream:
        subprocess.run(cmd,cwd=str(ROOT),env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    if not parent_dir.is_dir():
        raise RuntimeError('parent run produced no result directory')
    seqs=[x.strip() for x in MANIFEST.read_text().splitlines() if x.strip()]
    roots={'M1_CURRENT_1024':parent_dir,'M1_HISTORIC_9904':M1_RESULTS,'OLD_PREDICTOR_9915':OLD_RESULTS}
    for label,runid in (('S1_INDEPENDENT',1021),('S2_RELIABILITY',1022),('S3_HYSTERESIS',1023)):
        roots[label]=ROOT/'output/test/tracking_results/tbsi_track'/(CONFIG+'_'+str(runid))
    base={s:metrics(parent_dir,s) for s in seqs}
    rows=[]
    for s in seqs:
        row={'sequence':s,'cohort':'lowauc62' if s in __import__('run_m1_readout_structures_proxy_20261014').LOW else 'normal20'}
        for label,root in roots.items():
            current=base[s] if label=='M1_CURRENT_1024' else metrics(root,s)
            for m in ('AUC','PR','NPR','OP50','OP75'):
                row[label+'_'+m]=current[m]
                row[label+'_delta_'+m]=current[m]-base[s][m]
        rows.append(row)
    with (out/'per_sequence_matched_current_parent.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result={'parent_runid':1024,'parent_result_dir':str(parent_dir),'parent_command':cmd,
            'dataset_manifest':str(MANIFEST),'dataset_manifest_sha256':__import__('hashlib').sha256(MANIFEST.read_bytes()).hexdigest(),
            'checkpoint':str(ckpt),'arms':{},'per_sequence_csv':str(out/'per_sequence_matched_current_parent.csv')}
    for label in roots:
        result['arms'][label]={}
        for cohort in ('lowauc62','normal20','union82'):
            subset=rows if cohort=='union82' else [r for r in rows if r['cohort']==cohort]
            d={k:float(sum(r[label+'_delta_'+k] for r in subset)/len(subset)) for k in ('AUC','PR','NPR','OP50','OP75')}
            wins={k:int(sum(r[label+'_delta_'+k]>0 for r in subset)) for k in ('AUC','PR','NPR','OP50','OP75')}
            losses={k:int(sum(r[label+'_delta_'+k]<0 for r in subset)) for k in ('AUC','PR','NPR','OP50','OP75')}
            d.update({'wins':wins,'losses':losses,'n':len(subset)})
            result['arms'][label][cohort]=d
    (out/'matched_current_parent_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['arms'],indent=2),flush=True)

if __name__=='__main__': main()
