#!/usr/bin/env python3
"""Lightweight serial supervisor: train exit -> pair check -> eval -> mechanism -> report."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'output/experiments/MSHA-MR-RESIDUAL-TARGET-DIRECTION-QUICK-S42-20261115'
PARENT=ROOT/'output/experiments/MSHA-MR-RECEIVER-MESSAGE-RESIDUAL-QUICK-S42-20261115'

def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def pairing():
    a=json.loads((RUN/'candidate_retry1/initial_state.json').read_text());b=json.loads((PARENT/'candidate_final/initial_state.json').read_text())
    ba=[json.loads(s) for s in (RUN/'candidate_retry1/batches.jsonl').read_text().splitlines()]
    bb=[json.loads(s) for s in (PARENT/'candidate_final/batches.jsonl').read_text().splitlines()]
    completed=json.loads((RUN/'candidate_retry1/completed.json').read_text())
    checks=dict(common_initial_state_equal=a['shared_state_sha256']==b['shared_state_sha256'],
        exact_augmented_input_sample_metadata_equal=ba==bb,actual_microbatches=len(ba),expected_microbatches=3744,
        successful_updates=completed['successful_updates'],sampled_pairs=completed['samples'],train_sequences=completed['unique_sequences'],
        same_train_manifest=json.loads((RUN/'candidate_retry1/started.json').read_text())['manifest_sha256']==json.loads((PARENT/'candidate_final/started.json').read_text())['manifest_sha256'])
    checks['all_checks_pass']=all([checks['common_initial_state_equal'],checks['exact_augmented_input_sample_metadata_equal'],checks['same_train_manifest'],len(ba)==3744,completed['successful_updates']==468,completed['samples']==59904,completed['unique_sequences']==977])
    write(RUN/'pairing.json',checks)
    if not checks['all_checks_pass']:raise RuntimeError('Known pairing confound; ranking suspended. See pairing.json')

def main():
    p=argparse.ArgumentParser();p.add_argument('--training-pid',type=int,required=True);args=p.parse_args()
    assert str(ROOT.resolve()).startswith('/root/autodl-tmp/') and str(RUN.resolve()).startswith('/root/autodl-tmp/')
    write(RUN/'pipeline_status.json',dict(stage='waiting_for_training_process_exit',training_pid=args.training_pid,supervisor_pid=os.getpid(),time=time.time()))
    try:
        while True:
            try:os.kill(args.training_pid,0)
            except ProcessLookupError:break
            time.sleep(10)
        if not (RUN/'candidate_retry1/completed.json').exists():raise RuntimeError('Training exited without completed.json')
        pairing()
        stages=[('fixed96_evaluation',['tools/run_receiver_direction_quick_20261115.py','evaluate','--output',str(RUN/'eval_candidate')]),
                ('trained_direction_mechanism',['tools/probe_receiver_direction_learned_20261115.py']),
                ('final_metrics_and_registry',['tools/finalize_receiver_direction_quick_20261115.py'])]
        for stage,command in stages:
            write(RUN/'pipeline_status.json',dict(stage=stage,status='RUNNING',time=time.time(),command=[sys.executable,*command]))
            with (RUN/(stage+'.log')).open('w') as log:
                subprocess.run([sys.executable,*command],cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,check=True)
        write(RUN/'pipeline_status.json',dict(stage='complete',status='COMPLETE',report=str(RUN/'analysis/REPORT.md'),time=time.time()))
    except Exception as exc:
        import traceback
        write(RUN/'pipeline_failure.json',dict(error=str(exc),traceback=traceback.format_exc(),time=time.time()))
        write(RUN/'pipeline_status.json',dict(stage='stopped',status='FAILED',error=str(exc),time=time.time()))
        # Import only after the resource-heavy child exited; maintain both asset tables.
        sys.path.insert(0,str(ROOT/'tools'))
        from run_receiver_direction_quick_20261115 import record,CFG
        record(RUN/'candidate_retry1',CFG,42,'candidate','PIPELINE-FAILED / NO-UTILITY-CONCLUSION',str(exc),str(RUN/'candidate_retry1/checkpoint_u0468.pth.tar'))
        raise

if __name__=='__main__':main()
