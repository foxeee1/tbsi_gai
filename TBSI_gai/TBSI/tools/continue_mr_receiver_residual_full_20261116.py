#!/usr/bin/env python3
"""Serially continue candidate -> control -> full244 eval -> final report."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; RUN = ROOT / 'output/experiments/MR-RECEIVER-RESIDUAL-FULL15-S42-20261116'


def write(path, data): path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def wait_pid(pid):
    while True:
        try: os.kill(pid, 0)
        except ProcessLookupError: return
        try:
            state = Path('/proc/%d/stat' % pid).read_text().split()[2]
            if state == 'Z': return
        except FileNotFoundError:
            return
        time.sleep(15)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--candidate-pid', type=int, required=True); a = p.parse_args()
    status = RUN / 'pipeline_status.json'; write(status, {'stage': 'candidate_training', 'pid': a.candidate_pid, 'time': time.time()})
    try:
        wait_pid(a.candidate_pid)
        cand = RUN / 'candidate_retry1'
        if not (cand / 'completed.json').exists(): raise RuntimeError('candidate ended without completed.json')
        write(status, {'stage': 'control_training', 'time': time.time()})
        control = RUN / 'control'; log = RUN / 'control_train.log'
        with log.open('w') as f:
            subprocess.run([sys.executable, 'tools/run_mr_receiver_residual_full_20261116.py', 'control', '--output', str(control)], cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, check=True)
        if not (control / 'completed.json').exists(): raise RuntimeError('control ended without completed.json')
        write(status, {'stage': 'candidate_full244_evaluation', 'time': time.time()})
        with (RUN / 'candidate_eval.log').open('w') as f:
            subprocess.run([sys.executable, 'tools/evaluate_mr_receiver_residual_full_20261116.py', 'candidate', '--output', str(RUN / 'eval_candidate'), '--checkpoint', str(cand / 'checkpoint_u7020.pth.tar')], cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, check=True)
        write(status, {'stage': 'control_full244_evaluation', 'time': time.time()})
        with (RUN / 'control_eval.log').open('w') as f:
            subprocess.run([sys.executable, 'tools/evaluate_mr_receiver_residual_full_20261116.py', 'control', '--output', str(RUN / 'eval_control'), '--checkpoint', str(control / 'checkpoint_u7020.pth.tar')], cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, check=True)
        write(status, {'stage': 'finalizing', 'time': time.time()})
        with (RUN / 'finalize.log').open('w') as f:
            subprocess.run([sys.executable, 'tools/finalize_mr_receiver_residual_full_20261116.py'], cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, check=True)
        write(status, {'stage': 'complete', 'status': 'COMPLETE', 'report': str(RUN / 'analysis/REPORT.md'), 'time': time.time()})
    except Exception as exc:
        write(RUN / 'pipeline_failure.json', {'error': str(exc), 'time': time.time()}); write(status, {'stage': 'failed', 'status': 'FAILED', 'error': str(exc), 'time': time.time()}); raise


if __name__ == '__main__': main()
