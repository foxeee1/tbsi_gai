#!/usr/bin/env python3
"""Synchronized incremental asset entries; preserves existing rows."""
import csv
import hashlib
import json
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'output/experiments/MSHA-MR-RECEIVER-MESSAGE-RESIDUAL-QUICK-S42-20261115/analysis/task_direction_20261115'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
    return h.hexdigest()

def main():
    s=json.loads((DEST/'summary.json').read_text());c=json.loads((DEST/'CONTRACT.json').read_text())
    registry=ROOT/'experiments/experiment_asset_registry.csv';md=ROOT/'experiments/实验资产数据库.md'
    with registry.open(newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;old=list(reader)
    previous=md.read_text();identity='MR-RECEIVER-TASK-DIRECTION-S42-20261115';rows=[]
    row={k:'NA' for k in fields}
    row.update(ID=identity,Date='20261115',Parent='MR-RECEIVER-MESSAGE-RESIDUAL-QUICK-S42-20261115-CANDIDATE; MR-RECEIVER-CROP-PERTURBATION-S42-20261115',
        CommitOrSnapshot=str((DEST/'source/code_snapshot.tar.gz').relative_to(ROOT))+' SHA256 '+c['source_snapshot_sha256'],
        Config=c['config']+' SHA256 '+c['config_sha256'],Seed='42',Dataset='LasHeR testingset; inherited96 primary fixed events,91 previously target-covered loss/gradient eligible; dataset path in CONTRACT.json',
        EvalScope='Same u468 checkpoint, fixed control-previous-state crops;1713 inference+91 gradient forwards;0 optimizer steps; frozen native delta scales +/-1/4/16/64, token-roll and norm-matched offline GT oracle; primary highest FP32 and separate high precision comparison',
        GPU='1x NVIDIA GeForce RTX 4090 D 24 GiB',HardwareConfig='Python3.8.20/PyTorch2.0.0+cu118/CUDA11.8; network FP32,loss reductions float64; primary TF32 disabled; torch threads1/cv2 threads0; one exclusive process',
        GPUHours=str(round(s['execution']['gpu_hours_exclusive'],6)),Checkpoint=c['checkpoint']+' SHA256 '+c['checkpoint_sha256'],
        AssumptionOrHypothesis='Test whether learned residual direction consistently decreases GT tracking loss and merely needs amplification; separate target-direction quality from interface reachability and numerical precision.',
        Conclusion='91 events: total-loss directional derivatives favorable42/unfavorable46/near-zero3; mean descent alignment -0.000285. +/-16 finite differences agree signs88/88 above floor,corr0.999965. Frozen learned64 yields0 Hann peak changes,mean IoU+0.000176pp,loss decrease/increase16/17 at1e-4. Equal-per-module-norm GT oracle64 lowers loss91/91 at1e-3,mean IoU+0.230799pp,median+0.115374pp,practical6wins/0losses. Does not support amplitude-only fix; interface can convey useful edits; supervision vs input/expressivity unresolved.',
        Status='DIAGNOSTIC-COMPLETE / AMPLITUDE-ONLY-NOT-SUPPORTED / DIRECTION-LEARNING-UNRESOLVED',
        Evidence=str((DEST/'REPORT.md').relative_to(ROOT)),
        Notes='No model edits/training/full promotion. GT oracle is offline unconstrained output-space direction,not deployable or proven realizable by rank16 residual. Current dynamic-gate quick is not historical MR g=1. Frozen1/native parity0; gradient0/off parity0; high history replay max6.11e-5px. State unchanged,parameter grads absent. Single seed,selected fixed events; not sequence AUC. Prior closed-loop feedback evidence not invalidated.')
    rows.append(row)
    row2={k:'NA' for k in fields};failed=DEST.with_name(DEST.name+'_precision_setup_interrupted')
    row2.update(ID=identity+'-PRECISION-SETUP-INTERRUPTED',Date='20261115',Parent=identity,
        CommitOrSnapshot=str((failed/'source/code_snapshot.tar.gz').relative_to(ROOT))+' SHA256 '+sha(failed/'source/code_snapshot.tar.gz'),
        Config=c['config'],Seed='42',Dataset='Same96 locked event manifest; interrupted during first event',EvalScope='Agent stopped initial diagnostic before comparison to fix precision-factor restoration; exact partial forward count not persisted;0 optimizer steps',
        GPU='1x NVIDIA GeForce RTX 4090 D 24 GiB',HardwareConfig='Python3.8.20/PyTorch2.0.0+cu118; one process',Checkpoint=c['checkpoint'],
        AssumptionOrHypothesis='Keep historical high TF32 settings independent from highest-precision factor for every event.',
        Conclusion='Static review found cudnn TF32 flag would remain disabled on later high passes. Interrupted first event; restored historical cudnn flag explicitly every high pass and logged both TF32 flags per row. No model-performance result from partial run; complete retry passes historical parity.',
        Status='INTERRUPTED-TOOL-SETUP / PRECISION-FACTOR-REPAIRED',Evidence=str((failed/'interruption.json').relative_to(ROOT)),Notes='Original manifest/snapshot/log preserved. Partial duration and exact completed-forward count were not persisted and are NA; do not label model failure.')
    rows.append(row2)
    for r in rows:
        assert all(oldr['ID']!=r['ID'] for oldr in old)
        assert '<!-- '+r['ID']+' BEGIN -->' not in previous
    backup=DEST/'source/registry_before';backup.mkdir(exist_ok=False)
    for p in [registry,md]:shutil.copy2(p,backup/p.name)
    shutil.copy2(Path(__file__).resolve(),DEST/'source'/Path(__file__).name)
    with registry.open('a',newline='') as f:csv.DictWriter(f,fieldnames=fields).writerows(rows)
    with md.open('a') as f:
        for r in rows:
            f.write('\n\n<!-- '+r['ID']+' BEGIN -->\n### '+r['ID']+'\n\n')
            for k in fields:f.write('- **'+k+'**：'+r[k]+'\n')
            f.write('<!-- '+r['ID']+' END -->\n')
    with registry.open(newline='') as f:new=list(csv.DictReader(f))
    assert new[:-2]==old and new[-2:]==rows and md.read_text().startswith(previous)
    record=dict(ids=[r['ID'] for r in rows],csv_and_markdown_synchronized=True,prior_entries_unchanged=True,
        commands=['python tools/probe_receiver_residual_task_direction_20261115.py','python tools/analyze_receiver_residual_task_direction_20261115.py','python tools/register_receiver_residual_task_direction_20261115.py'],
        source_hashes={p.name:sha(p) for p in (DEST/'source').iterdir() if p.is_file()},
        output_hashes={p.name:sha(p) for p in DEST.iterdir() if p.is_file()})
    (DEST/'registration_record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
