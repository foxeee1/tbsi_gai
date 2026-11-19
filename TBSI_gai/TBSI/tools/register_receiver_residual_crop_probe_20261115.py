#!/usr/bin/env python3
"""Incrementally register the crop probe and its pre-forward setup failure."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'output/experiments/MSHA-MR-RECEIVER-MESSAGE-RESIDUAL-QUICK-S42-20261115/analysis/crop_perturbation_20261115'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    s=json.loads((DEST/'summary.json').read_text());contract=json.loads((DEST/'CONTRACT.json').read_text())
    registry=ROOT/'experiments/experiment_asset_registry.csv';md=ROOT/'experiments/实验资产数据库.md'
    with registry.open(newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;old=list(reader)
    ids={r['ID'] for r in old};text=md.read_text()
    parent='MR-RECEIVER-MESSAGE-RESIDUAL-QUICK-S42-20261115-CANDIDATE; matched dynamic-gate control'
    rows=[]
    identity='MR-RECEIVER-CROP-PERTURBATION-S42-20261115'
    assert identity not in ids and '<!-- '+identity+' BEGIN -->' not in text
    r={f:'NA' for f in fields}
    r.update(ID=identity,Date='20261115',Parent=parent,
        CommitOrSnapshot=str((DEST/'source/code_snapshot.tar.gz').relative_to(ROOT))+' SHA256 '+contract['source_snapshot_sha256'],
        Config='Explicit MR-RECEIVER-MESSAGE-RESIDUAL-QUICK-S42-20261115 control/candidate YAMLs; full paths and SHA256 in CONTRACT.json',
        Seed='42',Dataset='LasHeR testingset; same96 screen cohort; inherited96 prefix and192 retrospective events;286 unique valid-GT frames',
        EvalScope='Five exact original-pixel crops (center/left/right/up/down by1px); control/candidate native/candidate off;4290 forwards+24 repeats; primary91 target-covered events; zero optimizer steps',
        GPU='1x NVIDIA GeForce RTX 4090 D 24 GiB',HardwareConfig='Python3.8.20/PyTorch2.0.0+cu118/CUDA11.8; FP32 tensors, matmul high; torch threads1/cv2 threads0; single exclusive process',
        GPUHours=str(round(s['execution']['estimated_gpu_hours_single_exclusive_task'],6)),
        Checkpoint='u468 control SHA256 '+contract['checkpoints_sha256']['control']+'; candidate SHA256 '+contract['checkpoints_sha256']['candidate'],
        AssumptionOrHypothesis='Discriminate whole-model crop sensitivity from crop-dependent harmful sign reversals of the learned residual at fixed template/history/frame; all arms use identical image inputs.',
        Conclusion='Primary91: mean IoU crop range control7.0139pp/candidate7.5042pp/off7.5049pp; candidate-control utility flips beyond+/-0.5pp4/91; native-off peak changes0/455,max IoU difference0.039862pp,meaningful flips0. Residual cosine median0.99995984; no negative cosine. Supports shared crop sensitivity; does not support residual-specific large action flips. Supplements reverse model-sensitivity difference direction; no universal claim.',
        Status='DIAGNOSTIC-COMPLETE / CROP-SENSITIVITY-OBSERVED / RESIDUAL-FLIP-HYPOTHESIS-NOT-SUPPORTED',
        Evidence=str((DEST/'REPORT.md').relative_to(ROOT)),
        Notes='No training/model edits/full promotion; not AUC; current dynamic-gate quick is not historical MR g=1. All crop/template hashes match; repeat error0; states unchanged; control replay max6.11e-5px. Sequence tags overlap and are not per-frame degradation labels; continuous feedback causality from prior replay remains valid.')
    rows.append(r)
    identity2=identity+'-SETUP-REPAIR'
    assert identity2 not in ids and '<!-- '+identity2+' BEGIN -->' not in text
    failed=DEST.with_name(DEST.name+'_setup_failed')
    r2={f:'NA' for f in fields}
    r2.update(ID=identity2,Date='20261115',Parent=identity,CommitOrSnapshot='Pre-forward partial archive '+str((failed/'source/code_snapshot.tar.gz').relative_to(ROOT)),
        Config='Same locked event manifest; checkpoint not loaded',Seed='42',Dataset='Locked diagnostic manifest only; no model forward',EvalScope='Source snapshot construction; zero optimizer steps and zero forwards',
        GPU='CPU-only; no model loaded',HardwareConfig='Python3.8.20',GPUHours='0',
        AssumptionOrHypothesis='Archive exact diagnostic source before model execution.',
        Conclusion='Relative __file__ could not be made relative_to absolute ROOT; failed before checkpoint loading/forward. Resolve script path before archiving; retry completed. Tool setup error, not model result.',
        Status='FAILED-SETUP / REPAIRED / NO-MODEL-FORWARD',Evidence=str((failed/'failure.json').relative_to(ROOT)),Notes='Original failure log, partial archive and manifest preserved; successful diagnostic uses independent fresh output directory.')
    rows.append(r2)
    backup=DEST/'source/registry_before';backup.mkdir(exist_ok=False)
    shutil.copy2(registry,backup/registry.name);shutil.copy2(md,backup/md.name)
    shutil.copy2(__file__,DEST/'source'/Path(__file__).name)
    # Preserve existing CSV bytes rather than rewriting unrelated historical entries.
    with registry.open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writerows(rows)
    with md.open('a') as f:
        for row in rows:
            f.write('\n\n<!-- '+row['ID']+' BEGIN -->\n### '+row['ID']+'\n\n')
            for key in fields:f.write('- **'+key+'**：'+row[key]+'\n')
            f.write('<!-- '+row['ID']+' END -->\n')
    with registry.open(newline='') as f:new=list(csv.DictReader(f))
    assert new[:-2]==old and new[-2:]==rows
    updated=md.read_text();assert updated.startswith(text)
    for row in rows:assert updated.count('<!-- '+row['ID']+' BEGIN -->')==1
    write=dict(ids=[r['ID'] for r in rows],csv_and_markdown_synchronized=True,prior_records_unchanged=True,
        commands=['python tools/probe_receiver_residual_crop_perturbation_20261115.py','python tools/analyze_receiver_residual_crop_perturbation_20261115.py','python tools/register_receiver_residual_crop_probe_20261115.py'],
        source_files={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),ROOT/'tools/probe_receiver_residual_crop_perturbation_20261115.py',ROOT/'tools/analyze_receiver_residual_crop_perturbation_20261115.py']},
        output_hashes={p.name:sha(p) for p in DEST.iterdir() if p.is_file()})
    (DEST/'registration_record.json').write_text(json.dumps(write,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(write,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
