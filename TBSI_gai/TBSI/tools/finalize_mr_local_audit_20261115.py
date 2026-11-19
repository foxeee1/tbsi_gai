#!/usr/bin/env python3
"""Archive audit source and synchronize final asset records without model edits."""
import csv
import hashlib
import json
import sys
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from record_mr_cycle_difference_20261115 import upsert
RUN=ROOT/'output/experiments/MSHA-MR-LOCAL-TRANSMISSION-AUDIT-S42-20261115'


def main():
    files=list((ROOT/'lib').rglob('*.py'))
    files += [ROOT/'tools'/name for name in ('audit_mr_local_transmission_20261115.py',
        'audit_mr_source_local_20261115.py','reference_full_mr_local_audit_20261115.py',
        'analyze_mr_local_transmission_20261115.py','finalize_mr_local_audit_20261115.py',
        'run_dual_message_p1.py','record_mr_cycle_difference_20261115.py')]
    files += [ROOT/'experiments/tbsi_track/MR-CBMR-QUICK-S42-20261115-candidate.yaml',
        ROOT/'experiments/tbsi_track/MR-CBMR-QUICK-S42-20261115-control.yaml',
        RUN/'PROTOCOL.md',RUN/'frame_manifest.json',RUN/'REPORT.md']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    before=json.loads((RUN/'contracts/before_sha256.json').read_text())
    assert all(hashes[k]==v for k,v in before.items()),'Host/protected config modified during audit'
    with tarfile.open(RUN/'contracts/source_final.tar.gz','w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(ROOT)))
    (RUN/'contracts/source_final_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    registry=list(csv.DictReader((ROOT/'experiments/experiment_asset_registry.csv').open()))
    updates={
        'MR-LOCAL-TRANSMISSION-AUDIT-S42-20261115':(
            'DIAGNOSTIC-COMPLETE / PROVENANCE-CORRECTED / NO-TRAIN',
            'Quick dynamic gate is not historical MR-g1. High-FP32 small-effect FD signs54%; highest FP32 repeat96/98/100% at eps.05/.1/.5. Both5184 intervention sets retained; no efficacy claim.'),
        'MR-LOCAL-TRANSMISSION-FULL-G1-S42-20261115':(
            'DIAGNOSTIC-COMPLETE / LOCAL-UTILITY-UNRESOLVED / NO-TRAIN',
            '72/72 original-workspace bitwise parity;5184 regional message interventions,0 Hann peak changes;144/144 FD signs match. Source spatial variance exists but target semantics/consistent utility unproven. Source-content follow-up3888 also0 peak changes; no full promotion.'),
        'MR-SOURCE-CONTENT-LOCAL-FULL-G1-S42-20261115':(
            'DIAGNOSTIC-COMPLETE / MIXED-EFFECTS / NO-TRAIN',
            '3888 fixed-relation source perturbations;relation@V exact native message;state unchanged;0 Hann peak changes;local detail/region contrast losses mixed,not sufficient evidence for semantically useful local content or new module.')}
    for identity,(status,conclusion) in updates.items():
        row=next(x for x in registry if x['ID']==identity)
        row.update(Status=status,Conclusion=conclusion,CommitOrSnapshot=str(RUN/'contracts/source_final.tar.gz'),Evidence=str(RUN/'REPORT.md'))
        upsert(row)
    complete=json.loads((RUN/'global_controls_full/completed.json').read_text())
    previous=json.loads((RUN/'global_controls_full_v1/completed.json').read_text())
    row=dict(ID='MR-EFFECTIVE-ALLOCATOR-FULL-G1-S42-20261115',Date='20261115',
        Parent='MSHA-MRONLY-G1-U936-S42-FULL15-FULL244-20261109; MR-LOCAL-TRANSMISSION-FULL-G1-S42-20261115',
        CommitOrSnapshot=str(RUN/'contracts/source_final.tar.gz'),Config='Archived mr_only_g1.yaml SHA256 ebfeddd3840ca604af45353bb66bf81028233c8ffdf3e44cff7b5956f9601890; explicit verified adapter',
        Seed=42,Dataset='LasHeR fixed48 test frames/24 exploratory sequences plus24 train sequence frames;train length strata18low6middle0high',
        EvalScope='72 matched crops;MR-off and cross-off144 interventions twice;432 allocator captures;0 optimizer steps',
        GPU='1x NVIDIA GeForce RTX 4090 D24GiB',HardwareConfig='Python3.8.20/PyTorch2.0.0+cu118/CUDA11.8;highest FP32;matmul/cudnn TF32 disabled;eval;threads1;workers0',
        GPUHours=complete['gpu_wall_hours']+previous['gpu_wall_hours'],
        Checkpoint='u7020 SHA256 2da4a00a1cb8af18a6abba019af1eb0699c3d01c184f7b23b97dbb9a5d4762f5',
        AssumptionOrHypothesis='Determine effective source-score variation after softmax row invariance,conserved-mass content action,and immediate MR-vs-entire-cross sensitivity.',
        Conclusion='13.6048% score energy source-centered;conditional relation L1.000797;native MR message edit.0418%.MR-off final search~.0012-.0014%,0 peaks;cross-off~4%,1 peak outsideGT crop. Mass and zero-sum algebra errors<4.5e-7. Historical gains not causally explained by strong instantaneous ranking;training-path effect remains unresolved.',
        Status='DIAGNOSTIC-COMPLETE / EFFECTIVE-ACTION-NEAR-IDENTITY / NO-TRAIN',Evidence=str(RUN/'REPORT.md'),
        Notes='Cross-off accumulated across3 layers;MR-off same checkpoint is not retrainedB0;correspondence with priorseed43 is not multiseed method efficacy. Generic utility loss prior negative preserved.')
    upsert(row)
    csvrows=list(csv.DictReader((ROOT/'experiments/experiment_asset_registry.csv').open()))
    md=(ROOT/'experiments/实验资产数据库.md').read_text()
    ids=list(updates)+[row['ID'],'MR-LOCAL-TRANSMISSION-FULL-TOOL-REPAIR-20261115']
    for identity in ids:
        matches=[r for r in csvrows if r['ID']==identity]
        assert len(matches)==1
        assert '<!-- '+identity+' BEGIN -->' in md
        assert matches[0]['Status'] in md
    ref=json.loads((RUN/'full_reference.json').read_text())
    timed=[json.loads(p.read_text())['gpu_wall_hours'] for p in (RUN/'completed.json',RUN/'quick_highest_precision/completed.json',
        RUN/'full_mr_g1/completed.json',RUN/'source_content_full/completed.json',RUN/'global_controls_full_v1/completed.json',RUN/'global_controls_full/completed.json')]
    (RUN/'FINAL_STATE.json').write_text(json.dumps(dict(status='AUDIT-COMPLETE',optimizer_steps=0,new_checkpoints=0,
        counted_intervention_forwards=19728,successfully_timed_gpu_wall_hours=sum(timed)+ref['gpu_wall_hours'],
        total_gpu_hours_including_failed_attempt='NA',registry_ids=ids,registry_synchronized=True,
        pre_audit_host_and_config_hashes_unchanged=True,report=str(RUN/'REPORT.md'),
        source_archive_sha256=hashlib.sha256((RUN/'contracts/source_final.tar.gz').read_bytes()).hexdigest()),indent=2)+'\n')
    print((RUN/'FINAL_STATE.json').read_text())


if __name__=='__main__':main()
