#!/usr/bin/env python3
"""Incremental synchronized records for the single normalization refinement."""
import argparse
import csv
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'output/experiments/MSHA-MR-CYCLE-DIFFERENCE-QUICK-S42-20261115'


def upsert(row):
    csvpath=ROOT/'experiments/experiment_asset_registry.csv'
    mdpath=ROOT/'experiments/实验资产数据库.md'
    lines=csvpath.read_text().splitlines(keepends=True)
    fields=next(csv.reader([lines[0]]))
    value={k:row.get(k,'NA') for k in fields}
    buf=io.StringIO()
    csv.DictWriter(buf,fieldnames=fields,lineterminator='\n').writerow(value)
    found=False
    for index,line in enumerate(lines):
        if line.startswith(value['ID']+','):
            lines[index]=buf.getvalue(); found=True; break
    if not found:
        lines.append(buf.getvalue())
    csvpath.write_text(''.join(lines))
    begin='<!-- '+value['ID']+' BEGIN -->'
    end='<!-- '+value['ID']+' END -->'
    block=begin+'\n### '+value['ID']+'\n\n'+''.join('- **'+k+'**：'+str(value[k])+'\n' for k in fields)+end
    text=mdpath.read_text()
    if begin in text:
        a=text.index(begin);b=text.index(end,a)+len(end)
        text=text[:a]+block+text[b:]
    else:
        text+='\n\n'+block+'\n'
    mdpath.write_text(text)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['implementation','running','final'])
    args=p.parse_args()
    row=dict(ID='MR-CYCLE-DIFFERENCE-QUICK-S42-20261115-CANDIDATE',Date='20261115',
        Parent='MR-CBMR-QUICK-S42-20261115-CONTROL; normalization ablation vs CBMR candidate',
        CommitOrSnapshot=str(RUN/'contracts/source_at_launch.tar.gz'),
        Config='experiments/tbsi_track/MR-CYCLE-DIFFERENCE-QUICK-S42-20261115-candidate.yaml',
        Seed='42',Dataset='LasHeR train frozen977; 59904 pairs / 468 successful updates; original24 exploratory test / 20796 frames',
        EvalScope='FP32; tracker threads0; no template update; same manifest and source initialization as CBMR control',
        GPU='1x NVIDIA GeForce RTX 4090 D 24 GiB',
        HardwareConfig='Python3.8.20/PyTorch2.0.0+cu118/CUDA11.8; AMP B16 accum8; workers4 prefetch2 persistentFalse; fresh optimizer',
        AssumptionOrHypothesis='Only replace symmetric degree normalization with row-normalized pairwise message differences to preserve constant messages; test learning utility without changing graph/mask/strength/position.',
        Evidence=str(RUN/'PROTOCOL.md'),
        Notes='Explicitly supersedes unconditional useful-recovery-damage attribution. Old graph is dense and message common component dominates. Final checkpoint weights-only due disk; optimizer behavior unchanged; no continuation or full run.')
    if args.phase=='implementation':
        row.update(ID='MR-CYCLE-DIFFERENCE-IMPLEMENTATION-S42-20261115',
            Status='IMPLEMENTATION-CHECKS-PASS / NO-EFFICACY-CLAIM',
            Dataset='Synthetic tensor contracts and full model from public shared initialization; no optimizer steps',
            EvalScope='Before/after disabled and original symmetric bitwise parity; constant/zero/mask/isolation/gradient/full-model-save-load checks',
            GPUHours='NA',Conclusion='All targeted contracts pass; constant max error 2.384e-7;3 actual difference modules. Implementation check does not establish task utility.',
            Evidence=str(RUN/'contracts/implementation_checks.json'))
    elif args.phase=='running':
        row.update(Status='RUNNING / BOUNDED-QUICK',Conclusion='468-update single-stage candidate planned; reused control admissible only after parity, common state and all input-hash checks. No efficacy result yet.')
    else:
        s=json.loads((RUN/'analysis/summary.json').read_text())
        comp=json.loads((RUN/'candidate/completed.json').read_text())
        pairing=json.loads((RUN/'contracts/training_pairing.json').read_text())
        m=s['candidate'];delta=s['candidate_minus_control']['auc'];seq=s['auc_delta_stats']
        row.update(AUC=m['auc'],PR=m['precision'],NPR=m['npr'],OP50=m['op50'],OP75=m['op75'],
            FPS=s['throughput_fps']['candidate'],GPUHours=comp['gpu_wall_hours'],
            Checkpoint=str(RUN/'candidate/checkpoint_u0468.pth.tar')+' SHA256 '+s['checkpoint_sha256'],
            Conclusion='Paired control AUC %.4f; difference %.4f; delta %+.4fpp; %d/%d/%d wins/ties/losses; median %+.4fpp. All3744 input/box/metadata hashes and common init match. Exploratory only; no full promotion.' %
                (s['control']['auc'],m['auc'],delta,seq['wins'],seq['ties'],seq['losses'],seq['median_pp']),
            Status='QUICK-TRAIN-EVAL-COMPLETE / EXPLORATORY / NO-FULL-PROMOTION',Evidence=str(RUN/'analysis/REPORT.md'))
        if not pairing['all_checks_pass']:
            raise RuntimeError('Paired contract failed; suspend ranking')
    upsert(row)
    print(row['ID'],row['Status'])


if __name__=='__main__':
    main()
