#!/usr/bin/env python3
"""Frozen-crop CBMR diagnosis; checkpoints are never used for new training."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ['TBSI_INFER_FP16'] = '0'
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from run_dual_message_p1 import config, setup, environment, sha, write_json
from lib.test.evaluation import get_dataset
from lib.test.tracker.tbsi_track import TBSITrack
from lib.models.layers.cycle_laplacian_message_repair import _box_coverage

PARENT = ROOT / 'output/experiments/MSHA-MR-CBMR-QUICK-S42-20261115'
OUT = ROOT / 'output/experiments/MSHA-MR-CBMR-AUDIT-S42-20261115'


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    inter = np.maximum(0, np.minimum(a[:2] + a[2:], b[:2] + b[2:]) -
                       np.maximum(a[:2], b[:2])).prod()
    return float(inter / max(float(a[2:].prod() + b[2:].prod() - inter), 1e-12))


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


class Intervention:
    def __init__(self):
        self.mode = 'native'
        self.rows = []
        self.context = {}
        self.relations = {}

    def install(self, net):
        self.layers = [(n, m) for n, m in net.named_modules()
                       if getattr(m, 'cbmr_enabled', False)]
        self.handles = []
        for name, host in self.layers:
            for direction in ('rgb', 'tir'):
                def save(mod, args, output, key=(name, direction)):
                    self.relations[key] = output.detach()
                self.handles.append(getattr(host, 'mass_reallocator_' + direction).
                                    register_forward_hook(save))
            original = host.cbmr._repair_one
            def repair(module, message, forward, backward, background,
                       layer=name, original=original):
                return self.repair(module, message, forward, backward,
                                   background, layer, original)
            host.cbmr._repair_one = MethodType(repair, host.cbmr)

    def repair(self, module, message, forward, backward, bg, layer, original):
        f = forward.float() / forward.float().sum(-1, keepdim=True).clamp_min(1e-6)
        b = backward.float() / backward.float().sum(-1, keepdim=True).clamp_min(1e-6)
        cycle = f @ b
        w = (cycle + cycle.transpose(-1, -2)) * .5
        w *= bg[:, None, :, None] * bg[:, None, None, :]
        degree = w.sum(-1)
        support = degree > 1e-6
        inv = degree.clamp_min(1e-6).rsqrt()
        sym = w * inv[..., None] * inv[..., None, :]
        walk = w / degree.clamp_min(1e-6)[..., None]
        if self.mode == 'off':
            return message, degree, support
        if self.mode == 'walk':
            out = message.float() + module.strength * (walk @ message.float() - message.float())
            out = torch.where(support[..., None], out, message.float()).to(message.dtype)
            return out, degree, support
        out, _, _ = original(message, forward, backward, bg)
        # Gather graph diagnostics only for the historical operator; no writes
        # to the model, no GT-derived action, and no change in its calculation.
        direction = 'rgb' if self.call_index[layer] % 2 == 0 else 'tir'
        self.call_index[layer] += 1
        real = self.relations[(layer, direction)].float()
        real = real / real.sum(-1, keepdim=True).clamp_min(1e-6)
        foreground_middle = self.foreground[(layer, direction)]
        bridge = (f * foreground_middle[:, None, None, :]) @ b
        selected = support
        mean = lambda x: float(x[selected].mean()) if selected.any() else 0.
        n_bg = bg.sum(-1).clamp_min(1)[:, None, None]
        entropy = -(walk.clamp_min(1e-30).log() * walk).sum(-1)
        msg = message.float()
        bg_weight = bg[:, None, :, None].float()
        avg = (msg * bg_weight).sum(-2, keepdim=True) / bg_weight.sum(-2, keepdim=True).clamp_min(1)
        centered = (msg - avg) * bg_weight
        raw_energy = (msg.square() * bg_weight).sum()
        constant = (avg.expand_as(msg).square() * bg_weight).sum()
        rowsum = sym.sum(-1)
        self.rows.append(dict(self.context, layer=layer, direction=direction,
            bg_tokens=float(bg.sum(-1).float().mean()),
            support_fraction=float(selected.sum() / bg.sum().clamp_min(1) / forward.shape[1]),
            graph_entropy_fraction=mean(entropy / n_bg.log().clamp_min(1e-6)),
            graph_effective_neighbors=mean(entropy.exp()),
            graph_row_max=mean(walk.max(-1).values),
            symmetric_constant_drift=mean((rowsum - 1).abs()),
            symmetric_rowsum_min=float(rowsum[selected].min()) if selected.any() else 0.,
            symmetric_rowsum_max=float(rowsum[selected].max()) if selected.any() else 0.,
            fg_bridge_fraction=float((bridge * bg_weight).sum() / (cycle * bg_weight).sum().clamp_min(1e-9)),
            conditional_mr_vs_graph_l1=mean((real - f).abs().sum(-1)),
            constant_energy_fraction=float(constant / raw_energy.clamp_min(1e-9)),
            native_relative_edit=float((out.float()-msg).norm() / msg.norm().clamp_min(1e-9)),
            centered_message_norm=float(centered.norm())))
        return out, degree, support

    def prepare(self, boxes):
        self.call_index = {name: 0 for name, _ in self.layers}
        self.foreground = {}
        for name, host in self.layers:
            for direction, idx in (('rgb', 1), ('tir', 0)):
                self.foreground[(name, direction)] = (_box_coverage(boxes[:, idx], 8) > 1e-6).float()


def pairing():
    a = [json.loads(x) for x in (PARENT/'control/batches.jsonl').read_text().splitlines()]
    b = [json.loads(x) for x in (PARENT/'candidate/batches.jsonl').read_text().splitlines()]
    return dict(control_batches=len(a), candidate_batches=len(b),
        inputs_equal=len(a)==len(b) and all(x['input_sha256']==y['input_sha256'] for x,y in zip(a,b)),
        boxes_equal=len(a)==len(b) and all(x['template_box_sha256']==y['template_box_sha256'] for x,y in zip(a,b)),
        paired_metadata_equal=len(a)==len(b) and all(x['samples']==y['samples'] for x,y in zip(a,b)))


def main():
    setup(42)
    started=time.time()
    if (OUT/'summary.json').exists():
        raise RuntimeError('Refusing overwrite of completed diagnostic')
    cfg_path=ROOT/'experiments/tbsi_track/MR-CBMR-QUICK-S42-20261115-candidate.yaml'
    cfg=config(cfg_path)
    params=SimpleNamespace(cfg=cfg, checkpoint=str(PARENT/'candidate/checkpoint_u0468.pth.tar'),
        template_factor=cfg.TEST.TEMPLATE_FACTOR, template_size=cfg.TEST.TEMPLATE_SIZE,
        search_factor=cfg.TEST.SEARCH_FACTOR, search_size=cfg.TEST.SEARCH_SIZE,
        debug=0, save_all_boxes=False)
    tracker=TBSITrack(params,'lasher_test')
    hook=Intervention()
    hook.install(tracker.network)
    names=(PARENT/'contracts/eval_flip_gain_regression_24.txt').read_text().splitlines()
    dataset={s.name:s for s in get_dataset('lasher_test')}
    manifest=[]
    for name in names:
        seq=dataset[name]
        pred=np.loadtxt(PARENT/'eval_control'/(name+'.txt'),ndmin=2)
        for frame in np.unique(np.linspace(1,len(seq.ground_truth_rect)-1,6,dtype=int)):
            manifest.append(dict(sequence=name,frame=int(frame),crop_state=pred[frame-1].tolist()))
    write_json(OUT/'frame_manifest.json',manifest)
    outputs=[]
    def image(seq, frame):
        images=[cv2.cvtColor(cv2.imread(seq.frames[k][frame]),cv2.COLOR_BGR2RGB) for k in (0,1)]
        return np.concatenate(images,axis=2)
    for arm in ('control','candidate'):
        ckpt=PARENT/arm/'checkpoint_u0468.pth.tar'
        state=torch.load(str(ckpt),map_location='cpu')['net']
        tracker.network.load_state_dict(state,strict=True)
        del state
        for index,name in enumerate(names):
            seq=dataset[name]
            tracker.initialize(image(seq,0),seq.init_info())
            boxes=torch.stack((tracker.template_boxes,tracker.template_boxes),dim=1)
            hook.prepare(boxes)
            entries=[x for x in manifest if x['sequence']==name]
            for entry in entries:
                img=image(seq,entry['frame'])
                digest=hashlib.sha256(img.tobytes()).hexdigest()
                for mode in ('off','native','walk'):
                    hook.mode=mode
                    hook.call_index={n:0 for n,_ in hook.layers}
                    hook.context=dict(checkpoint_arm=arm,sequence=name,frame=entry['frame'])
                    tracker.state=list(entry['crop_state'])
                    tracker.frame_id=entry['frame']-1
                    with torch.inference_mode():
                        result=tracker.track(img)
                    box=result['target_bbox']
                    outputs.append(dict(checkpoint_arm=arm,sequence=name,frame=entry['frame'],mode=mode,
                        image_sha256=digest,crop_state=json.dumps(entry['crop_state']),
                        iou=iou(box,seq.ground_truth_rect[entry['frame']]),box=json.dumps(box)))
            print(json.dumps(dict(arm=arm,finished=index+1,total=len(names),seconds=time.time()-started)),flush=True)
    write_csv(OUT/'frame_interventions.csv',outputs)
    write_csv(OUT/'graph_diagnostics.csv',hook.rows)
    summary=dict(environment=environment(),pairing=pairing(),frames=len(manifest),
        scope='same checkpoint interventions; fixed control-previous-state crops; not tracking AUC or retraining',
        checkpoints={arm:sha(PARENT/arm/'checkpoint_u0468.pth.tar') for arm in ('control','candidate')},
        config_sha256=sha(cfg_path),manifest_sha256=sha(OUT/'frame_manifest.json'),
        wall_seconds=time.time()-started,arms={})
    for arm in ('control','candidate'):
        rows=[r for r in outputs if r['checkpoint_arm']==arm]
        vals={mode:np.array([r['iou'] for r in rows if r['mode']==mode]) for mode in ('off','native','walk')}
        native=[r for r in hook.rows if r['checkpoint_arm']==arm]
        keys=[k for k,v in native[0].items() if isinstance(v,float)]
        summary['arms'][arm]=dict(mean_iou={k:float(v.mean()) for k,v in vals.items()},
            graph_means={k:float(np.mean([r[k] for r in native])) for k in keys},
            intervention_deltas={m:dict(mean=float((vals[m]-vals['off']).mean()),
                wins=int(((vals[m]-vals['off'])>1e-6).sum()),
                ties=int((abs(vals[m]-vals['off'])<=1e-6).sum()),
                losses=int(((vals[m]-vals['off'])<-1e-6).sum())) for m in ('native','walk')})
    write_json(OUT/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
