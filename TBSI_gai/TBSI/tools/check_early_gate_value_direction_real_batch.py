"""One fixed real-GT batch; no tracker loop or performance conclusion."""
import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from lib.config.tbsi_track.config import cfg, update_config_from_file
from lib.models.tbsi_track import build_tbsi_track
from lib.models.layers.early_conditioned_value_direction import install_value_direction
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, giou_loss
from lib.utils.heapmap_utils import generate_heatmap
from lib.utils.focal_loss import FocalLoss


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def run(args, report):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    report.update(python=platform.python_version(), torch=torch.__version__,
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(0), gpu_count_used=1,
        threads=1, dataloader_workers=0, precision='FP32; TF32 off',
        worker_reason='fixed diagnostic batch loaded directly, no DataLoader',
        config=str(args.config), config_sha256=sha(args.config),
        checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
        seed=42, parent_epoch=15, optimizer='fresh AdamW, lr=1e-4, wd=1e-4',
        train_mode='parent eval/frozen, new operator only; not joint short train')
    if report['checkpoint_sha256'] != 'e38d18fe50c2895e75e4b9392bf22d9323bb8e6ad31ccbe39ad8c038e3fc1811':
        raise RuntimeError('explicit Early gate parent checksum mismatch')
    update_config_from_file(str(args.config))
    model = build_tbsi_track(copy.deepcopy(cfg), training=False)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    missing, unexpected = model.load_state_dict(checkpoint['net'], strict=False)
    report['load_missing'], report['load_unexpected'] = missing, unexpected
    if missing or unexpected:
        raise RuntimeError('parent state keys do not match retained architecture')
    del checkpoint
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    root = args.dataset
    names = sorted(set(x.strip().split(',')[0] for x in
        (root.parent/'trainingsetList.txt').read_text().splitlines() if x.strip()))[:2]
    z, x, gt, samples = [[], []], [[], []], [], []
    for name in names:
        folder = root/name
        boxes = np.loadtxt(str(folder/'init.txt'), delimiter=',').reshape(-1,4)
        frame = min(10,len(boxes)-1)
        if frame <= 0 or (boxes[[0,frame],2:] <= 0).any():
            raise RuntimeError('preselected sample has invalid annotation')
        template_box = torch.tensor(boxes[0]).float()
        search_box = torch.tensor(boxes[frame]).float()
        files = []
        for mi, modality in enumerate(('visible','infrared')):
            images = sorted(p for p in (folder/modality).iterdir()
                if p.suffix.lower() in ('.jpg','.png','.jpeg'))
            if len(images) <= frame:
                raise RuntimeError('preselected sample has insufficient images')
            mean = torch.tensor(cfg.DATA.MEAN[mi*3:mi*3+3]).view(3,1,1)
            std = torch.tensor(cfg.DATA.STD[mi*3:mi*3+3]).view(3,1,1)
            for destination,index,size,factor,box in (
                (z,0,128,2.0,template_box),(x,frame,256,4.0,search_box)):
                image = cv2.cvtColor(cv2.imread(str(images[index])),cv2.COLOR_BGR2RGB)
                crop, resize, _ = sample_target(image, box, factor, size)
                tensor = torch.from_numpy(crop.copy()).permute(2,0,1).float()/255.
                destination[mi].append((tensor-mean)/std)
                files.append(dict(path=str(images[index]),sha256=sha(images[index])))
                if destination is x and mi == 0:
                    gt.append(transform_image_to_crop(search_box,search_box,resize,
                        torch.tensor([size,size]),normalize=True))
        samples.append(dict(sequence=name,template_frame_index=0,search_frame_index=frame,
            files=files,rule='sorted official training list first two; no outcome selection'))
    z = [torch.stack(t).to(device) for t in z]
    x = [torch.stack(t).to(device) for t in x]
    gt = torch.stack(gt).to(device)
    report['samples'] = samples
    report['input_sha256'] = [tensor_sha(t) for t in z+x+[gt]]
    report['crop_definition'] = 'GT-centered deterministic crops; no augmentation; diagnostic only'
    report['dataset'] = str(root.resolve())
    heatmap = generate_heatmap(gt.unsqueeze(0),256,16)[0].unsqueeze(1)
    focal = FocalLoss()
    def loss_of(out):
        boxes = out['pred_boxes'].reshape(-1,4)
        giou, iou = giou_loss(box_cxcywh_to_xyxy(boxes),box_xywh_to_xyxy(gt).clamp(0,1))
        center_gt = torch.cat((gt[:,:2]+gt[:,2:]/2,gt[:,2:]),-1)
        l1 = F.l1_loss(boxes,center_gt)
        location = focal(out['score_map'],heatmap)
        return 2*giou+5*l1+location, dict(giou=float(giou.detach()),l1=float(l1.detach()),
            focal=float(location.detach()),iou=float(iou.mean().detach()))
    with torch.no_grad():
        before = model(template=z,search=x)
    report['installed_layers'] = install_value_direction(model,rank=8,max_step=.1)
    with torch.no_grad():
        after = model(template=z,search=x)
    report['zero_full_model_parity'] = {key: torch.equal(before[key],after[key])
        for key in ('pred_boxes','score_map','size_map','offset_map')}
    if not all(report['zero_full_model_parity'].values()):
        raise RuntimeError('full-model zero parity failed')
    parent_state = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()
        if '.value_direction.' not in key}
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
    trace=[]
    for step in range(args.steps):
        optimizer.zero_grad()
        out = model(template=z,search=x)
        loss, parts = loss_of(out)
        if not torch.isfinite(loss):
            raise RuntimeError('real GT loss not finite')
        loss.backward()
        grads = {name:float(p.grad.norm()) for name,p in model.named_parameters()
            if p.requires_grad and p.grad is not None}
        expected_grads = sum(p.requires_grad for p in model.parameters())
        if len(grads) != expected_grads or not all(np.isfinite(v) for v in grads.values()):
            raise RuntimeError('real GT gradients missing/nonfinite')
        if step == 0 and not all(value > 0 for name,value in grads.items() if name.endswith('.up')):
            raise RuntimeError('real GT does not drive each up transform')
        torch.nn.utils.clip_grad_norm_(params,.1)
        optimizer.step()
        trace.append(dict(successful_update=step+1,loss=float(loss.detach()),
            components=parts,gradients=grads))
    report['optimization_trace'] = trace
    report['successful_optimizer_steps'] = len(trace)
    report['parent_state_unchanged'] = all(torch.equal(value,model.state_dict()[key].detach().cpu())
        for key,value in parent_state.items())
    if not report['parent_state_unchanged']:
        raise RuntimeError('frozen parent state drifted')
    readouts={}
    with torch.no_grad():
        for mode in ('off','direction','radial'):
            for module in model.modules():
                if hasattr(module,'value_direction_enabled'):
                    module.value_direction_enabled = mode != 'off'
                    module.value_direction_mode = mode if mode != 'off' else 'direction'
            out=model(template=z,search=x)
            loss,parts=loss_of(out)
            readouts[mode] = dict(loss=float(loss),components=parts)
    report['same_batch_readouts'] = readouts
    report['status'] = 'REAL-GT-GRADIENT-CHECK-COMPLETE / NO-EFFICACY-CLAIM'


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=8)
    args=parser.parse_args()
    if not 1 <= args.steps <= 32:
        raise ValueError('this diagnostic is limited to 1..32 updates')
    if args.output.exists():
        raise FileExistsError('diagnostic run output already exists')
    report={'scope':'two real training samples; fixed GT-centered crops; not overfit completion or evaluation'}
    start=time.monotonic()
    try:
        run(args,report)
    except Exception as error:
        report.update(status='FAILED-REAL-BATCH-CONTRACT',error=repr(error))
        raise
    finally:
        report['wall_seconds']=time.monotonic()-start
        report['gpu_hours_wall_upper_bound']=report['wall_seconds']/3600
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in
            ('optimization_trace','samples')},indent=2,ensure_ascii=False))
