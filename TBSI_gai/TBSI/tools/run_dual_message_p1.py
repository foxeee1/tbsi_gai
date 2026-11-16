#!/usr/bin/env python3
"""Isolated P1 runner. Does not import any historical admission/audit script."""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('TBSI_INFER_FP16', '0')
import cv2
import numpy as np
import torch
import yaml

RUN = Path(os.environ.get('TBSI_DUALMSG_RUN_ROOT', str(
    ROOT / 'experiments/early_gate_closed_loop_20261026/dual_message_p1_20261027')))
PARENT_CONFIG = ROOT / 'experiments/主实验/配置/M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023.yaml'
PUBLIC_PRETRAIN = ROOT / 'pretrained_models/TBSITrack_SOT_Pretrained.pth.tar'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def setup(seed):
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')


def config(path):
    from lib.config.tbsi_track.config import cfg, update_config_from_file
    update_config_from_file(str(path))
    return cfg


def tensor_hash(data, metadata):
    h = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for modality in ['visible', 'infrared']:
        for key in ['template_images', 'search_images', 'template_anno', 'search_anno',
                    'template_att', 'search_att', 'template_masks', 'search_masks']:
            t = data[modality][key].detach().cpu().contiguous()
            h.update((modality + ':' + key + ':' + str(t.dtype) + ':' + str(tuple(t.shape))).encode())
            h.update(t.numpy().tobytes())
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def environment():
    return {'python': sys.version, 'executable': sys.executable,
            'pytorch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0), 'gpu_count': torch.cuda.device_count(),
            'platform': platform.platform(), 'command': sys.argv,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'float32_matmul_precision': torch.get_float32_matmul_precision()}


def train(args, recorder=None):
    from lib.models.tbsi_track import build_tbsi_track
    from lib.train.actors import TBSITrackActor
    from lib.train.admin.settings import Settings
    from lib.train.base_functions import update_settings, get_optimizer_scheduler
    from lib.train.data import processing, transforms as tfm
    from lib.train.data.image_loader import opencv_loader
    from lib.train.dataset.lasher import LasHeR
    from lib.train.data.early_gate_paired_sampling import PairedStratifiedSampler, paired_collate
    from lib.utils import TensorDict
    from lib.utils.box_ops import giou_loss
    from lib.utils.focal_loss import FocalLoss

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'started.json').exists():
        raise RuntimeError('Refusing reused run identity: ' + str(output))
    cfg = config(args.config)
    if cfg.MODEL.INIT_CHECKPOINT:
        raise ValueError('Research contract forbids trained-tracker continuation: INIT_CHECKPOINT must be empty')
    if cfg.MODEL.PRETRAIN_FILE != PUBLIC_PRETRAIN.name or getattr(cfg.MODEL, 'STAGE2_BASELINE', ''):
        raise ValueError('Only the explicit public SOT pretraining initialization is valid; no multi-stage baseline')
    if sha(PUBLIC_PRETRAIN) != '1f6ebf5905ed0a404a11beac944ef5d544c9e81605120c2052a2518467d2651b':
        raise ValueError('Public pretrained checkpoint identity changed')
    setup(args.seed)
    if recorder is None:
        from record_dual_message_p1 import run_record
    else:
        run_record = recorder
    if not args.diagnostic_only:
        run_record(output, args.config, args.seed, args.arm, 'RUNNING / PAIRED-TRAINING',
                   'Training launched; no utility result yet.')
    write_json(output / 'started.json', {'time': time.time(), 'environment': environment(),
               'config': str(Path(args.config).resolve()), 'config_sha256': sha(args.config),
               'initial_checkpoint': str(PUBLIC_PRETRAIN.resolve()), 'initial_sha256': sha(PUBLIC_PRETRAIN),
               'training_semantics': 'single-stage joint training from public pretrained weights; no trained tracker, optimizer or scheduler is resumed',
               'manifest_sha256': sha(args.manifest), 'arm': args.arm, 'seed': args.seed})
    net = build_tbsi_track(cfg, training=True)
    export_shared_path = getattr(args, 'export_shared_init', None)
    if export_shared_path:
        export_path = Path(export_shared_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        exported_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        # The candidate has extra value-direction tensors.  The paired control
        # must be checked against the intersection of public/shared tensors,
        # rather than the candidate's full state.
        exported_common_state = {
            k: v for k, v in exported_state.items() if '.value_direction.' not in k
        }
        torch.save({'state': exported_state,
                    'source_state_sha256': state_hash(exported_state),
                    'source_shared_state_sha256': state_hash(exported_common_state),
                    'source_identity': {'config': str(Path(args.config).resolve()),
                                        'config_sha256': sha(args.config),
                                        'public_pretrain_sha256': sha(PUBLIC_PRETRAIN),
                                        'seed': args.seed,
                                        'semantics': 'public SOT fresh initialization; pre-optimizer'}},
                   str(export_path))
        write_json(output / 'shared_init_export.json', {'path': str(export_path.resolve()),
                   'sha256': sha(export_path), 'state_sha256': state_hash(net.state_dict()),
                   'stage': 'after public checkpoint load, before optimizer/training'})
    shared_init_path = getattr(args, 'shared_init_snapshot', None)
    shared_init_meta = None
    paired_common_state_sha256 = None
    copied_shared_keys = []
    if shared_init_path:
        snapshot = torch.load(str(Path(shared_init_path)), map_location='cpu')
        if not isinstance(snapshot, dict) or 'state' not in snapshot:
            raise ValueError('Shared initialization snapshot must contain a state mapping')
        current = net.state_dict()
        for key, value in snapshot['state'].items():
            if key in current and tuple(current[key].shape) == tuple(value.shape):
                # Copy from the common public-pretrained construction before creating the optimizer.
                current[key].copy_(value)
                copied_shared_keys.append(key)
        if not copied_shared_keys:
            raise ValueError('Shared initialization snapshot has no compatible tensors')
        net.load_state_dict(current, strict=True)
        shared_init_meta = {'path': str(Path(shared_init_path).resolve()),
                            'sha256': sha(shared_init_path),
                            'copied_tensors': len(copied_shared_keys),
                            'source_identity': snapshot.get('source_identity', 'unknown')}
        snapshot_common = {
            k: v for k, v in snapshot['state'].items()
            if '.value_direction.' not in k and k in current
            and tuple(current[k].shape) == tuple(v.shape)
        }
        # The control may omit candidate-only modules beyond value_direction.
        # Hash exactly the compatible key/shape intersection used for copying.
        paired_common_state_sha256 = state_hash(snapshot_common)
    full_state = net.state_dict()
    shared_state = {k: v for k, v in full_state.items() if '.value_direction.' not in k}
    expected = torch.load(str(PUBLIC_PRETRAIN), map_location='cpu')['net']
    loaded_keys = sorted(k for k in shared_state if k in expected and shared_state[k].shape == expected[k].shape)
    disagreements = [k for k in loaded_keys if not torch.equal(shared_state[k], expected[k])]
    if disagreements:
        raise ValueError('Common pretrained tensor not loaded unchanged: ' + repr(disagreements))
    new_architecture_keys = sorted(set(shared_state) - set(expected))
    new_parameter_count = sum(p.numel() for n, p in net.named_parameters()
                              if n in new_architecture_keys)
    actual_hash = state_hash(shared_state)
    expected_hash = getattr(args, 'expected_shared_state_sha256', None)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError('Shared pre-optimizer state hash mismatch: expected %s, got %s' %
                         (expected_hash, actual_hash))
    matched_state = {k: net.state_dict()[k] for k in copied_shared_keys}
    matched_state_hash = state_hash(matched_state) if matched_state else actual_hash
    if shared_init_meta and paired_common_state_sha256 and matched_state_hash != paired_common_state_sha256:
        raise ValueError('Common post-load state differs from P initialization snapshot')
    write_json(output / 'initial_state.json', {'shared_state_sha256': actual_hash,
               'matched_parent_state_sha256': matched_state_hash,
               'matched_parent_key_count': len(matched_state),
               'matched_parent_keys': copied_shared_keys,
               'public_pretrain_sha256': sha(PUBLIC_PRETRAIN), 'public_loaded_tensor_count': len(loaded_keys),
               'public_unexpected_keys': sorted(set(expected) - set(shared_state)),
               'common_architecture_new_keys': new_architecture_keys,
               'shared_tensor_count': len(shared_state), 'seed': args.seed,
               'compare_shared_state_hash_against_other_arm_before_utility_ranking': True,
               'new_parameters': new_parameter_count,
               'shared_init_snapshot': shared_init_meta,
               'paired_common_state_sha256': paired_common_state_sha256})
    del expected, shared_state, full_state
    net = net.cuda().train()
    settings = Settings()
    settings.local_rank = -1
    update_settings(settings, cfg)
    settings.num_template = cfg.DATA.TEMPLATE.NUMBER
    settings.num_search = cfg.DATA.SEARCH.NUMBER
    with contextlib.redirect_stdout(io.StringIO()) as optimizer_log:
        optimizer, scheduler = get_optimizer_scheduler(net, cfg)
    (output / 'optimizer_membership.txt').write_text(optimizer_log.getvalue())
    counts = [sum(p.numel() for p in g['params']) for g in optimizer.param_groups]
    expected_ids = {id(p) for p in net.parameters() if p.requires_grad}
    observed_ids = [id(p) for g in optimizer.param_groups for p in g['params']]
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != expected_ids:
        raise ValueError('Optimizer membership is incomplete or duplicated')
    write_json(output / 'optimizer_contract.json', {'groups': [dict(lr=g['lr'], parameters=n,
               weight_decay=g['weight_decay']) for g, n in zip(optimizer.param_groups, counts)],
               'fresh_adamw': True, 'lr_drop_full_epoch': cfg.TRAIN.LR_DROP_EPOCH,
               'no_quick_schedule_compression': True})
    manifest = json.loads(Path(args.manifest).read_text())
    dataset = LasHeR(str(ROOT / 'data/lasher/trainingset'), split='train',
                    image_loader=opencv_loader, sequence_list_file=str(Path(args.manifest).parent / 'train_sequences.txt'))
    joint = tfm.Transform(tfm.ToGrayscale(probability=0), tfm.RandomHorizontalFlip(probability=0.5))
    transform = tfm.Transform(tfm.ToTensorAndJitter(0.2), tfm.RandomHorizontalFlip_Norm(probability=0.5),
                              tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
    process = processing.STARKProcessing(search_area_factor=settings.search_area_factor,
                  output_sz=settings.output_sz, center_jitter_factor=settings.center_jitter_factor,
                  scale_jitter_factor=settings.scale_jitter_factor, mode='sequence', transform=transform,
                  joint_transform=joint, settings=settings)
    samples = args.updates * cfg.TRAIN.BATCH_SIZE * cfg.TRAIN.GRAD_ACCUM_STEPS
    sampler = PairedStratifiedSampler(manifest, args.seed, datasets=[dataset], p_datasets=[1],
                  samples_per_epoch=samples, max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
                  num_search_frames=settings.num_search, num_template_frames=settings.num_template,
                  processing=process, frame_sample_mode=cfg.DATA.SAMPLER_MODE)
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(sampler, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                  num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, collate_fn=paired_collate,
                  pin_memory=True, prefetch_factor=cfg.TRAIN.PREFETCH_FACTOR,
                  persistent_workers=cfg.TRAIN.PERSISTENT_WORKERS, generator=generator)
    actor = TBSITrackActor(net, {'giou': giou_loss, 'l1': torch.nn.functional.l1_loss,
                  'focal': FocalLoss()}, {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT,
                  'focal': 1.}, settings, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.AMP, init_scale=128.)
    # Input loader has its own RNG. Candidate installation preserves model RNG.
    setup(args.seed)
    iterator = iter(loader)
    seen, attempted, successful, failed_attempts = set(), 0, 0, 0
    started = time.time()
    milestones = [int(s) for s in args.milestones.split(',') if s.strip()]
    contract_parameters = [(name, param) for name, param in net.named_parameters()
                           if ('structured_evidence.' in name or
                               'dual_message.' in name or
                               'fixed_source_gate.' in name or
                               'fu_allocator.' in name or
                               'receiver_write_' in name or
                               'early_gate_scale' in name or
                               'cross_relation_' in name)]
    diagnostic_history = []
    with (output / 'batches.jsonl').open('w') as batch_file, (output / 'updates.jsonl').open('w') as step_file:
        while successful < args.updates:
            window = []
            for _ in range(cfg.TRAIN.GRAD_ACCUM_STEPS):
                data, meta = next(iterator)
                digest = tensor_hash(data, meta)
                template_box_v = data['visible']['template_anno'][0].float()
                template_box_i = data['infrared']['template_anno'][0].float()
                batch_file.write(json.dumps({'update': successful + 1, 'microbatch': len(window),
                     'input_sha256': digest, 'samples': meta,
                     'template_box_sha256': hashlib.sha256(
                         template_box_v.contiguous().numpy().tobytes()).hexdigest(),
                     'template_box_shape': list(template_box_v.shape),
                     'template_box_minmax': [float(template_box_v.min()), float(template_box_v.max())],
                     'rgb_tir_template_box_max_abs_delta': float((template_box_v-template_box_i).abs().max())},
                     sort_keys=True) + '\n')
                seen.update(m['sequence'] for m in meta)
                window.append(data)
            batch_file.flush()
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all()
            buffers = {n: b.detach().clone() for n, b in net.named_buffers()}
            for retry in range(8):
                attempted += 1
                optimizer.zero_grad(set_to_none=True)
                losses, ious = [], []
                for data in window:
                    gpu_data = TensorDict({key: TensorDict(value).to('cuda', non_blocking=True)
                                           for key, value in data.items()})
                    gpu_data['epoch'] = 1 + (successful // 468)
                    gpu_data['settings'] = settings
                    with torch.cuda.amp.autocast(enabled=cfg.TRAIN.AMP):
                        loss, stats = actor(gpu_data)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite task loss')
                    scaler.scale(loss / cfg.TRAIN.GRAD_ACCUM_STEPS).backward()
                    losses.append(float(stats['Loss/total']))
                    ious.append(float(stats['IoU']))
                    del gpu_data, loss
                scaler.unscale_(optimizer)
                contract_gradients = {}
                contract_before = {}
                if args.diagnostic_only:
                    for name, param in contract_parameters:
                        if param.grad is not None:
                            contract_gradients[name] = float(param.grad.detach().float().norm())
                        if any(token in name for token in (
                                'structured_evidence.encoder.0.weight',
                                'structured_evidence.support.weight',
                                'dual_message.self_head.2.weight',
                                'dual_message.cross_head.2.weight',
                                'fixed_source_gate.source_logits',
                                'fu_allocator.relation_scale',
                                'fu_allocator.rejection_bias',
                                'cross_relation_rgb.beta',
                                'cross_relation_tir.beta')):
                            contract_before[name] = param.detach().clone()
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.TRAIN.GRAD_CLIP_NORM)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                updated = scaler.get_scale() >= previous_scale
                if updated:
                    successful += 1
                else:
                    failed_attempts += 1
                row = {'attempt': attempted, 'successful_updates': successful, 'updated': updated,
                       'overflow_retry': retry, 'loss': float(np.mean(losses)), 'iou': float(np.mean(ious)),
                       'gradient_norm': float(grad_norm) if torch.isfinite(grad_norm) else None,
                       'loss_scale': scaler.get_scale(), 'unique_sequences': len(seen),
                       'consumed_samples': successful * 128, 'seconds': time.time() - started}
                step_file.write(json.dumps(row, allow_nan=False) + '\n'); step_file.flush()
                if updated:
                    if args.diagnostic_only:
                        changed = {}
                        values = {}
                        for name, before in contract_before.items():
                            param = dict(net.named_parameters())[name]
                            changed[name] = bool(not torch.equal(before, param.detach()))
                            values[name] = {'before_l2': float(before.float().norm()),
                                            'after_l2': float(param.detach().float().norm())}
                        diagnostic_history.append({'successful_update': successful,
                            'gradient_norms': contract_gradients, 'parameter_updates': changed,
                            'parameter_values': values})
                        write_json(output / 'operator_contract.json', {
                            'mode': 'real LasHeR batch; diagnostic-only; no checkpoint written',
                            'successful_updates': successful,
                            'finite_task_loss': all(np.isfinite(losses)),
                            'template_boxes_normalized_xywh': True,
                            'template_boxes_match_rgb_tir': max(
                                float((w['visible']['template_anno'][0]-w['infrared']['template_anno'][0]).abs().max())
                                for w in window) <= 1e-6,
                            'template_boxes_batch_shapes': sorted(set(tuple(w['visible']['template_anno'][0].shape)
                                for w in window)),
                            'update_history': diagnostic_history,
                            'new_parameter_count_in_optimizer': sum(p.numel() for n, p in net.named_parameters()
                                if ('structured_evidence.' in n or 'dual_message.' in n or
                                    'fu_allocator.' in n or
                                    'fixed_source_gate.' in n or
                                    'cross_relation_' in n) and p.requires_grad),
                            'no_checkpoint_written': not any(output.glob('checkpoint_*.pth.tar'))})
                    break
                # Repeat this exact input/dropout window at lower loss scale.
                # Undo BN buffer advancement from the skipped attempt.
                with torch.no_grad():
                    for n, b in net.named_buffers():
                        b.copy_(buffers[n])
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state_all(cuda_rng)
            else:
                raise RuntimeError('8 AMP overflows on one fixed update window')
            if successful % 16 == 0 or successful in milestones:
                print(json.dumps(row), flush=True)
                write_json(output / 'progress.json', row)
            if successful % 468 == 0:
                scheduler.step()
            if successful in milestones and not args.diagnostic_only:
                checkpoint = output / ('checkpoint_u%04d.pth.tar' % successful)
                checkpoint_payload = {'net': net.state_dict(), 'epoch': successful / 468.,
                            'successful_updates': successful, 'attempted_updates': attempted,
                            'seed': args.seed, 'arm': args.arm, 'config_sha256': sha(args.config),
                            'manifest_sha256': sha(args.manifest), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                            'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
                if getattr(args, 'checkpoint_weights_only', False):
                    # Opt-in storage policy; optimizer updates and evaluation
                    # tensors are unchanged. Such artifacts cannot resume training.
                    for storage_key in ('optimizer', 'scheduler', 'scaler', 'cpu_rng', 'cuda_rng'):
                        checkpoint_payload.pop(storage_key)
                    checkpoint_payload['storage_policy'] = 'weights-only; no optimizer resume'
                torch.save(checkpoint_payload, str(checkpoint))
                write_json(output / ('checkpoint_u%04d.json' % successful),
                           {'path': str(checkpoint), 'sha256': sha(checkpoint), 'progress': row})
    write_json(output / 'completed.json', {'successful_updates': successful, 'attempted_updates': attempted,
               'skipped_attempts_replayed': failed_attempts, 'samples': successful * 128,
               'unique_sequences': len(seen), 'sequence_coverage': sorted(seen),
               'expected_sequences': len(manifest['train_sequences']), 'gpu_wall_hours': (time.time() - started)/3600.})
    if args.diagnostic_only:
        if not (output / 'operator_contract.json').exists():
            raise RuntimeError('Diagnostic run ended without operator contract record')
        if recorder is None:
            from record_dual_message_p1 import run_record
        else:
            run_record = recorder
        run_record(output, args.config, args.seed, 'P0-OPERATOR-CONTRACT',
                   'P0-COMPLETE / LEARNING-TRANSMISSION-ONLY',
                   'One real-data optimizer update; no utility claim and no checkpoint reuse.',
                   checkpoint='NA / diagnostic checkpoint intentionally not saved',
                   gpu_hours=(time.time() - started)/3600.)
    else:
        run_record(output, args.config, args.seed, args.arm, 'TRAINING-COMPLETE / EVALUATION-PENDING',
                   '%d successful updates; %d/%d sequences; %d skipped attempts replayed; no efficacy claim.' %
                   (successful, len(seen), len(manifest['train_sequences']), failed_attempts),
                   str(output / ('checkpoint_u%04d.pth.tar' % successful)), (time.time() - started)/3600.)


def evaluate(args):
    from lib.test.evaluation import get_dataset
    from lib.test.evaluation.tracker import Tracker
    setup(args.seed)
    torch.cuda.reset_peak_memory_stats()
    os.environ['TBSI_CHECKPOINT_OVERRIDE'] = str(Path(args.checkpoint).resolve())
    os.environ['TBSI_INFER_FP16'] = '0'
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prefetch_workers = int(getattr(args, 'prefetch_workers', 0))
    if prefetch_workers < 0:
        raise ValueError('prefetch_workers must be non-negative')
    identity = {'checkpoint': str(Path(args.checkpoint).resolve()), 'sha256': sha(args.checkpoint),
                'config': str(Path(args.config).resolve()), 'config_sha256': sha(args.config),
                'environment': environment(), 'precision': 'FP32', 'threads': 0,
                'image_prefetch_threads': prefetch_workers, 'template_update': False}
    if (output / 'identity.json').exists():
        old = json.loads((output / 'identity.json').read_text())
        if old['sha256'] != identity['sha256'] or old['config_sha256'] != identity['config_sha256'] or \
                old.get('image_prefetch_threads', 0) != prefetch_workers:
            raise ValueError('Evaluation identity changed')
    else:
        write_json(output / 'identity.json', identity)
    dataset = get_dataset('lasher_test')
    if args.list:
        names = set(Path(args.list).read_text().splitlines())
        dataset = [s for s in dataset if s.name in names]
        if len(dataset) != len(names):
            raise ValueError('Requested evaluation sequence missing')
    wrapper = Tracker('tbsi_track', Path(args.config).stem, 'lasher_test')
    started, total_frames = time.time(), 0
    for index, sequence in enumerate(dataset):
        target = output / (sequence.name + '.txt')
        if target.exists():
            saved = np.loadtxt(str(target), ndmin=2)
            if saved.shape != sequence.ground_truth_rect.shape or not np.isfinite(saved).all():
                raise ValueError('Invalid existing result: ' + str(target))
            total_frames += len(saved)
            continue
        result = wrapper.run_sequence(sequence, debug=0, prefetch_workers=prefetch_workers)
        # New directory; preserve native FP32 tracker behavior and full-precision outputs.
        boxes = np.asarray(result['target_bbox'], dtype=np.float64)
        if boxes.shape != sequence.ground_truth_rect.shape or not np.isfinite(boxes).all():
            raise ValueError('Missing/invalid frame predictions: ' + sequence.name)
        np.savetxt(str(target), boxes, fmt='%.9f', delimiter='\t')
        np.savetxt(str(output / (sequence.name + '_time.txt')), result['time'], fmt='%.9f')
        total_frames += len(boxes)
        progress = {'sequence': sequence.name, 'finished': index + 1, 'total': len(dataset),
                    'frames': total_frames, 'wall_seconds': time.time() - started}
        print(json.dumps(progress), flush=True)
        write_json(output / 'progress.json', progress)
    elapsed = time.time() - started
    frame_times = [float(x) for path in output.glob('*_time.txt')
                   for x in np.loadtxt(str(path), ndmin=1).tolist() if np.isfinite(x) and x > 0]
    write_json(output / 'completed.json', {'sequences': len(dataset), 'frames': total_frames,
               'seconds': elapsed, 'throughput_fps': total_frames / max(elapsed, 1e-9),
               'per_frame_latency_ms_p50': float(np.percentile(frame_times, 50) * 1000) if frame_times else None,
               'per_frame_latency_ms_p95': float(np.percentile(frame_times, 95) * 1000) if frame_times else None,
               'peak_gpu_memory_bytes': int(torch.cuda.max_memory_allocated()),
               'prediction_sha256': {p.name: sha(p) for p in sorted(output.glob('*.txt'))}})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['train', 'evaluate'])
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--arm', default='parent_reference')
    p.add_argument('--manifest', default=str(RUN / 'data_manifest.json'))
    p.add_argument('--updates', type=int, default=468)
    p.add_argument('--milestones', default='234,468')
    p.add_argument('--checkpoint')
    p.add_argument('--list')
    p.add_argument('--shared-init-snapshot')
    p.add_argument('--export-shared-init')
    p.add_argument('--diagnostic-only', action='store_true')
    p.add_argument('--checkpoint-weights-only', action='store_true')
    args = p.parse_args()
    try:
        train(args) if args.mode == 'train' else evaluate(args)
    except Exception as exc:
        import traceback
        write_json(Path(args.output) / 'failure.json', {'type': type(exc).__name__,
                  'error': str(exc), 'traceback': traceback.format_exc(), 'time': time.time()})
        from record_dual_message_p1 import run_record
        if args.mode == 'train':
            run_record(args.output, args.config, args.seed, args.arm,
                       'EXECUTION-FAILED / NO-UTILITY-CONCLUSION', type(exc).__name__ + ': ' + str(exc))
        raise


if __name__ == '__main__':
    main()
