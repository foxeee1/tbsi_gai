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

RUN = ROOT / 'experiments/early_gate_closed_loop_20261026/p1_sot_v3'
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
        from record_early_gate_p1_v2 import run_record
    else:
        run_record = recorder
    run_record(output, args.config, args.seed, args.arm, 'RUNNING / PAIRED-TRAINING',
               'Training launched; no utility result yet.')
    write_json(output / 'started.json', {'time': time.time(), 'environment': environment(),
               'config': str(Path(args.config).resolve()), 'config_sha256': sha(args.config),
               'initial_checkpoint': str(PUBLIC_PRETRAIN.resolve()), 'initial_sha256': sha(PUBLIC_PRETRAIN),
               'training_semantics': 'single-stage joint training from public pretrained weights; no trained tracker, optimizer or scheduler is resumed',
               'manifest_sha256': sha(args.manifest), 'arm': args.arm, 'seed': args.seed})
    net = build_tbsi_track(cfg, training=True)
    shared_init_path = getattr(args, 'shared_init_snapshot', None)
    shared_init_meta = None
    paired_common_state_sha256 = None
    if shared_init_path:
        snapshot = torch.load(str(Path(shared_init_path)), map_location='cpu')
        if not isinstance(snapshot, dict) or 'state' not in snapshot:
            raise ValueError('Shared initialization snapshot must contain a state mapping')
        current = net.state_dict()
        copied = []
        for key, value in snapshot['state'].items():
            if key in current and tuple(current[key].shape) == tuple(value.shape):
                current[key].copy_(value)
                copied.append(key)
        if not copied:
            raise ValueError('Shared initialization snapshot has no compatible tensors')
        net.load_state_dict(current, strict=True)
        shared_init_meta = {'path': str(Path(shared_init_path).resolve()),
                            'sha256': sha(shared_init_path),
                            'copied_tensors': len(copied),
                            'source_identity': snapshot.get('source_identity', 'unknown')}
        paired_common_state_sha256 = snapshot.get('source_shared_state_sha256')
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
    write_json(output / 'initial_state.json', {'shared_state_sha256': actual_hash,
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
    def build_loader():
        sampler = PairedStratifiedSampler(manifest, args.seed, datasets=[dataset], p_datasets=[1],
                      samples_per_epoch=samples, max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
                      num_search_frames=settings.num_search, num_template_frames=settings.num_template,
                      processing=process, frame_sample_mode=cfg.DATA.SAMPLER_MODE)
        generator = torch.Generator().manual_seed(args.seed)
        return torch.utils.data.DataLoader(sampler, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                      num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, collate_fn=paired_collate,
                      pin_memory=True, prefetch_factor=cfg.TRAIN.PREFETCH_FACTOR,
                      persistent_workers=cfg.TRAIN.PERSISTENT_WORKERS, generator=generator)

    loader = build_loader()
    actor = TBSITrackActor(net, {'giou': giou_loss, 'l1': torch.nn.functional.l1_loss,
                  'focal': FocalLoss()}, {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT,
                  'focal': 1.}, settings, cfg)
    setup(args.seed)
    utility_calibration = None
    direct_margin_calibration = None
    if actor.source_utility is not None:
        calibration_rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all(),
                           random.getstate(), np.random.get_state())
        buffers = {name: value.detach().clone() for name, value in net.named_buffers()}
        probe_iter = iter(loader)
        task_norms, rank_norms, hashes = [], [], []
        valid_rows, valid_gt_fractions, stable_pair_fractions = [], [], []
        actor._utility_calibration_mode = True
        actor._utility_loss_weight = 0.0
        with torch.enable_grad():
            for _ in range(8):
                data, meta = next(probe_iter)
                hashes.append(tensor_hash(data, meta))
                gpu_data = TensorDict({key: TensorDict(value).to('cuda', non_blocking=True)
                                       for key, value in data.items()})
                gpu_data['epoch'] = 1
                gpu_data['settings'] = settings
                with torch.cuda.amp.autocast(enabled=cfg.TRAIN.AMP):
                    probe_loss, probe_stats = actor(gpu_data)
                task_norms.append(probe_stats['MSHA/source_utility_task_gate_grad_norm'])
                rank_norms.append(probe_stats['MSHA/source_utility_rank_gate_grad_norm'])
                valid_rows.append(probe_stats.get('MSHA/source_utility_valid_rows', 0.0))
                valid_gt_fractions.append(probe_stats.get('MSHA/source_utility_valid_gt_fraction', 0.0))
                stable_pair_fractions.append(probe_stats.get('MSHA/source_utility_stable_pair_fraction', 0.0))
                del gpu_data, probe_loss
        del probe_iter
        with torch.no_grad():
            for name, value in net.named_buffers():
                value.copy_(buffers[name])
        actor._utility_calibration_mode = False
        task_median = float(np.median(task_norms))
        rank_median = float(np.median(rank_norms))
        if not np.isfinite(task_median + rank_median) or task_median <= 0 or rank_median <= 0:
            raise RuntimeError('Utility loss calibration has zero/nonfinite gate gradient norm')
        target_ratio = float(cfg.TRAIN.MSHA_SOURCE_UTILITY_WEIGHT)
        effective_weight = target_ratio * task_median / rank_median
        actor._utility_loss_weight = effective_weight if target_ratio > 0 else 0.0
        utility_calibration = {'microbatch_sha256': hashes,
             'task_gate_grad_norms': task_norms, 'rank_gate_grad_norms': rank_norms,
             'valid_rows': valid_rows, 'valid_gt_fractions': valid_gt_fractions,
             'stable_pair_fractions': stable_pair_fractions,
             'task_gate_grad_norm_median': task_median,
             'rank_gate_grad_norm_median': rank_median,
             'target_rank_to_task_gradient_ratio': target_ratio,
             'effective_loss_weight': actor._utility_loss_weight}
        write_json(output / 'utility_calibration.json', utility_calibration)
        loader = build_loader()
        del calibration_rng
        setup(args.seed)
    elif actor.direct_margin is not None:
        buffers = {name: value.detach().clone() for name, value in net.named_buffers()}
        probe_iter = iter(loader)
        task_norms, margin_norms, hashes, valid_fractions, q_means = [], [], [], [], []
        actor._direct_margin_calibration_mode = True
        actor._direct_margin_weight = 0.0
        with torch.enable_grad():
            for _ in range(8):
                data, meta = next(probe_iter)
                hashes.append(tensor_hash(data, meta))
                gpu_data = TensorDict({key: TensorDict(value).to('cuda', non_blocking=True)
                                       for key, value in data.items()})
                gpu_data['epoch'] = 1
                gpu_data['settings'] = settings
                with torch.cuda.amp.autocast(enabled=cfg.TRAIN.AMP):
                    probe_loss, probe_stats = actor(gpu_data)
                task_norms.append(probe_stats['MSHA/direct_margin_task_score_grad_norm'])
                margin_norms.append(probe_stats['MSHA/direct_margin_aux_score_grad_norm'])
                valid_fractions.append(probe_stats['MSHA/direct_margin_valid_fraction'])
                q_means.append(probe_stats['MSHA/direct_margin_q_mean'])
                del gpu_data, probe_loss
        del probe_iter
        with torch.no_grad():
            for name, value in net.named_buffers():
                value.copy_(buffers[name])
        actor._direct_margin_calibration_mode = False
        task_median = float(np.median(task_norms))
        margin_median = float(np.median(margin_norms))
        target_ratio = float(cfg.TRAIN.MSHA_DIRECT_MARGIN_WEIGHT)
        if (not np.isfinite(task_median + margin_median) or
                (target_ratio > 0 and (task_median <= 0 or margin_median <= 0))):
            raise RuntimeError('Direct margin score-gradient calibration is zero/nonfinite')
        effective_weight = (target_ratio * task_median / margin_median
                            if target_ratio > 0 else 0.0)
        actor._direct_margin_weight = effective_weight
        direct_margin_calibration = {
            'microbatch_sha256': hashes,
            'task_score_grad_norms': task_norms,
            'margin_score_grad_norms': margin_norms,
            'valid_fractions': valid_fractions,
            'q_means': q_means,
            'task_score_grad_norm_median': task_median,
            'margin_score_grad_norm_median': margin_median,
            'target_margin_to_task_gradient_ratio': target_ratio,
            'effective_loss_weight': effective_weight,
        }
        write_json(output / 'direct_margin_calibration.json', direct_margin_calibration)
        loader = build_loader()
        setup(args.seed)
    calibration_record = utility_calibration or direct_margin_calibration
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.AMP, init_scale=128.)
    # Rebuilt workers and reset model/RNG state make calibration invisible to training.
    iterator = iter(loader)
    seen, attempted, successful, failed_attempts = set(), 0, 0, 0
    started = time.time()
    milestones = [int(s) for s in args.milestones.split(',')]
    with (output / 'batches.jsonl').open('w') as batch_file, (output / 'updates.jsonl').open('w') as step_file:
        while successful < args.updates:
            window = []
            for _ in range(cfg.TRAIN.GRAD_ACCUM_STEPS):
                data, meta = next(iterator)
                digest = tensor_hash(data, meta)
                if successful == 0 and len(window) < 8 and calibration_record is not None:
                    expected_digest = calibration_record['microbatch_sha256'][len(window)]
                    if digest != expected_digest:
                        raise RuntimeError('Utility calibration changed the deterministic training prefix')
                batch_file.write(json.dumps({'update': successful + 1, 'microbatch': len(window),
                     'input_sha256': digest, 'samples': meta}, sort_keys=True) + '\n')
                seen.update(m['sequence'] for m in meta)
                window.append(data)
            batch_file.flush()
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all()
            buffers = {n: b.detach().clone() for n, b in net.named_buffers()}
            for retry in range(8):
                attempted += 1
                optimizer.zero_grad(set_to_none=True)
                losses, ious, utility_coverage, direct_margin_coverage = [], [], [], []
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
                    if actor.source_utility is not None:
                        utility_coverage.append({
                            'valid_rows': float(stats.get('MSHA/source_utility_valid_rows', 0.0)),
                            'stable_pair_fraction': float(stats.get('MSHA/source_utility_stable_pair_fraction', 0.0)),
                            'valid_gt_fraction': float(stats.get('MSHA/source_utility_valid_gt_fraction', 0.0)),
                        })
                    if actor.direct_margin is not None:
                        direct_margin_coverage.append({
                            'valid_fraction': float(stats.get('MSHA/direct_margin_valid_fraction', 0.0)),
                            'q_mean': float(stats.get('MSHA/direct_margin_q_mean', 0.0)),
                            'loss_raw': float(stats.get('MSHA/direct_margin_loss_raw', 0.0)),
                            'loss_weighted': float(stats.get('Loss/direct_localization_margin_weighted', 0.0)),
                        })
                    del gpu_data, loss
                scaler.unscale_(optimizer)
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
                if utility_coverage:
                    row['utility_coverage'] = {
                        key: float(np.mean([item[key] for item in utility_coverage]))
                        for key in utility_coverage[0]}
                if direct_margin_coverage:
                    row['direct_margin'] = {
                        key: float(np.mean([item[key] for item in direct_margin_coverage]))
                        for key in direct_margin_coverage[0]}
                step_file.write(json.dumps(row, allow_nan=False) + '\n'); step_file.flush()
                if updated:
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
            if successful in milestones:
                checkpoint = output / ('checkpoint_u%04d.pth.tar' % successful)
                checkpoint_payload = {
                    'net': net.state_dict(), 'epoch': successful / 468.,
                    'successful_updates': successful, 'attempted_updates': attempted,
                    'seed': args.seed, 'arm': args.arm, 'config_sha256': sha(args.config),
                    'manifest_sha256': sha(args.manifest),
                }
                if not args.inference_only_checkpoint:
                    checkpoint_payload.update({
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                        'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                    })
                torch.save(checkpoint_payload, str(checkpoint))
                write_json(output / ('checkpoint_u%04d.json' % successful),
                           {'path': str(checkpoint), 'sha256': sha(checkpoint), 'progress': row})
    write_json(output / 'completed.json', {'successful_updates': successful, 'attempted_updates': attempted,
               'skipped_attempts_replayed': failed_attempts, 'samples': successful * 128,
               'unique_sequences': len(seen), 'sequence_coverage': sorted(seen),
               'expected_sequences': len(manifest['train_sequences']), 'gpu_wall_hours': (time.time() - started)/3600.,
               'utility_calibration': utility_calibration,
               'direct_margin_calibration': direct_margin_calibration})
    run_record(output, args.config, args.seed, args.arm, 'TRAINING-COMPLETE / EVALUATION-PENDING',
               '%d successful updates; %d/%d sequences; %d skipped attempts replayed; no efficacy claim.' %
               (successful, len(seen), len(manifest['train_sequences']), failed_attempts),
               str(output / ('checkpoint_u%04d.pth.tar' % successful)), (time.time() - started)/3600.)


def evaluate(args):
    from lib.test.evaluation import get_dataset
    from lib.test.evaluation.tracker import Tracker
    setup(args.seed)
    os.environ['TBSI_CHECKPOINT_OVERRIDE'] = str(Path(args.checkpoint).resolve())
    os.environ['TBSI_INFER_FP16'] = '0'
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {'checkpoint': str(Path(args.checkpoint).resolve()), 'sha256': sha(args.checkpoint),
                'config': str(Path(args.config).resolve()), 'config_sha256': sha(args.config),
                'environment': environment(), 'precision': 'FP32', 'threads': 0, 'template_update': False}
    if (output / 'identity.json').exists():
        old = json.loads((output / 'identity.json').read_text())
        if old['sha256'] != identity['sha256'] or old['config_sha256'] != identity['config_sha256']:
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
        result = wrapper.run_sequence(sequence, debug=0)
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
    write_json(output / 'completed.json', {'sequences': len(dataset), 'frames': total_frames,
               'seconds': time.time() - started, 'prediction_sha256': {p.name: sha(p) for p in sorted(output.glob('*.txt'))}})


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
    p.add_argument('--record-module')
    p.add_argument('--inference-only-checkpoint', action='store_true')
    args = p.parse_args()
    recorder = None
    if args.record_module:
        sys.path.insert(0, str(ROOT / 'tools'))
        module = __import__(args.record_module)
        recorder = module.run_record
    try:
        if args.mode == 'train':
            train(args, recorder=recorder)
        else:
            evaluate(args)
    except Exception as exc:
        import traceback
        write_json(Path(args.output) / 'failure.json', {'type': type(exc).__name__,
                  'error': str(exc), 'traceback': traceback.format_exc(), 'time': time.time()})
        from record_early_gate_p1_v2 import run_record
        if args.mode == 'train':
            callback = recorder or run_record
            callback(args.output, args.config, args.seed, args.arm,
                     'EXECUTION-FAILED / NO-UTILITY-CONCLUSION', type(exc).__name__ + ': ' + str(exc))
        raise


if __name__ == '__main__':
    main()
