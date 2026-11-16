#!/usr/bin/env python3
"""Fixed-state and short closed-loop path attribution for the ED checkpoint."""
import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ['TBSI_INFER_FP16'] = '0'

import cv2
import numpy as np
import torch

RUN = Path('/autodl-fs/data/early_gate_closed_loop/dual_message_p1_v2_20261027/runs')
P2 = Path('/autodl-fs/data/early_gate_closed_loop/dual_message_p1_v2_20261027/p2_attribution_retry4')
CHECKPOINT = RUN / 'ED/checkpoint_u0468.pth.tar'
CHECKPOINT_SHA = 'caa2b5f08f4769e9fb434819cd2935fc30d9ddb3b2e116284ac1fdd79ffaaec9'
CONFIG_NAME = 'EG_DUALMSG_V2_ED_QUICK_SOT_S42_20261027'
CONFIG = ROOT / ('experiments/tbsi_track/' + CONFIG_NAME + '.yaml')
CASES = [
    dict(sequence='10runone', frame=69, role='failure-onset'),
    dict(sequence='1boycoming', frame=69, role='stable-control'),
    dict(sequence='umbrellawillopen', frame=272, role='failure-onset'),
    dict(sequence='bawgirl', frame=272, role='stable-control'),
]
MODES = ['ed', 'self_identity', 'cross_identity', 'both_identity', 'remove_cross_message']
HORIZON = 20


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous()
        return hashlib.sha256((str(value.dtype) + str(tuple(value.shape))).encode() +
                              value.numpy().tobytes()).hexdigest()
    if isinstance(value, (list, tuple)):
        return hashlib.sha256(''.join(tensor_sha(item) for item in value).encode()).hexdigest()
    if isinstance(value, dict):
        return hashlib.sha256(''.join(str(key) + tensor_sha(value[key])
                                      for key in sorted(value)).encode()).hexdigest()
    raise TypeError('Unsupported tensor hash input: ' + type(value).__name__)


def clone_state(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, list):
        return [clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_state(item) for item in value)
    if isinstance(value, dict):
        return {key: clone_state(item) for key, item in value.items()}
    return copy.deepcopy(value)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def bbox_iou(a, b):
    ax1, ay1, aw, ah = [float(x) for x in a]
    bx1, by1, bw, bh = [float(x) for x in b]
    ix = max(0.0, min(ax1 + aw, bx1 + bw) - max(ax1, bx1))
    iy = max(0.0, min(ay1 + ah, by1 + bh) - max(ay1, by1))
    intersection = ix * iy
    return intersection / max(aw * ah + bw * bh - intersection, 1e-12)


def read_rgbt(sequence, frame_index):
    visible = cv2.imread(sequence.frames[0][frame_index], cv2.IMREAD_COLOR)
    infrared = cv2.imread(sequence.frames[1][frame_index], cv2.IMREAD_COLOR)
    if visible is None or infrared is None:
        raise RuntimeError('Could not decode RGB/TIR frame %s:%d' %
                           (sequence.name, frame_index))
    visible = cv2.cvtColor(visible, cv2.COLOR_BGR2RGB)
    infrared = cv2.cvtColor(infrared, cv2.COLOR_BGR2RGB)
    return np.concatenate((visible, infrared), axis=2)


class Intervention:
    def __init__(self, modules):
        self.mode = 'ed'
        self.capture = False
        self.records = []
        self.handles = [module.register_forward_hook(self.hook) for module in modules]

    @staticmethod
    def message_norm(message):
        merged = message.float().transpose(1, 2).reshape(message.shape[0], message.shape[2], -1)
        return float(merged.norm(dim=-1).mean().detach().cpu())

    @staticmethod
    def scaled_message_norm(message, scale):
        scaled = message * scale[:, None, :, :]
        merged = scaled.float().transpose(1, 2).reshape(scaled.shape[0], scaled.shape[2], -1)
        return float(merged.norm(dim=-1).mean().detach().cpu())

    def hook(self, module, args, output):
        if not isinstance(output, tuple) or len(output) != 3:
            raise RuntimeError('Unexpected DualMessageModulator output contract')
        self_scale, cross_scale, diag = output
        old_self, old_cross = self_scale, cross_scale
        if self.mode in ('self_identity', 'both_identity'):
            self_scale = torch.ones_like(self_scale)
        if self.mode in ('cross_identity', 'both_identity'):
            cross_scale = torch.ones_like(cross_scale)
        elif self.mode == 'remove_cross_message':
            cross_scale = torch.zeros_like(cross_scale)
        if self.capture:
            self.records.append(dict(
                mode=self.mode,
                learned_self=dict(mean=float(old_self.float().mean().cpu()),
                                  min=float(old_self.float().min().cpu()),
                                  max=float(old_self.float().max().cpu())),
                learned_cross=dict(mean=float(old_cross.float().mean().cpu()),
                                   min=float(old_cross.float().min().cpu()),
                                   max=float(old_cross.float().max().cpu())),
                applied_self=dict(mean=float(self_scale.float().mean().cpu()),
                                  min=float(self_scale.float().min().cpu()),
                                  max=float(self_scale.float().max().cpu())),
                applied_cross=dict(mean=float(cross_scale.float().mean().cpu()),
                                   min=float(cross_scale.float().min().cpu()),
                                   max=float(cross_scale.float().max().cpu())),
                self_message_norm=self.message_norm(args[4]),
                cross_message_norm=self.message_norm(args[5]),
                applied_self_message_norm=self.scaled_message_norm(args[4], self_scale),
                applied_cross_message_norm=self.scaled_message_norm(args[5], cross_scale),
                receiver_coverage_mean=float(args[2].float().mean().cpu()),
                cross_attention_mass_mean=float(diag['cross_attention_mass'].mean().cpu())))
        return self_scale, cross_scale, diag

    def close(self):
        for handle in self.handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(P2))
    parser.add_argument('--finalize-existing', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    if args.finalize_existing:
        partial = json.loads((output / 'partial_results.json').read_text())
        expected_modes = set(MODES)
        cases = partial.get('cases', [])
        if partial.get('completed_cases') != len(CASES) or len(cases) != len(CASES):
            raise RuntimeError('Cannot finalize: incomplete case set')
        for case_result in cases:
            if not case_result.get('paired_anchor_inputs'):
                raise RuntimeError('Cannot finalize: unpaired intervention inputs')
            if {branch.get('mode') for branch in case_result.get('branches', [])} != expected_modes:
                raise RuntimeError('Cannot finalize: incomplete intervention modes')
        started = json.loads((output / 'started.json').read_text())
        completed = dict(
            status='P2-COMPLETE / FIXED-STATE-PAIRED / FINALIZED-FROM-COMPLETE-PARTIAL',
            cases=cases, checkpoint_sha256=CHECKPOINT_SHA,
            config_sha256=sha(CONFIG), elapsed_seconds=time.time() - started['start_time'],
            finalization_note='All inference slices were written and paired before the reporting-only flush error.')
        write_json(output / 'completed.json', completed)
        print(json.dumps(dict(status=completed['status'], cases=len(cases), output=str(output))), flush=True)
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Refusing nonempty output/run identity: ' + str(output))
    if sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError('ED checkpoint hash mismatch')
    output.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('high')

    from lib.models.layers.dual_message_modulator import DualMessageModulator
    from lib.test.evaluation import get_dataset, Tracker

    os.environ['TBSI_CHECKPOINT_OVERRIDE'] = str(CHECKPOINT)
    os.environ['TBSI_INFER_FP16'] = '0'
    dataset = get_dataset('lasher_test')
    dataset = {sequence.name: sequence for sequence in dataset}
    tracker_wrapper = Tracker('tbsi_track', CONFIG_NAME, 'lasher_test')
    params = tracker_wrapper.get_parameters()
    if Path(params.checkpoint).resolve() != CHECKPOINT.resolve():
        raise RuntimeError('Tracker resolved a different checkpoint')
    params.debug = 0
    tracker = tracker_wrapper.create_tracker(params)
    modules = [module for module in tracker.network.modules()
               if isinstance(module, DualMessageModulator)]
    if len(modules) != 1:
        raise RuntimeError('Expected one active first-fusion DualMessageModulator, got %d' % len(modules))
    intervention = Intervention(modules)
    original_forward = tracker.network.forward
    target_capture = {'enabled': False, 'value': None}

    def wrapped_forward(*forward_args, **kwargs):
        result = original_forward(*forward_args, **kwargs)
        if target_capture['enabled']:
            score = result['score_map']
            response = tracker.output_window * score
            flat = response.float().flatten()
            top = torch.topk(flat, k=2)
            target_capture['value'] = dict(
                template_hash=tensor_sha(kwargs['template']),
                search_hash=tensor_sha(kwargs['search']),
                score_peak_index=int(top.indices[0].cpu()),
                score_peak_value=float(top.values[0].cpu()),
                score_second_value=float(top.values[1].cpu()),
                score_peak_margin=float((top.values[0] - top.values[1]).cpu()),
                center_score_map_shape=list(score.shape),
                message_records=list(intervention.records))
        return result

    tracker.network.forward = wrapped_forward
    manifest = dict(
        id='EG-DUALMSG-V2-P2-ATTRIBUTION-RETRY4-20261028',
        checkpoint=str(CHECKPOINT), checkpoint_sha256=CHECKPOINT_SHA,
        config=str(CONFIG), config_sha256=sha(CONFIG),
        modes=MODES, cases=CASES, horizon=HORIZON,
        precision='FP32', tracker_threads=0, template_update=False,
        cudnn_deterministic=True, cudnn_benchmark=False,
        python=sys.version, pytorch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), gpu_count=torch.cuda.device_count(),
        command=sys.argv)
    manifest['start_time'] = start_time
    write_json(output / 'started.json', manifest)
    write_json(output / 'manifest.json', dict(
        checkpoint_sha256=CHECKPOINT_SHA, config_sha256=sha(CONFIG),
        screen96_manifest_sha256=sha(Path('/autodl-fs/data/early_gate_closed_loop/dual_message_p1_v2_20261027/screen96_max2000.txt')),
        preexisting_analysis_sha256=sha(ROOT / 'experiments/early_gate_closed_loop_20261026/dual_message_implementation_20261027/detailed_analysis_20261028/REPORT.md'),
        source_sha256={str(path): sha(path) for path in [Path(__file__), CONFIG,
            ROOT / 'lib/models/layers/dual_message_modulator.py',
            ROOT / 'lib/models/layers/tbsi_layer.py',
            ROOT / 'lib/test/tracker/tbsi_track.py']}))

    results = []
    try:
        for case in CASES:
            sequence = dataset[case['sequence']]
            target = case['frame']
            case_horizon = min(HORIZON, len(sequence.frames[0]) - 1 - target)
            intervention.mode = 'ed'
            intervention.capture = False
            if case_horizon < 0:
                raise RuntimeError('Target frame outside sequence: ' + case['sequence'])
            tracker.initialize(read_rgbt(sequence, 0), sequence.init_info())
            saved = np.loadtxt(RUN / 'ED/eval96' / (case['sequence'] + '.txt'), ndmin=2)
            prefix = []
            for frame_num in range(1, target):
                image = read_rgbt(sequence, frame_num)
                info = sequence.frame_info(frame_num)
                info.update(sequence_name=sequence.name, frame_num=frame_num,
                            gt_bbox=sequence.ground_truth_rect[frame_num])
                prediction = tracker.track(image, info)['target_bbox']
                prediction = np.asarray(prediction, dtype=np.float64).reshape(4)
                if not np.allclose(prediction, saved[frame_num], atol=1e-5, rtol=1e-6):
                    raise RuntimeError('ED prefix replay diverged at %s frame %d' %
                                       (case['sequence'], frame_num))
                prefix.append(prediction.tolist())

            anchor = dict(state=clone_state(tracker.state),
                          frame_id=tracker.frame_id,
                          prev_search=clone_state(tracker.prev_search),
                          prev_temporal_tokens=clone_state(tracker.prev_temporal_tokens))
            case_result = dict(case=case, observed_post_target_frames=case_horizon,
                               prefix_frames=len(prefix),
                               prefix_final_state=anchor['state'], branches=[])
            anchor_state_sha = tensor_sha(torch.as_tensor(anchor['state']))
            for mode in MODES:
                tracker.state = clone_state(anchor['state'])
                tracker.frame_id = anchor['frame_id']
                tracker.prev_search = clone_state(anchor['prev_search'])
                tracker.prev_temporal_tokens = clone_state(anchor['prev_temporal_tokens'])
                intervention.mode = mode
                intervention.capture = True
                intervention.records = []
                branch = dict(mode=mode, frames=[])
                first_input = None
                for frame_num in range(target, target + case_horizon + 1):
                    image = read_rgbt(sequence, frame_num)
                    info = sequence.frame_info(frame_num)
                    info.update(sequence_name=sequence.name, frame_num=frame_num,
                                gt_bbox=sequence.ground_truth_rect[frame_num])
                    state_before = np.asarray(tracker.state, dtype=np.float64).reshape(4).tolist()
                    state_before_sha = tensor_sha(torch.as_tensor(state_before))
                    if frame_num == target:
                        if state_before_sha != anchor_state_sha:
                            raise RuntimeError('Anchor state changed while restoring branch state')
                        target_capture['enabled'] = True
                        target_capture['value'] = None
                        intervention.records = []
                    else:
                        target_capture['enabled'] = False
                        intervention.capture = False
                    output_frame = tracker.track(image, info)
                    prediction = np.asarray(output_frame['target_bbox'], dtype=np.float64).reshape(4)
                    gt = sequence.ground_truth_rect[frame_num]
                    iou = bbox_iou(prediction, gt)
                    row = dict(frame=frame_num, state_before=state_before,
                               state_before_sha256=state_before_sha,
                               prediction=prediction.tolist(), gt=np.asarray(gt).tolist(),
                               iou=float(iou), severe_failure=bool(iou < 0.1))
                    if frame_num == target:
                        capture = target_capture['value']
                        if capture is None:
                            raise RuntimeError('Target network input/output capture missing')
                        first_input = dict(state_before_sha256=state_before_sha,
                                           template_sha256=capture['template_hash'],
                                           search_sha256=capture['search_hash'])
                        row.update(capture)
                        row['message_records'] = capture['message_records']
                        gt_cx = float(gt[0]) + 0.5 * float(gt[2])
                        gt_cy = float(gt[1]) + 0.5 * float(gt[3])
                        pred_cx = float(prediction[0]) + 0.5 * float(prediction[2])
                        pred_cy = float(prediction[1]) + 0.5 * float(prediction[3])
                        row['center_error_pixels'] = float(((pred_cx - gt_cx) ** 2 +
                                                            (pred_cy - gt_cy) ** 2) ** 0.5)
                    branch['frames'].append(row)
                intervention.capture = False
                branch['input_hashes_at_anchor'] = first_input
                branch['mean_iou'] = float(np.mean([r['iou'] for r in branch['frames']]))
                branch['last5_mean_iou'] = float(np.mean([r['iou'] for r in branch['frames'][-5:]]))
                branch['failure_frames'] = int(sum(r['severe_failure'] for r in branch['frames']))
                branch['longest_failure_run'] = 0
                current = 0
                for row in branch['frames']:
                    current = current + 1 if row['severe_failure'] else 0
                    branch['longest_failure_run'] = max(branch['longest_failure_run'], current)
                case_result['branches'].append(branch)

            anchor_hashes = {json.dumps(branch['input_hashes_at_anchor'], sort_keys=True)
                             for branch in case_result['branches']}
            case_result['paired_anchor_inputs'] = len(anchor_hashes) == 1
            if not case_result['paired_anchor_inputs']:
                raise RuntimeError('Intervention branches do not share anchor state/crop hashes')
            results.append(case_result)
            write_json(output / 'partial_results.json', dict(completed_cases=len(results), cases=results))
            print(json.dumps(dict(case=case['sequence'], frame=target,
                                  paired=case_result['paired_anchor_inputs'])), flush=True)
    except Exception as error:
        write_json(output / 'failed.json', dict(
            status='EXECUTION-FAILED / NO MECHANISM CONCLUSION',
            error_type=type(error).__name__, error=str(error),
            completed_cases=len(results), checkpoint_sha256=CHECKPOINT_SHA,
            config_sha256=sha(CONFIG), elapsed_seconds=time.time() - start_time))
        raise
    finally:
        intervention.close()

    completed = dict(status='P2-COMPLETE / FIXED-STATE-PAIRED', cases=results,
                     checkpoint_sha256=CHECKPOINT_SHA, config_sha256=sha(CONFIG),
                     elapsed_seconds=time.time() - start_time)
    write_json(output / 'completed.json', completed)
    print(json.dumps(dict(status=completed['status'], cases=len(results), output=str(output))), flush=True)


if __name__ == '__main__':
    main()
