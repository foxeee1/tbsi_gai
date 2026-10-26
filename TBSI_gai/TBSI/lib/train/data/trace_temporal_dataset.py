"""Dataset for frozen-M1 causal temporal actor audits."""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class TraceTemporalDataset(Dataset):
    def __init__(self, trace_root, sequence_ids=None):
        self.trace_root = os.path.abspath(trace_root)
        allowed = set(sequence_ids) if sequence_ids else None
        self.rows = []
        for sequence_id in sorted(os.listdir(self.trace_root)):
            if allowed is not None and sequence_id not in allowed:
                continue
            path = os.path.join(self.trace_root, sequence_id, 'frames.jsonl')
            if not os.path.isfile(path):
                continue
            rows = [json.loads(line) for line in open(path) if line.strip()]
            for index, row in enumerate(rows):
                if index == 0 or not row.get('has_history') or not row.get('gt_valid'):
                    continue
                previous = rows[index - 1]
                if (row.get('sequence_id') != sequence_id or
                        row.get('history_sequence_id') != sequence_id or
                        int(row['history_frame_id']) != int(previous['frame_id']) or
                        int(row['history_frame_id']) >= int(row['frame_id'])):
                    raise ValueError('invalid causal trace row: %s frame %s' %
                                     (sequence_id, row.get('frame_id')))
                if row.get('history_crop_xywh') != previous.get('current_crop_xywh'):
                    raise ValueError('history crop does not link to preceding trace row')
                if not row.get('raw_head_file') or not previous.get('raw_head_file'):
                    raise ValueError('trace row is missing raw head token files')
                self.rows.append((sequence_id, row, previous))

        if not self.rows:
            raise ValueError('no valid trace-backed temporal rows found')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        sequence_id, row, previous = self.rows[index]
        seq_root = os.path.join(self.trace_root, sequence_id)
        current = np.load(os.path.join(seq_root, row['raw_head_file']), allow_pickle=False)
        history = np.load(os.path.join(seq_root, previous['raw_head_file']), allow_pickle=False)
        current = np.asarray(current, dtype=np.float32)
        history = np.asarray(history, dtype=np.float32)
        if current.shape != (1, 256, 768) or history.shape != (1, 256, 768):
            raise ValueError('raw tokens must have shape [1,256,768]')
        if not np.isfinite(current).all() or not np.isfinite(history).all():
            raise ValueError('raw tokens contain non-finite values')
        return {
            'sequence_id': sequence_id,
            'history_sequence_id': sequence_id,
            'frame_id': int(row['frame_id']),
            'history_frame_id': int(previous['frame_id']),
            'current_raw': torch.from_numpy(current[0].copy()),
            'history_raw': torch.from_numpy(history[0].copy()),
            'current_crop_xywh': torch.tensor(row['current_crop_xywh'], dtype=torch.float32),
            'history_crop_xywh': torch.tensor(previous['current_crop_xywh'], dtype=torch.float32),
            'current_crop_transform': torch.tensor(row['current_crop_transform'], dtype=torch.float32),
            'gt_box_image': torch.tensor(row['gt_box'], dtype=torch.float32),
            'gt_valid': bool(row['gt_valid']),
        }
