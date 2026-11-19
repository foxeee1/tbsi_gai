"""Explicit opt-in, index-addressed sampling for the EVDR P1 paired experiment.

The native sampler and all historical configurations remain untouched. The
manifest covers every eligible official training sequence before repeats.
"""
import hashlib
import json
import random

import numpy as np
import torch

from .sampler import TrackingSampler
from .loader import ltr_collate_stack1


class PairedStratifiedSampler(TrackingSampler):
    def __init__(self, manifest, seed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.manifest = manifest
        self.seed = int(seed)
        self.epoch = 0
        self.offset = 0
        self.names = self.datasets[0].sequence_list
        if self.names != [r['sequence'] for r in manifest['train_sequences']]:
            raise ValueError('Manifest and actual dataset sequence order differ')
        self.order = manifest['train_order']
        self.position = {n: i for i, n in enumerate(self.names)}
        self._calls = 0
        self._frame_record = None

    def sample_seq_from_dataset(self, dataset, is_video_dataset):
        self._calls += 1
        if self._calls > 16:
            raise RuntimeError('16 invalid processing attempts for fixed sequence: ' + self._selected)
        seq_id = self.position[self._selected]
        info = dataset.get_sequence_info(seq_id)
        if len(info['visible']) < 20:
            raise ValueError('Prevalidated sequence is too short: ' + self._selected)
        return seq_id, info['visible'], info

    def __getitem__(self, index):
        global_index = self.offset + int(index)
        self._selected = self.order[global_index % len(self.order)]
        self._calls = 0
        sample_seed = int.from_bytes(hashlib.sha256(
            ('EVDR-P1-input:%d:%d' % (self.seed, global_index)).encode()).digest()[:4], 'big')
        # CPU-only worker RNG; no forked CUDA RNG access.
        random.seed(sample_seed)
        np.random.seed(sample_seed)
        torch.random.default_generator.manual_seed(sample_seed)
        ds = self.datasets[0]
        original = ds.get_frames
        frames = []

        def record_frames(seq_id, frame_ids, *args, **kwargs):
            frames.append([int(f) for f in frame_ids])
            return original(seq_id, frame_ids, *args, **kwargs)

        ds.get_frames = record_frames
        try:
            result = super().getitem(global_index)
        finally:
            ds.get_frames = original
        # Meta is separated before native collation, never passed to the model.
        result['_pair_meta'] = {'index': global_index, 'sequence': self._selected,
                                'sample_seed': sample_seed, 'attempts': self._calls,
                                'template_frames': frames[-2], 'search_frames': frames[-1]}
        return result


def paired_collate(samples):
    metadata = [sample.pop('_pair_meta') for sample in samples]
    return ltr_collate_stack1(samples), metadata
