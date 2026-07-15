"""
MiniLasHeR test dataset: Subset of LasHeR for rapid evaluation.
Uses MINI_TEST_SEQUENCES (30 sequences from the LasHeR test set)
covering Normal/Illumination/ThermalCross/Occlusion/FastMotion.
"""

import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
from lib.train.dataset.mini_lasher import MINI_TEST_SEQUENCES
import os


class MiniLasHeRDataset(BaseDataset):
    """ MiniLasHeR evaluation dataset — 30 test sequences, 5 categories. """

    def __init__(self, split='testingset'):
        super().__init__()
        self.base_path = os.path.join(self.env_settings.lasher_path, split)
        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/init.txt'.format(self.base_path, sequence_name)
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path_i = '{}/{}/infrared'.format(self.base_path, sequence_name)
        frames_path_v = '{}/{}/visible'.format(self.base_path, sequence_name)
        frame_list_i = [frame for frame in os.listdir(frames_path_i) if frame.endswith(".jpg")]
        frame_list_i.sort(key=lambda f: int(f[1:-4]))
        frame_list_v = [frame for frame in os.listdir(frames_path_v) if frame.endswith(".jpg")]
        frame_list_v.sort(key=lambda f: int(f[1:-4]))
        frames_list_i = [os.path.join(frames_path_i, frame) for frame in frame_list_i]
        frames_list_v = [os.path.join(frames_path_v, frame) for frame in frame_list_v]
        frames_list = [frames_list_v, frames_list_i]
        return Sequence(sequence_name, frames_list, 'mini_lasher', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        # Read the full testing set list file
        list_path = os.path.join(os.path.dirname(self.base_path), 'testingsetList.txt')
        with open(list_path) as f:
            all_seqs = f.read().splitlines()

        # Filter to only MINI_TEST_SEQUENCES that exist
        available = set(all_seqs)
        filtered = [s for s in MINI_TEST_SEQUENCES if s in available]
        missing = [s for s in MINI_TEST_SEQUENCES if s not in available]
        if missing:
            print(f"[MiniLasHeR Test] WARNING: {len(missing)} sequence(s) not "
                  f"found in test set: {missing}")

        return filtered
