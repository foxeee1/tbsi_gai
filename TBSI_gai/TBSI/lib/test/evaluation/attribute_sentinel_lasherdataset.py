import os
from pathlib import Path

import numpy as np

from lib.test.evaluation.data import BaseDataset, Sequence, SequenceList
from lib.test.evaluation.lasherdataset import SKIP_SEQUENCES
from lib.test.utils.load_text import load_text


class AttributeSentinelLasHeRDataset(BaseDataset):
    """Fixed LasHeR attribute-sentinel subset for fast Stage-1 validation."""

    def __init__(self, split="testingset"):
        super().__init__()
        self.base_path = os.path.join(self.env_settings.lasher_path, split)
        self.split = split
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = "{}/{}/init.txt".format(self.base_path, sequence_name)
        ground_truth_rect = load_text(str(anno_path), delimiter=",", dtype=np.float64)

        frames_path_i = "{}/{}/infrared".format(self.base_path, sequence_name)
        frames_path_v = "{}/{}/visible".format(self.base_path, sequence_name)
        frame_list_i = [frame for frame in os.listdir(frames_path_i) if frame.endswith(".jpg")]
        frame_list_i.sort(key=lambda f: int(f[1:-4]))
        frame_list_v = [frame for frame in os.listdir(frames_path_v) if frame.endswith(".jpg")]
        frame_list_v.sort(key=lambda f: int(f[1:-4]))
        frames_list_i = [os.path.join(frames_path_i, frame) for frame in frame_list_i]
        frames_list_v = [os.path.join(frames_path_v, frame) for frame in frame_list_v]
        return Sequence(sequence_name, [frames_list_v, frames_list_i],
                        "lasher_attribute_sentinel", ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        root = Path(__file__).resolve().parents[3]
        list_path = root / "experiments" / "tbsi_track" / "attribute_sentinel_sequences.txt"
        if not os.path.isfile(list_path):
            raise FileNotFoundError(
                "Missing attribute sentinel list: {}".format(list_path))
        with open(list_path, "r") as f:
            sequence_list = [line.strip() for line in f if line.strip()]

        available = set(os.listdir(self.base_path))
        filtered = [s for s in sequence_list if s in available and s not in SKIP_SEQUENCES]
        missing = [s for s in sequence_list if s not in available]
        skipped = [s for s in sequence_list if s in SKIP_SEQUENCES]
        if missing:
            print("[AttributeSentinelLasHeR] WARNING: missing sequences: {}".format(missing))
        if skipped:
            print("[AttributeSentinelLasHeR] WARNING: skipped sequences: {}".format(skipped))
        return filtered
