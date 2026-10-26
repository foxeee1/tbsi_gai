#!/usr/bin/env python3
"""Overlay only trained target-margin tensors onto the trained early-gate parent."""
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PARENT = ROOT / 'output/experiments/M1_EARLY_CONDITIONED_MSHA_GATE_PROXY_20261021/checkpoints/TBSITrack_ep0003.pth.tar'
CANDIDATE = ROOT / 'output/experiments/M1_EARLY_GATE_TARGET_MARGIN_QUICK3_20261022/checkpoints/TBSITrack_ep0003.pth.tar'
OUT = ROOT / 'output/experiments/M1_EARLY_GATE_MARGIN_OPERATOR_20261022/checkpoints/early_gate_parent_plus_margin_ep0003.pth.tar'
META = OUT.parent.parent / 'checkpoint_merge_provenance.json'


def state_of(checkpoint):
    if 'net' in checkpoint:
        return checkpoint['net']
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    return checkpoint


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parent, candidate = torch.load(str(PARENT), map_location='cpu'), torch.load(str(CANDIDATE), map_location='cpu')
    parent_state, candidate_state = state_of(parent), state_of(candidate)
    keys = [key for key in candidate_state if '.early_margin_gate.' in key]
    if len(keys) != 15 or set(parent_state).difference(candidate_state):
        raise RuntimeError('checkpoint tensor contract failed: margin keys={}, missing parent={}'.format(
            len(keys), len(set(parent_state).difference(candidate_state))))
    changed_shared = [key for key in parent_state
                      if not torch.equal(parent_state[key], candidate_state[key])]
    non_bn = [key for key in changed_shared
              if not (key.endswith('.running_mean') or key.endswith('.running_var') or
                      key.endswith('.num_batches_tracked'))]
    if non_bn:
        raise RuntimeError('unexpected trained shared state changes: {}'.format(non_bn))

    merged = candidate
    merged_state = state_of(merged)
    for key, value in parent_state.items():
        merged_state[key] = value
    for key in keys:
        merged_state[key] = candidate_state[key]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, str(OUT))
    final = state_of(torch.load(str(OUT), map_location='cpu'))
    if any(not torch.equal(value, final[key]) for key, value in parent_state.items()):
        raise RuntimeError('merged parent entries differ from trained early-gate checkpoint')
    meta = {
        'parent_checkpoint': str(PARENT.relative_to(ROOT)), 'parent_sha256': sha256(PARENT),
        'candidate_checkpoint': str(CANDIDATE.relative_to(ROOT)), 'candidate_sha256': sha256(CANDIDATE),
        'merged_checkpoint': str(OUT.relative_to(ROOT)), 'merged_sha256': sha256(OUT),
        'parent_state_keys': len(parent_state), 'margin_tensor_count': len(keys),
        'restored_bn_buffer_count': len(changed_shared), 'non_margin_non_bn_changes': 0,
        'margin_parameter_l2': {key: float(final[key].float().norm()) for key in keys},
    }
    META.write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
