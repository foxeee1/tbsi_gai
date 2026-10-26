#!/usr/bin/env python3
"""Overlay trained early-gate tensors onto the protected M1 state."""
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PARENT = ROOT / 'output/experiments/TBSI_E_MHSA_FULL_4090D_seed42/checkpoints/TBSITrack_ep0015.pth.tar'
CANDIDATE = ROOT / 'output/experiments/M1_EARLY_CONDITIONED_MSHA_GATE_PROXY_20261021/checkpoints/TBSITrack_ep0003.pth.tar'
OUT = ROOT / 'output/experiments/M1_EARLY_GATE_STRENGTH_REPLAY_20261022/checkpoints/M1_parent_state_plus_gate_ep0003.pth.tar'
PROVENANCE = OUT.parent.parent / 'checkpoint_merge_provenance.json'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def state_of(checkpoint):
    if 'net' in checkpoint:
        return checkpoint['net']
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    return checkpoint


def main():
    parent = torch.load(str(PARENT), map_location='cpu')
    candidate = torch.load(str(CANDIDATE), map_location='cpu')
    parent_state, candidate_state = state_of(parent), state_of(candidate)
    gate_keys = [key for key in candidate_state
                 if '.early_gate.' in key or key.endswith('.early_gate_scale')]
    if len(gate_keys) != 15:
        raise RuntimeError('expected 15 learned gate tensors, found {}'.format(len(gate_keys)))
    if set(parent_state).difference(candidate_state):
        raise RuntimeError('candidate is missing parent state keys')
    changed_shared = [key for key in parent_state
                      if not torch.equal(parent_state[key], candidate_state[key])]
    allowed_buffers = [key for key in changed_shared
                       if key.endswith('.running_mean') or key.endswith('.running_var') or
                       key.endswith('.num_batches_tracked')]
    if len(changed_shared) != 39 or set(changed_shared) != set(allowed_buffers):
        raise RuntimeError('unexpected shared-state changes: {}'.format(changed_shared))

    merged = candidate
    merged_state = state_of(merged)
    for key, value in parent_state.items():
        merged_state[key] = value
    # Keep the trained gate tensors after restoring every shared M1 tensor/buffer.
    for key in gate_keys:
        merged_state[key] = candidate_state[key]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, str(OUT))

    reloaded = torch.load(str(OUT), map_location='cpu')
    final_state = state_of(reloaded)
    mismatches = [key for key, value in parent_state.items()
                  if not torch.equal(value, final_state[key])]
    if mismatches:
        raise RuntimeError('merged checkpoint does not preserve M1 state: {}'.format(mismatches))
    record = {
        'parent_checkpoint': str(PARENT.relative_to(ROOT)),
        'parent_sha256': sha256(PARENT),
        'candidate_checkpoint': str(CANDIDATE.relative_to(ROOT)),
        'candidate_sha256': sha256(CANDIDATE),
        'merged_checkpoint': str(OUT.relative_to(ROOT)),
        'merged_sha256': sha256(OUT),
        'parent_state_keys': len(parent_state),
        'gate_tensor_count': len(gate_keys),
        'gate_keys': gate_keys,
        'shared_bn_buffers_restored': changed_shared,
        'non_gate_candidate_shared_changes': 0,
        'lambda_zero_contract': 'all 267 shared parent state entries exactly match protected M1; gate branch bypasses before message computation',
    }
    PROVENANCE.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
