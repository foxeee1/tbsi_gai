"""Prepare one SOT-initialized state shared by matched STSER content arms."""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.config.tbsi_track.config import cfg, update_config_from_file
from lib.models.tbsi_track import build_tbsi_track


PREFIX = 'M1_STSER_CONTENT'
SOT = ROOT / 'pretrained_models/TBSITrack_SOT_Pretrained.pth.tar'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=('quick3', 'full15'), required=True)
    args = parser.parse_args()
    scope = args.scope.upper()
    control = '{}_{}_CONTROL_20261025'.format(PREFIX, scope)
    candidate = '{}_{}_CANDIDATE_20261025'.format(PREFIX, scope)
    config_path = ROOT / 'experiments/tbsi_track/{}.yaml'.format(control)
    other_path = ROOT / 'experiments/tbsi_track/{}.yaml'.format(candidate)
    out = ROOT / 'experiments/snapshots/m1_stser_content_pair_{}_20261025'.format(args.scope)
    for path in (config_path, other_path, SOT):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    if out.exists():
        raise FileExistsError('paired initialization directory already exists: {}'.format(out))

    update_config_from_file(str(config_path))
    declared = (ROOT / str(cfg.MODEL.INIT_CHECKPOINT)).resolve()
    if (declared != (out / 'sot_initialized_shared_net.pth.tar').resolve() or
            cfg.MODEL.BACKBONE.SEARCH_CONDITIONED_T2S_MODE != 'stser_competition_content'):
        raise ValueError('preparation requires the content host and scoped paired init path')
    if float(cfg.MODEL.BACKBONE.SEARCH_CONDITIONED_T2S_CONTENT_STRENGTH) != 0:
        raise ValueError('prepare from the control architecture at zero correction strength')
    import yaml
    with other_path.open() as handle:
        other = yaml.safe_load(handle)
    with config_path.open() as handle:
        parent = yaml.safe_load(handle)
    other_strength = other['MODEL']['BACKBONE']['SEARCH_CONDITIONED_T2S_CONTENT_STRENGTH']
    other['MODEL']['BACKBONE']['SEARCH_CONDITIONED_T2S_CONTENT_STRENGTH'] = 0.0
    if other != parent or float(other_strength) != 1.0:
        raise ValueError('the two configurations must differ only in content strength 0 versus 1')
    if cfg.MODEL.PRETRAIN_FILE != SOT.name or cfg.TRAIN.MATCHED_SEED != 42:
        raise ValueError('unexpected pretraining or seed contract')

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    # Build with the same architecture but let the standard SOT loading branch
    # initialize it before the paired checkpoint exists.
    init_path = cfg.MODEL.INIT_CHECKPOINT
    cfg.MODEL.INIT_CHECKPOINT = ''
    try:
        model = build_tbsi_track(cfg, training=True)
    finally:
        cfg.MODEL.INIT_CHECKPOINT = init_path
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    content_keys = [key for key in state if key.endswith('search_conditioned_t2s.content_logit')]
    if len(content_keys) != 1:
        raise RuntimeError('content host was not built exactly once')
    old_branch = [key for key in state if '.search_conditioned_t2s.' in key]
    native = [key for key in state if '.v2_template_fusion.' in key]
    if not old_branch or not native:
        raise RuntimeError('missing STSER or native MSHA state')
    out.mkdir(parents=True)
    path = out / 'sot_initialized_shared_net.pth.tar'
    torch.save({'net': state, 'origin': SOT.name, 'seed': 42,
                'scope': args.scope, 'kind': 'common_initialized_weights_no_optimizer'}, str(path))
    with (out / 'state_tensor_sha256.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('name', 'shape', 'dtype', 'sha256'))
        for key, value in sorted(state.items()):
            raw = value.contiguous().numpy().tobytes()
            writer.writerow((key, list(value.shape), str(value.dtype), hashlib.sha256(raw).hexdigest()))
    metadata = {
        'scope': args.scope,
        'seed': 42,
        'origin': str(SOT),
        'origin_sha256': sha(SOT),
        'shared_initial_checkpoint': str(path),
        'shared_initial_sha256': sha(path),
        'configs': {control: sha(config_path), candidate: sha(other_path)},
        'content_parameter_names': content_keys,
        'native_fusion_state_count': len(native),
        'stser_state_count': len(old_branch),
        'state_tensor_count': len(state),
        'python': os.sys.version,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'execution': 'model construction and CPU checkpoint preparation only; no forward or optimizer',
    }
    (out / 'initialization_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
