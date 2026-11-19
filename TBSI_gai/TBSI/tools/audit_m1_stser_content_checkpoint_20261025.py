"""Read checkpoint optimizer progress without changing model or result files."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyze_m1_stser_content_pair_20261025 import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-epoch', type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(str(args.output))
    checkpoint = torch.load(str(args.checkpoint), map_location='cpu')
    if checkpoint.get('epoch') != args.expected_epoch:
        raise ValueError('checkpoint epoch does not match requested schedule')
    optimizer = checkpoint.get('optimizer', {})
    steps = []
    for state in optimizer.get('state', {}).values():
        value = state.get('step')
        if value is not None:
            steps.append(int(value.item() if isinstance(value, torch.Tensor) else value))
    result = {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha(args.checkpoint),
              'epoch': checkpoint['epoch'], 'optimizer_type': 'AdamW (from locked config)',
              'optimizer_state_count': len(optimizer.get('state', {})),
              'optimizer_step_min': min(steps) if steps else None,
              'optimizer_step_max': max(steps) if steps else None,
              'optimizer_step_unique': sorted(set(steps)),
              'nominal_steps_from_config': 237 if args.expected_epoch == 3 else 7020,
              'note': 'actual optimizer steps are per-parameter AdamW state; AMP skips may lower them'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
