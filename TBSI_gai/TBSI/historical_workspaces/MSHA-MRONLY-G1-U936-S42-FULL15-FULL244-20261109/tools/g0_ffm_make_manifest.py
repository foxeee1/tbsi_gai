#!/usr/bin/env python3
"""Freeze the one-time FFM train/development sequence split."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def digest(items):
    return hashlib.sha256(('\n'.join(items) + '\n').encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hard-events', required=True)
    ap.add_argument('--all-sequences', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--dev-count', type=int, default=24)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    hard = [json.loads(line) for line in Path(args.hard_events).read_text().splitlines()]
    rescue = sorted({row['sequence'] for row in hard if 'rescue_failure' in row['groups']})
    if len(rescue) < args.dev_count:
        raise RuntimeError('not enough rescue sequences')
    rng = random.Random(args.seed)
    dev = sorted(rng.sample(rescue, args.dev_count))
    all_sequences = [line.strip() for line in Path(args.all_sequences).read_text().splitlines() if line.strip()]
    train = [name for name in all_sequences if name not in set(dev)]
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('refusing non-empty manifest directory')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ffm_dev_rescue_sequences.txt').write_text('\n'.join(dev) + '\n')
    (out / 'ffm_train_sequences.txt').write_text('\n'.join(train) + '\n')
    manifest = {'seed': args.seed, 'dev_count': len(dev), 'train_count': len(train),
                'rescue_candidate_count': len(rescue), 'dev_sequences': dev,
                'train_sequences': train, 'dev_sha256': digest(dev), 'train_sha256': digest(train),
                'selection': 'sorted rescue sequence names; Python Random(seed=42).sample; train excludes dev'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'dev_count': len(dev), 'train_count': len(train),
                      'dev_sha256': manifest['dev_sha256'], 'train_sha256': manifest['train_sha256']}, indent=2))


if __name__ == '__main__':
    main()
