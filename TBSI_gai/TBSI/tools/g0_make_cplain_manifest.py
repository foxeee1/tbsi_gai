#!/usr/bin/env python3
"""Freeze a Cplain training list disjoint from the G0-Expand validation list."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_list(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proxy-train', required=True)
    parser.add_argument('--fresh', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    proxy = read_list(args.proxy_train)
    fresh = set(read_list(args.fresh))
    train = [name for name in proxy if name not in fresh]
    if not train or set(train) & fresh:
        raise RuntimeError('Cplain train/validation sequence overlap')
    train_path = output / 'cplain_train_sequences.txt'
    fresh_path = output / 'cplain_eval_sequences.txt'
    train_path.write_text('\n'.join(train) + '\n')
    fresh_path.write_text('\n'.join(read_list(args.fresh)) + '\n')
    manifest = {
        'protocol': 'G0-Cplain-short-closed-loop',
        'parent': 'G0-EXPAND-CONDITIONAL-PASS-20261004',
        'train_sequences': len(train),
        'eval_sequences': len(fresh),
        'train_list': str(train_path),
        'eval_list': str(fresh_path),
        'train_list_sha256': sha256(train_path),
        'eval_list_sha256': sha256(fresh_path),
        'disjoint': True,
        'epochs': 1,
        'seed': 42,
        'adapter': 'EarlySearchResidual; block2 -> location9 search only; RGB/TIR separate; zero scalar gates',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
