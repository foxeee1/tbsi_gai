#!/usr/bin/env python3
"""Freeze the G0-Expand external train-sequence validation manifest."""
import argparse, hashlib, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def event_frames(seq_len):
    if seq_len < 3: return []
    return sorted(set(round(x) for x in [1, seq_len - 1]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sequence-list', default=str(ROOT / 'experiments/proxy/lasher_rpp_v1/proxy_train_sequences.txt'))
    ap.add_argument('--old-export', default=str(ROOT / 'output/experiments/g0_information_source_20261004/export'))
    ap.add_argument('--old-probe', default=str(ROOT / 'output/experiments/g0_information_source_20261004/probe/probe_metrics.json'))
    ap.add_argument('--output', required=True)
    ap.add_argument('--count', type=int, default=24)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.count < 20: raise ValueError('G0-Expand requires at least 20 fresh sequences')
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()): raise RuntimeError('refusing non-empty output {}'.format(out))
    out.mkdir(parents=True, exist_ok=True)
    sequence_list = Path(args.sequence_list).resolve()
    all_sequences = [x.strip() for x in sequence_list.read_text().splitlines() if x.strip()]
    old_selected = {x.strip() for x in (Path(args.old_export) / 'selected_sequences.txt').read_text().splitlines() if x.strip()}
    old = json.loads(Path(args.old_probe).read_text())
    old_train = set(old['train_sequences'])
    old_valid = set(old['valid_sequences'])
    if not old_train or not old_valid: raise RuntimeError('old G0 split missing')
    if not old_train.isdisjoint(old_valid): raise RuntimeError('old split overlaps')
    if not old_train.issubset(old_selected | set(all_sequences)): raise RuntimeError('old train split invalid')
    excluded = old_train | old_valid
    candidates = [x for x in all_sequences if x not in excluded]
    if len(candidates) < args.count: raise RuntimeError('insufficient fresh sequences')
    rng = random.Random(args.seed)
    candidates = sorted(candidates)
    rng.shuffle(candidates)
    selected = sorted(candidates[:args.count])
    if set(selected) & excluded: raise RuntimeError('fresh sequence overlap')
    rows = [{'sequence': s, 'event_index': i, 'event_rule': 'frame 1 and last frame'}
            for i, s in enumerate(selected)]
    (out / 'fresh_sequences.txt').write_text('\n'.join(selected) + '\n')
    (out / 'manifest.json').write_text(json.dumps({
        'protocol': 'G0-Expand', 'seed': args.seed, 'count': len(selected),
        'sequence_list': str(sequence_list), 'sequence_list_sha256': sha256(sequence_list),
        'old_train_sequences': sorted(old_train), 'old_validation_sequences': sorted(old_valid),
        'fresh_sequences': selected, 'excluded_sequences': sorted(excluded),
        'events_per_sequence': 2, 'event_rule': 'frame 1 and last frame; sequence IDs frozen before export',
        'conditions': ['L_zero','L_E','L_Lcopy','L_E_spatial','L_E_event','XY_only'],
        'probe_seeds': [42, 3407], 'min_delta_ap': 0.03}, indent=2) + '\n')
    print(json.dumps({'fresh_sequences': len(selected), 'old_train': len(old_train),
                      'old_valid': len(old_valid), 'candidate_pool': len(candidates)}, indent=2))

if __name__ == '__main__': main()
