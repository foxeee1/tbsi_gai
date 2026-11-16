#!/usr/bin/env python3
"""Prepare a pre-registered 48-sequence evaluation cohort from existing labels."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'output/experiments/MSHA-MR-CONTENT-PROJECTION-QUICK-S42-20261114'
SOURCE = ROOT / 'output/experiments/MSHA-B0-U936-S43-FULL15-FULL244-20261112/analysis/two_seed_failure_20261113/per_sequence_two_seed.csv'
EXISTING = RUN / 'analysis/per_sequence.csv'
OUT = RUN / 'contracts'
ATTRS = ['NO','PO','TO','HO','MB','LI','HI','AIV','LR','DEF','BC','SA','CM','TC','FL','OV','FM','SV','ARV']
GROUPS = ['consistent_mr_gain','consistent_mr_regression','seed_direction_flip','one_seed_tie']

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    old = list(csv.DictReader(EXISTING.open()))
    old_names = [r['sequence'] for r in old]
    if len(old_names) != 24 or len(set(old_names)) != 24:
        raise RuntimeError('Expected the existing fixed 24 sequence cohort')
    rows = list(csv.DictReader(SOURCE.open()))
    by_name = {r['sequence']: r for r in rows}
    if set(old_names) - set(by_name):
        raise RuntimeError('Existing cohort is absent from the source attribute table')

    # Select six unseen members per historical outcome stratum. The greedy score
    # rewards unseen attributes, length-bin coverage, and broad sequence lengths.
    selected = []
    covered = {a: 0 for a in ATTRS}
    lengths = [int(by_name[n]['frames']) for n in old_names]
    bins = [min(4, int(5 * (x - min(lengths)) / max(1, max(lengths)-min(lengths)))) for x in lengths]
    covered_bins = set(bins)
    for group in GROUPS:
        candidates = [r for r in rows if r['cross_seed_group'] == group and r['sequence'] not in old_names]
        if not candidates:
            continue
        target = 6
        for _ in range(min(target, len(candidates))):
            def score(r):
                attrs = sum(int(r['attr_'+a]) for a in ATTRS)
                new_attrs = sum(int(r['attr_'+a]) and not covered[a] for a in ATTRS)
                length = int(r['frames'])
                b = min(4, int(5 * (length - min(lengths)) / max(1, max(lengths)-min(lengths))))
                new_bin = int(b not in covered_bins)
                # Attribute novelty dominates; length spread breaks ties.
                return (4*new_attrs + new_bin + 0.01*attrs, -abs(length - sum(lengths)/len(lengths)))
            best = max(candidates, key=score)
            selected.append(best)
            candidates.remove(best)
            for a in ATTRS:
                covered[a] += int(best['attr_'+a])
            length = int(best['frames'])
            covered_bins.add(min(4, int(5 * (length - min(lengths)) / max(1, max(lengths)-min(lengths)))))

    names = old_names + [r['sequence'] for r in selected]
    if len(names) != 48 or len(set(names)) != 48:
        raise RuntimeError('Cohort selection did not produce 48 unique sequences')
    manifest = OUT / 'eval_content_projection_diverse48_20261114.txt'
    manifest.write_text('\n'.join(names)+'\n')
    metadata = {
        'run': RUN.name, 'selection_type': 'pre-registered diverse expansion',
        'sequence_count': len(names), 'existing_count': len(old_names), 'added_count': len(selected),
        'source_csv': str(SOURCE), 'source_sha256': sha(SOURCE),
        'existing_eval_csv_sha256': sha(EXISTING), 'manifest_sha256': sha(manifest),
        'groups': {g: sum(by_name[n]['cross_seed_group'] == g for n in names) for g in GROUPS},
        'added_by_group': {g: [r['sequence'] for r in selected if r['cross_seed_group'] == g] for g in GROUPS},
        'attribute_counts': {a: sum(int(by_name[n]['attr_'+a]) for n in names) for a in ATTRS},
        'frame_range': [min(int(by_name[n]['frames']) for n in names), max(int(by_name[n]['frames']) for n in names)],
        'selection_rule': 'Existing 24 retained; six unseen sequences per four historical strata, greedy coverage of official attributes and frame-length bins; no candidate metrics used.',
        'limitations': 'Historical strata and attributes are descriptive, overlapping, and sourced from prior MR/B0 analysis; this is not an independent holdout.'
    }
    (OUT / 'eval_content_projection_diverse48_20261114.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
