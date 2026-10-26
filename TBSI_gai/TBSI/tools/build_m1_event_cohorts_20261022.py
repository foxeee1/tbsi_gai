#!/usr/bin/env python3
"""Freeze candidate-disjoint M1 event-mining cohorts from LasHeR train."""
import hashlib
import json
from pathlib import Path

import build_m1_screening_protocol_20261022 as protocol

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/protocols/m1_screening_v1_20261022'
TRAIN_SPLITS = (('calibration', 48), ('development', 24), ('locked_holdout', 24))
FEATURES = ('relative_area_median', 'relative_motion_p90',
            'rgb_target_luma_median', 'tir_target_ring_contrast_median')


def main():
    output_names = set()
    for name, _ in TRAIN_SPLITS:
        output_names.update((name + '_sequences.txt', name + '_manifest.json'))
    output_names.add('event_mining_cohorts_manifest.json')
    if any((OUT / name).exists() for name in output_names):
        raise RuntimeError('event cohort assets already exist; refusing to overwrite')

    official_train = ROOT.parent / 'data/lasher/trainingset'
    rows = []
    for path in sorted(official_train.iterdir()):
        if path.is_dir() and path.name not in protocol.SKIP_SEQUENCES and \
                (path / 'visible').is_dir() and (path / 'infrared').is_dir() and (path / 'init.txt').is_file():
            row = protocol.all_length_sequence_stats(path)
            if row:
                rows.append(row)
    attr_names, attr_map = protocol.load_attributes({row['name'] for row in rows})
    rows = [row for row in rows if row['name'] in attr_map]
    for row in rows:
        row['attributes'] = attr_map[row['name']]
    if len(rows) != 977:
        raise RuntimeError('expected 977 official train sequences; found {}'.format(len(rows)))

    train_names = set((OUT / 'quick_train_sequences.txt').read_text().splitlines())
    remaining = [row for row in rows if row['name'] not in train_names]
    if len(train_names) != 176 or len(remaining) != 801:
        raise RuntimeError('quick-train disjointness/count mismatch: train={}, remaining={}'.format(
            len(train_names), len(remaining)))

    output_rows = {}
    for index, (split_name, count) in enumerate(TRAIN_SPLITS):
        selected, edges, objective = protocol.select_proportional(
            remaining, attr_names, FEATURES, count, seed=protocol.SEED + index + 1)
        remaining_names = {row['name'] for row in selected}
        remaining = [row for row in remaining if row['name'] not in remaining_names]
        sequence_file = OUT / (split_name + '_sequences.txt')
        sequence_file.write_text('\n'.join(row['name'] for row in selected) + '\n')
        clean_rows = []
        for row in selected:
            clean = {key: value for key, value in row.items()
                     if key not in ('attributes', '_balance_vector', '_tie')}
            clean['attributes'] = row['attributes']
            clean_rows.append(clean)
        manifest = {
            'id': 'M1-EVENT-MINING-{}-20261022'.format(split_name.upper()),
            'split': 'LasHeR official trainingset', 'dataset_root': str(official_train),
            'attribute_archive_sha256': protocol.sha256(protocol.ATTRIBUTE_ZIP),
            'source_population': 977,
            'protected_parent_config_sha256': 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1',
            'protected_parent_checkpoint_sha256': '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef',
            'candidate_training_manifest': 'quick_train_manifest.json',
            'candidate_train_sequence_count': 176, 'candidate_train_overlap': 0,
            'sequence_count': count, 'sampling_seed': protocol.SEED + index + 1,
            'selection_rule': 'Outcome-blind fixed-count hash initialization and swap descent matching official overlapping attribute and scene-proxy marginals; selected after reserving the 176 candidate-training sequences.',
            'proxy_quantile_edges': edges, 'balance_objective': objective,
            'sequence_list': str(sequence_file.relative_to(ROOT)),
            'sequence_list_sha256': protocol.sha256(sequence_file), 'sequences': clean_rows,
            'usage': 'Freeze sequence membership before running baseline event mining. Use one calibration event per sequence; do not move sequences across splits after inspecting outcomes.',
            'parent_exposure_limit': 'Protected M1 was trained on all 977 LasHeR train sequences. These cohorts are held out from candidate updates only; they are not unseen to the parent model.'
        }
        protocol.write_json(OUT / (split_name + '_manifest.json'), manifest)
        output_rows[split_name] = {'count': count, 'sequence_list_sha256': protocol.sha256(sequence_file)}

    protocol.write_json(OUT / 'event_mining_cohorts_manifest.json', {
        'id': 'M1-EVENT-MINING-COHORTS-V1-20261022',
        'parent': 'Protected M1 TBSI_E_MHSA_FULL_4090D_seed42 epoch15',
        'candidate_train_manifest': 'quick_train_manifest.json',
        'candidate_train_count': 176,
        'candidate_evaluation_cohorts': output_rows,
        'total_candidate_heldout_sequences': sum(size for _, size in TRAIN_SPLITS),
        'official_train_population': 977,
        'sequence_overlap_train_vs_cohorts': 0,
        'sequence_overlap_between_cohorts': 0,
        'baseline_exposure': 'M1 saw all official train sequences during baseline training. This split protects the new candidate correction from sequence-level training overlap, but not parent pretraining exposure.',
        'event_mining_rule_to_apply_after_baseline_replay': 'First run of at least 3 consecutive valid frames with IoU < 0.5 and GT crop coverage >= 99.9%; use the first event per sequence and retain the immediately preceding frame. Separately record all valid-frame target/background evidence; do not select events by candidate improvement.',
        'split_roles': {
            'calibration': 'Choose measurement definitions and debug extraction contracts only.',
            'development': 'Test one frozen causal hypothesis and choose at most one candidate version.',
            'locked_holdout': 'Evaluate the selected candidate once after all design choices are frozen.'
        }
    })
    print(json.dumps(output_rows, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
