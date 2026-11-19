#!/usr/bin/env python3
"""Validate and hash all frozen M1 screening protocol artifacts."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'experiments/protocols/m1_screening_v1_20261022'
DATA = ROOT.parent / 'data/lasher'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def members(filename):
    values = (PACKAGE / filename).read_text().splitlines()
    if len(values) != len(set(values)):
        raise RuntimeError('duplicate sequence in {}'.format(filename))
    return set(values)


def main():
    train = members('quick_train_sequences.txt')
    calibration = members('calibration_sequences.txt')
    development = members('development_sequences.txt')
    holdout = members('locked_holdout_sequences.txt')
    evaluation = members('full244_eval_sequences.txt')
    assert len(train) == 176 and len(calibration) == 48
    assert len(development) == 24 and len(holdout) == 24 and len(evaluation) == 244
    cohorts = [train, calibration, development, holdout]
    if any(cohorts[i] & cohorts[j] for i in range(len(cohorts)) for j in range(i + 1, len(cohorts))):
        raise RuntimeError('candidate train/event cohort overlap found')

    quick = json.loads((PACKAGE / 'quick_train_manifest.json').read_text())
    if quick['source_sequences'] != 977 or quick['selected_sequences'] != 176:
        raise RuntimeError('quick-train manifest count mismatch')
    max_attribute_deviation = max(
        abs(quick['selected_attribute_counts'][name] -
            quick['selected_sequences'] * population / quick['source_sequences'])
        for name, population in quick['population_attribute_counts'].items())
    if max_attribute_deviation > 1.1:
        raise RuntimeError('attribute marginal deviation exceeds protocol tolerance')

    with (PACKAGE / 'paper_case_events.csv').open(newline='') as handle:
        paper_cases = list(csv.DictReader(handle))
    if len(paper_cases) != 4:
        raise RuntimeError('expected four retrospective paper cases')
    with (PACKAGE / 'historical_proxy_calibration.csv').open(newline='') as handle:
        calibration_rows = list(csv.DictReader(handle))
    if not calibration_rows or any(len(row) != 21 for row in calibration_rows):
        raise RuntimeError('historical proxy calibration table schema mismatch')

    cleanup = json.loads((PACKAGE / 'abandoned_checkpoint_cleanup_20261022.json').read_text())
    if cleanup['mode'] != 'delete' or cleanup['count'] != 12 or any(x['status'] != 'DELETED' for x in cleanup['files']):
        raise RuntimeError('checkpoint cleanup record is incomplete')

    artifact_paths = sorted(path for path in PACKAGE.rglob('*') if path.is_file() and
                            path.name not in ('protocol_integrity_manifest.json', 'build_summary.json'))
    tool_paths = [ROOT / 'tools/build_m1_screening_protocol_20261022.py',
                  ROOT / 'tools/build_m1_event_cohorts_20261022.py',
                  ROOT / 'tools/cleanup_m1_abandoned_checkpoints_20261022.py',
                  ROOT / 'tools/cleanup_m1_legacy_screening_assets_20261022.py',
                  ROOT / 'tools/cleanup_m1_legacy_sampling_manifests_20261022.py',
                  ROOT / 'tools/finalize_m1_screening_protocol_20261022.py']
    source_files = artifact_paths + tool_paths + [DATA / 'Attributes&Order.zip']
    hashes = {}
    for path in source_files:
        try:
            key = str(path.relative_to(ROOT))
        except ValueError:
            key = str(path)
        hashes[key] = sha256(path)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT)).decode().strip()
    manifest = {
        'id': 'M1-SCREENING-PROTOCOL-INTEGRITY-20261022',
        'git_commit': commit, 'working_tree_was_dirty': True,
        'parent': 'TBSI_E_MHSA_FULL_4090D_seed42 epoch15',
        'parent_config_sha256': 'fab8e818b6cf7796f37fe53bf162512a4e5e8bcebdc34f484ac8ec96316b4e1',
        'parent_checkpoint_sha256': '1302de3bc2613bee1691c57251822ac6667deb1fae6533f99785aea1f54e04ef',
        'dataset': 'LasHeR', 'dataset_root': str(DATA),
        'official_counts': {'train': 977, 'test': 244},
        'cohorts': {'candidate_train': len(train), 'calibration': len(calibration),
                    'development': len(development), 'locked_holdout': len(holdout),
                    'paper_retrospective_cases': len(paper_cases), 'full_eval': len(evaluation)},
        'candidate_train_event_cohort_overlap': 0,
        'max_official_attribute_count_deviation': max_attribute_deviation,
        'abandoned_checkpoint_cleanup': {'count': cleanup['count'], 'bytes': cleanup['bytes'],
                                         'all_deleted': True, 'anchors_excluded': True},
        'sha256': hashes,
    }
    out = PACKAGE / 'protocol_integrity_manifest.json'
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    summary = {
        'protocol_id': 'M1-RESEARCH-SCREENING-PROTOCOL-V1-20261022',
        'candidate_train': len(train), 'candidate_train_fraction': len(train) / 977,
        'event_cohorts': {'calibration': len(calibration), 'development': len(development),
                          'locked_holdout': len(holdout)},
        'full_eval': len(evaluation), 'paper_retrospective_cases': len(paper_cases),
        'candidate_event_overlap': 0, 'max_official_attribute_count_deviation': max_attribute_deviation,
        'historical_proxy_calibration': 'NOT-CALIBRATED; only two directionally comparable candidates and contracts differ',
        'abandoned_checkpoint_cleanup_bytes': cleanup['bytes'],
        'protocol_integrity_manifest_sha256': sha256(out),
    }
    (PACKAGE / 'build_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
