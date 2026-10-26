"""Read-only archive/replay summary for the M1 fusion bottleneck audit."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENES = ROOT / 'output/experiments/M1_FULL244_SCENE_MINING_20261009/per_sequence_scene.csv'
METRICS = ROOT / 'output/experiments/M1_ARCHIVED_PR_SR_SCENES_20261010/per_sequence_metrics.csv'
FLOW = ROOT / 'output/experiments/M1_FUSION_FLOW_FULL244_DIAG_20261009/gradient_sampled_32/per_frame.csv'
GROUPS = ('LR', 'SA', 'TC', 'LI', 'BC', 'HO', 'MB', 'CM', 'DEF')
FLOW_FIELDS = (
    'loss_giou', 'loss_l1', 'loss_focal', 'response_peak', 'response_gt_cell',
    'mhsa_9_cross_mass_rgb_to_tir', 'mhsa_9_cross_mass_tir_to_rgb',
    'mhsa_9_r_to_t_mass', 'mhsa_9_t_to_r_mass', 'search_input_cosine',
    'search_input_grad_rgb_norm', 'search_input_grad_tir_norm',
    'search_output_grad_norm', 'mhsa_9_entropy',
)


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError('refusing nonempty output directory: {}'.format(output_dir))

    scenes, metrics, flow = read_csv(SCENES), read_csv(METRICS), read_csv(FLOW)
    if len(scenes) != 244 or len(metrics) != 244 or len(flow) != 7807:
        raise RuntimeError('unexpected full-244 source counts')
    metric_by_name = {row['sequence']: row for row in metrics}
    if len(metric_by_name) != 244 or {r['sequence'] for r in scenes} != set(metric_by_name):
        raise RuntimeError('archive sequence join is not one-to-one')

    archive = {}
    for name in ('ALL',) + GROUPS:
        rows = [row for row in scenes if name == 'ALL' or name in row['attributes'].split(';')]
        count = lambda field: sum(int(row[field]) for row in rows)
        inside = count('severe_inside_frames')
        archive[name] = {
            'n_sequences': len(rows),
            'sequence_equal_auc': statistics.mean(float(metric_by_name[r['sequence']]['success_auc']) for r in rows),
            'sequence_equal_pr20': statistics.mean(float(metric_by_name[r['sequence']]['precision_20px']) for r in rows),
            'sequence_equal_op75': statistics.mean(float(metric_by_name[r['sequence']]['op75']) for r in rows),
            'severe_valid_frames': count('severe_valid_frames'),
            'severe_inside_frames': inside,
            'severe_inside_center_large_frames': count('severe_inside_center_large_frames'),
            'severe_inside_center_only_frames': count('severe_inside_center_only_frames'),
            'severe_inside_area_only_frames': count('severe_inside_area_only_frames'),
            'persistent_severe_runs': count('severe_run_ge30_count'),
            'sequences_with_persistent_severe_run': sum(int(r['severe_run_ge30_count']) > 0 for r in rows),
        }
        archive[name]['center_large_pct_of_inside'] = (
            100 * archive[name]['severe_inside_center_large_frames'] / inside if inside else None)
        archive[name]['center_only_pct_of_inside'] = (
            100 * archive[name]['severe_inside_center_only_frames'] / inside if inside else None)
        archive[name]['area_only_pct_of_inside'] = (
            100 * archive[name]['severe_inside_area_only_frames'] / inside if inside else None)

    inside = [row for row in flow if row['gt_valid'] == 'True'
              and float(row['crop_gt_area_fraction']) >= 0.999]
    subsets = {
        'crop_inside_good_iou_ge_05': [row for row in inside if float(row['tracking_iou']) >= 0.5],
        'crop_inside_severe_iou_lt_02': [row for row in inside if float(row['tracking_iou']) < 0.2],
    }
    replay = {
        name: {'n_frames': len(rows), 'n_sequences': len({row['sequence'] for row in rows}),
               'median': {field: statistics.median(float(row[field]) for row in rows)
                          for field in FLOW_FIELDS}}
        for name, rows in subsets.items()
    }
    if len(inside) != 6632 or replay['crop_inside_good_iou_ge_05']['n_frames'] != 5372 or replay['crop_inside_severe_iou_lt_02']['n_frames'] != 785:
        raise RuntimeError('unexpected replay category counts')

    result = {
        'id': 'M1-FUSION-FULL244-PRIORITY-AUDIT-20261010',
        'sources': {str(path.relative_to(ROOT)): {'sha256': sha256(path), 'rows': len(rows)}
                    for path, rows in ((SCENES, scenes), (METRICS, metrics), (FLOW, flow))},
        'archive': archive,
        'current_worktree_replay': replay,
        'replay_crop_inside_sample_count': len(inside),
        'limits': [
            'Archive and current-worktree replay are not frame-equivalent; no frame-level join is made.',
            'Attribute groups overlap; severe-frame fractions are frame-weighted and descriptive.',
            'Sampled replay gradients are frozen-loss sensitivity, not parameter updates or causal contributions.',
            'No new tracking inference, training, or test-set method selection is performed.',
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'summary.json').open('w') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(output_dir / 'summary.json')


if __name__ == '__main__':
    main()
