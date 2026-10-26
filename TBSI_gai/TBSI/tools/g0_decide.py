#!/usr/bin/env python3
"""Apply the pre-registered G0 information-source decision gate."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export-dir', required=True)
    ap.add_argument('--labels-dir', required=True)
    ap.add_argument('--probe', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--min-delta-ap', type=float, default=0.03)
    args = ap.parse_args()
    export = Path(args.export_dir).resolve()
    labels = Path(args.labels_dir).resolve()
    probe = json.loads(Path(args.probe).read_text())
    parity = json.loads((export / 'parity.json').read_text())
    label_summary = json.loads((labels / 'label_summary.json').read_text())
    paired = probe['paired']
    checks = {
        'export_parity': bool(parity.get('passed')) and parity.get('max_abs_box_diff') == 0.0,
        'target_coverage_reported': label_summary['events'] > 0,
        'L_plus_E_delta_ap': paired['L_plus_E']['mean_delta_ap'] >= args.min_delta_ap,
        'L_plus_E_beats_Lcontrol': paired['L_plus_E']['mean_delta_ap'] > paired['L_plus_Lcontrol']['mean_delta_ap'],
        'Eshuffle_loses_increment': paired['L_plus_Eshuffle']['mean_delta_ap'] < paired['L_plus_E']['mean_delta_ap'],
        'valid_split_exists': probe['valid_events'] > 0 and len(probe['valid_sequences']) > 0,
    }
    # The current 24-sequence stage is a veto/expand decision only; it never authorizes training.
    passed = all(checks.values())
    decision = 'EXPAND_DIAGNOSTIC' if passed else 'STOP'
    result = {'decision': decision, 'checks': checks, 'thresholds': {'min_delta_ap': args.min_delta_ap},
              'parity': parity, 'label_summary': label_summary, 'probe_paired': paired,
              'training_authorized': False,
              'interpretation': ('Early features show a conditional probe signal; expand only to the pre-registered '
                                 'sequence-isolated diagnostic set before any SCSU/P1 training.') if passed else
                                ('G0 did not establish a robust conditional early-feature signal; do not implement '
                                 'or train SCSU from this evidence.')}
    Path(args.output).resolve().write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
