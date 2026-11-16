# LEGACY V1: retained for provenance only. Do not use as the current admission rule.
# Current policy: experiments/early_gate_closed_loop_20261026/audit_v2/AUDIT_RULES_V2.md
"""Validate required research metadata; this does not prove causal validity."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):
            h.update(part)
    return h.hexdigest()


def validate(path,for_training=False):
    d=json.loads(path.read_text())
    project=Path(__file__).resolve().parents[1]
    for key in ('hypothesis','parent','single_variable','falsifier',
                'allowed_claim_now','forbidden_claim_now'):
        if not isinstance(d.get(key),str) or not d[key].strip():
            raise ValueError('missing explicit '+key)
    module=Path(d['candidate_module'])
    if str(module.parent)!='lib/models/layers' or module.suffix!='.py' or not (project/module).is_file():
        raise ValueError('candidate must be an independent existing layers/*.py')
    if set(d['arms']) != {'early_gate_only','radial','direction'}:
        raise ValueError('missing a discriminative primary arm')
    fixed=d['fixed']
    if fixed['qk']!='native' or fixed['self_value']!='native' or fixed['new_loss'] or fixed['search_writeback']:
        raise ValueError('multiple main interventions conflict with current contract')
    requirements=d['required_before_training']
    missing=[key for key,value in requirements.items() if value is None or value=='' or value==[]]
    if d.get('training_ready') and missing:
        raise ValueError('training_ready=true but required evidence is missing')
    if for_training:
        if missing or not d.get('training_ready'):
            raise ValueError('training not ready: '+', '.join(missing))
        for field in ('dataset_manifest','common_parent_checkpoint'):
            target=project/requirements[field]
            if not target.is_file() or digest(target)!=requirements[field+'_sha256']:
                raise ValueError('missing/mismatched '+field)
        for entry in requirements['config_paths_and_sha256']:
            target=project/entry['path']
            if not target.is_file() or digest(target)!=entry['sha256']:
                raise ValueError('missing/mismatched config')
        for key in ('post_augmentation_repeatability_evidence','optimizer_successful_step_logger',
                    'resource_process_audit'):
            if not (project/requirements[key]).is_file():
                raise ValueError('missing evidence file '+key)
        protected={'TBSI_E_MHSA_FULL_4090D_seed42',
                   'M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_JOINT_PRETRAIN_20261023'}
        for name in requirements['explicit_output_directories']:
            target=(project/name).resolve()
            if any(part in protected for part in target.parts) or target==project.resolve():
                raise ValueError('protected/unsafe output directory')
            if target.exists() and any(target.iterdir()):
                raise ValueError('output directory already contains artifacts')
    return dict(design_metadata_valid=True,training_ready=bool(d['training_ready']),
        missing_training_evidence=missing,note='metadata presence only; scientific validity needs review')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('contract',type=Path)
    parser.add_argument('--for-training',action='store_true')
    args=parser.parse_args()
    print(json.dumps(validate(args.contract,args.for_training),indent=2))
