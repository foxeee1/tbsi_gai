"""CPU counterexamples and tensor contracts; NOT a method admission tool.

No dataset, model checkpoint, tracking evaluation or training is performed.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.models.layers.early_source_evidence import EarlySourceEvidence
from lib.models.layers.target_support_contrast import TargetSupportContrast
from lib.models.layers.cross_source_attenuation import CrossSourceAttenuation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Refusing to overwrite diagnosis evidence')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not checks[name]:
            raise RuntimeError('Contract counterexample failed: ' + name)

    # Same cosine multiset, different spatial meaning. A max/statistics-only
    # function cannot distinguish these even after perfect training.
    sim = torch.tensor([[[0.9, 0.1, 0.1, 0.1]], [[0.1, 0.9, 0.1, 0.1]]])
    support = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]])
    stats = EarlySourceEvidence._distribution_stats(sim, 4)
    check('max_indistinguishable', torch.equal(sim.amax(-1)[0], sim.amax(-1)[1]))
    check('previous_five_statistics_indistinguishable', torch.equal(stats[0], stats[1]))
    rng = torch.get_rng_state().clone()
    readout = TargetSupportContrast()
    action = CrossSourceAttenuation(enabled=True)
    check('construction_does_not_consume_rng', torch.equal(rng, torch.get_rng_state()))
    check('new_parameter_count_zero', sum(p.numel() for m in (readout, action) for p in m.parameters()) == 0)
    contrast = readout(sim, support)
    check('target_vs_background_distinguished', contrast.margin[0, 0] > 0 and contrast.margin[1, 0] < 0)
    reverse = readout(sim, support.roll(1, dims=-1))
    check('wrong_reference_reverses_decision', reverse.margin[0, 0] < 0 and reverse.margin[1, 0] > 0)
    joint_permutation = torch.tensor([2, 0, 3, 1])
    moved = readout(sim[..., joint_permutation], support[..., joint_permutation])
    check('joint_spatial_relabeling_invariant', torch.allclose(moved.margin, contrast.margin))

    values = torch.arange(1., 25.).reshape(2, 2, 1, 6).requires_grad_()
    margin = contrast.margin.clone().requires_grad_()
    output = action(values, margin, contrast.usable)
    coefficients = action.coefficients(margin, contrast.usable)
    check('nonnegative_margin_preserves_cross_value', torch.equal(output[0], values[0]))
    check('coefficients_bounded', coefficients.min() >= .75 and coefficients.max() <= 1.)
    check('source_vector_direction_preserved', torch.allclose(
        torch.nn.functional.normalize(output, dim=-1), torch.nn.functional.normalize(values, dim=-1)))
    check('default_off_exact_identity', CrossSourceAttenuation()(values, margin, contrast.usable) is values)
    check('zero_strength_exact_identity', CrossSourceAttenuation(0, enabled=True)(values, margin, contrast.usable) is values)
    output.sum().backward()
    check('value_gradient_reaches_each_element', torch.isfinite(values.grad).all() and (values.grad > 0).all())
    check('new_evidence_action_gradient_detached', margin.grad is None)
    for fill in (0., 1.):
        invalid = readout(sim, torch.full_like(support, fill))
        check('degenerate_support_identity_' + str(fill), not invalid.usable.any() and torch.equal(
            action(values, invalid.margin, invalid.usable), values))
    flat = readout(sim, torch.full_like(support, .5))
    check('uninformative_uniform_support_identity', torch.equal(action(values, flat.margin, flat.usable), values))

    # A non-square attention block catches query/source confusion.
    attn = torch.softmax(torch.arange(12.).reshape(1, 2, 3, 2), dim=-1)
    source = torch.tensor([[[[1., 2.], [3., 1.]], [[2., 1.], [1., 4.]]]])
    per_source = torch.tensor([[.4, -.8]])
    usable = torch.ones_like(per_source, dtype=torch.bool)
    got = attn @ action(source, per_source, usable)
    coeff = action.coefficients(per_source, usable)
    expected = sum(attn[..., i:i+1] * source[:, :, i:i+1, :] * coeff[:, None, i:i+1, None]
                   for i in range(2))
    check('rectangular_attention_scales_source_columns', torch.allclose(got, expected))
    stream = io.BytesIO()
    torch.save(action.state_dict(), stream)
    stream.seek(0)
    clone = CrossSourceAttenuation(enabled=True)
    clone.load_state_dict(torch.load(stream, map_location='cpu'), strict=True)
    check('save_load_with_same_explicit_constructor', torch.equal(clone(source, per_source, usable), action(source, per_source, usable)))
    # Cancellation counterexample: attenuating each vector cannot guarantee
    # aggregate norm reduction. Recording this prevents a false paper claim.
    cancellation = torch.tensor([[[[1.], [-1.]]]])
    cancellation_margin = torch.tensor([[.2, -1.]])
    summed = (torch.ones(1, 1, 1, 2) * .5) @ action(cancellation, cancellation_margin, usable)
    check('aggregation_norm_can_increase_due_to_cancellation', summed.norm() > 0)
    try:
        action(torch.ones(1, 1, 3, 2), per_source, usable)
        axis_rejected = False
    except ValueError:
        axis_rejected = True
    check('query_axis_misuse_rejected', axis_rejected)

    table = ROOT / 'experiments/early_gate_closed_loop_20261026/source_evidence_p1_20261027/paired_analysis_u0468_screen96_max2000/per_sequence.csv'
    rows = list(csv.DictReader(table.open()))
    groups = {}
    for group in ('ALL', 'low_target', 'middle', 'normal_protection'):
        selected = [r for r in rows if group == 'ALL' or r['group'] == group]
        selected.sort(key=lambda r: float(r['auc_delta']))
        groups[group] = {
            'n': len(selected), 'mean_delta_pp': sum(float(r['auc_delta']) for r in selected) / len(selected),
            'worst3': [{'sequence': r['sequence'], 'delta_pp': float(r['auc_delta'])} for r in selected[:3]],
            'mean_without_worst3_pp_descriptive_only': sum(float(r['auc_delta']) for r in selected[3:]) / (len(selected) - 3),
        }
    files = [table, Path(__file__).resolve(), ROOT / 'lib/models/layers/target_support_contrast.py',
             ROOT / 'lib/models/layers/cross_source_attenuation.py', ROOT / 'lib/models/layers/early_source_evidence.py']
    result = {'checks': checks, 'check_count': len(checks), 'counterexample_margin': contrast.margin.tolist(),
              'counterexample_coefficients': coefficients.tolist(), 'historical_development_cohort': groups,
              'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              'environment': {'python': sys.version, 'torch': torch.__version__, 'platform': platform.platform(),
                              'device': 'CPU only', 'threads': torch.get_num_threads(), 'seed': 42},
              'command': sys.argv,
              'status': 'ALGEBRAIC-CONTRACT-ONLY / NOT-INTEGRATED / NO-TRAINING / NO-PERFORMANCE-CLAIM',
              'limitations': ['Constructed counterexamples do not identify the cause of historical tracking losses.',
                              'The deployment target-support estimator is not implemented or validated.',
                              'Dropping worst sequences is descriptive, not a valid performance estimate.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
