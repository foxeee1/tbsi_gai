"""Train-only foreground-source supervision for the archived fixed-g=1 MR.

No model parameter or inference path is added. Recompute only the existing
allocator on detached inputs, so auxiliary gradients cannot change the base
attention, Value features, or the context encoder to lower this loss.
"""
import math

import torch
from torch import nn


def template_token_coverage(boxes, tokens):
    """Fraction of each row-major token covered by normalized crop xywh GT.

    Uses box/cell intersection, retaining tiny and boundary-straddling targets.
    Boxes may extend beyond the crop. Invalid or empty clipped boxes are masked.
    """
    side = math.isqrt(tokens)
    if side * side != tokens or boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError('Expected square template grid and [B,4] normalized xywh')
    box = boxes.detach().float()
    valid = torch.isfinite(box).all(-1) & (box[:, 2:] > 0).all(-1)
    box = torch.where(valid[:, None], box, torch.zeros_like(box))
    lo = box[:, :2].clamp(0, 1)
    hi = (box[:, :2] + box[:, 2:]).clamp(0, 1)
    axis = torch.arange(side, device=box.device, dtype=torch.float32) / side
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    cell_lo = torch.stack((xx.flatten(), yy.flatten()), -1)
    overlap = (torch.minimum(hi[:, None], cell_lo[None] + 1. / side) -
               torch.maximum(lo[:, None], cell_lo[None])).clamp_min(0)
    coverage = (overlap.prod(-1) * side * side).clamp(0, 1)
    return coverage, valid & (coverage.sum(-1) > 0)


class MRSourceForegroundLoss(nn.Module):
    """Capped -log foreground mass, averaged over six allocator directions.

    Source GT is a spatial training proxy, not a reliability label. Foreground
    receivers are weighted by fractional coverage and detached native cross
    mass. Confident rows stop receiving this auxiliary gradient at threshold.
    Existing allocator references live in a Python list, not a ModuleList;
    neither they nor new parameters are registered on this loss object.
    """

    def __init__(self, model, threshold=0.5):
        super().__init__()
        if not 0 < threshold < 1:
            raise ValueError('foreground threshold must lie in (0,1)')
        self.threshold = float(threshold)
        records = []
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'ReceiverSourceMassReallocator':
                if name.endswith('mass_reallocator_rgb'):
                    records.append((name, module, 0, 1))
                elif name.endswith('mass_reallocator_tir'):
                    records.append((name, module, 1, 0))
        if len(records) != 6:
            raise ValueError('Expected exactly six original pair MR allocators')
        self.records = records
        self.captured = {}
        self.capture_enabled = True
        self.handles = [module.register_forward_hook(self._capture(i))
                        for i, (_, module, _, _) in enumerate(records)]

    def _capture(self, index):
        def hook(module, inputs, output):
            if self.capture_enabled and module.training and torch.is_grad_enabled():
                if index in self.captured:
                    raise RuntimeError('Allocator reused before previous batch was cleared')
                self.captured[index] = tuple(value.detach() for value in inputs)
        return hook

    def begin_batch(self):
        self.captured.clear()

    def forward(self, template_boxes):
        """template_boxes is [B,2,4], RGB then TIR, after crop/flip transforms."""
        if template_boxes.ndim != 3 or template_boxes.shape[1:] != (2, 4):
            raise ValueError('Expected actual augmented RGB/TIR template boxes [B,2,4]')
        captured, self.captured = self.captured, {}
        if set(captured) != set(range(6)):
            raise RuntimeError('All six original allocator paths must execute once')
        terms, foreground, baseline, active, valid_fractions, mass_errors = [], [], [], [], [], []
        self.capture_enabled = False
        try:
            # This matches native allocator autocast behavior; FP32 is used for
            # normalization/log reduction below, not a different forward kernel.
            for index, (_, module, receiver_id, source_id) in enumerate(self.records):
                inputs = captured[index]
                attention = inputs[0]
                b, h, nq, ns = attention.shape
                if template_boxes.shape[0] != b:
                    raise ValueError('Template annotation batch differs from allocator')
                receiver_fg, receiver_valid = template_token_coverage(template_boxes[:, receiver_id], nq)
                source_fg, source_valid = template_token_coverage(template_boxes[:, source_id], ns)
                source_has_background = (1 - source_fg).sum(-1) > 1e-6
                valid = receiver_valid & source_valid & source_has_background
                adjusted = module(*inputs)
                with torch.cuda.amp.autocast(enabled=False):
                    a, adjusted = attention.float(), adjusted.float()
                    native_mass = a.sum(-1)
                    adjusted_mass = adjusted.sum(-1)
                    q = adjusted / adjusted_mass[..., None].clamp_min(1e-20)
                    p = a / native_mass[..., None].clamp_min(1e-20)
                    probability = (q * source_fg[:, None, None]).sum(-1).clamp(0, 1)
                    native_probability = (p * source_fg[:, None, None]).sum(-1)
                    weights = (receiver_fg[:, None, :] * native_mass.detach() *
                               valid[:, None, None]).detach()
                    # For targets smaller than a token, attainable fractional
                    # coverage can be <0.5 everywhere. Scale the cap by maximum
                    # source coverage instead of imposing an impossible label.
                    cap = self.threshold * source_fg.amax(-1)[:, None, None]
                    row_loss = (cap.clamp_min(1e-8).log() -
                                probability.clamp_min(1e-8).log()).clamp_min(0)
                    denom = weights.sum()
                    if float(denom) > 0:
                        terms.append((row_loss * weights).sum() / denom)
                        foreground.append(((probability.detach() * weights).sum() / denom))
                        baseline.append(((native_probability * weights).sum() / denom))
                        active.append((((probability.detach() < cap) * weights).sum() / denom))
                    valid_fractions.append(valid.float().mean())
                    mass_errors.append((adjusted_mass.detach() - native_mass).abs().amax())
            zero = sum(module.readout[-1].weight.sum() * 0 for _, module, _, _ in self.records)
            loss = torch.stack(terms).mean() if terms else zero
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite MR foreground auxiliary loss')
            def mean(values):
                return float(torch.stack(values).mean()) if values else 0.
            return loss, {
                'Loss/mr_source_foreground_raw': float(loss.detach()),
                'MRForeground/allocated_fg_mass': mean(foreground),
                'MRForeground/native_fg_mass': mean(baseline),
                'MRForeground/active_row_weight_fraction': mean(active),
                'MRForeground/valid_sample_fraction': mean(valid_fractions),
                'MRForeground/max_cross_mass_error': float(torch.stack(mass_errors).max()),
                'MRForeground/valid_directions': len(terms)}
        finally:
            self.capture_enabled = True

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.captured.clear()
