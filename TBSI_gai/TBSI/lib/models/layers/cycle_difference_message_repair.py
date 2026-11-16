"""Constant-preserving refinement of the historical CBMR action.

Only graph normalization changes. The cycle graph, mask, fixed strength and
insertion point match CBMR. This is a diagnostic refinement, not a claimed
reliability estimator. Row-stochastic averaging preserves constant messages
and gives a convex combination; it does not imply an L2 contraction.
"""
import torch

from .cycle_laplacian_message_repair import CycleLaplacianMessageRepair


class CycleDifferenceMessageRepair(CycleLaplacianMessageRepair):
    def _repair_one(self, message, relation_forward, relation_backward,
                    background):
        forward = relation_forward.float()
        backward = relation_backward.float()
        forward = forward / forward.sum(-1, keepdim=True).clamp_min(self.eps)
        backward = backward / backward.sum(-1, keepdim=True).clamp_min(self.eps)
        cycle = forward @ backward
        weight = 0.5 * (cycle + cycle.transpose(-1, -2))
        bg = background[:, None, :].to(weight.dtype)
        weight = weight * bg.unsqueeze(-1) * bg.unsqueeze(-2)
        degree = weight.sum(-1)
        supported = degree > self.eps
        walk = weight / degree.clamp_min(self.eps).unsqueeze(-1)
        msg = message.float()
        edited = msg + self.strength * (walk @ msg - msg)
        edited = torch.where(supported.unsqueeze(-1), edited, msg)
        edited = torch.where(background[:, None, :, None], edited, msg)
        return edited.to(message.dtype), degree, supported
