"""Explicit self/cross message normalization for the joint MSHA path.
The module has no trainable parameters.  It keeps the original Value inputs
and returns the two message blocks separately so the host can retain the
existing residual/output projection contract.

Modes used in the 2026-10-28 quick screen:
  ``cross_row``: normalize each cross row to unit mass while leaving the
  self block on its original joint-softmax scale.
  ``block``: normalize self and cross blocks independently, then mix them
  with a head-shared cross budget (the mean original cross mass).

The head-shared budget in ``block`` is intentional: using a per-head budget
would algebraically reproduce the original joint softmax and would be a
no-op.  The module is kept parameter-free so this screen tests the attention
geometry rather than adding another learned gate.
"""

import torch
from torch import nn


class MSHAMessageNormalization(nn.Module):
    VALID_MODES = ("off", "cross_row", "block")

    def __init__(self, mode="off", eps=1e-6):
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode not in self.VALID_MODES:
            raise ValueError("unknown MSHA message normalization mode: %s" % self.mode)
        self.eps = float(eps)

    @staticmethod
    def _check(attention, values, name):
        if attention.ndim != 4 or values.ndim != 4:
            raise ValueError("%s attention/values must be rank-4" % name)
        if attention.shape[:2] != values.shape[:2] or attention.shape[3] != values.shape[2]:
            raise ValueError("%s attention/values axis mismatch" % name)

    def forward(self, self_attention, cross_attention, self_values, cross_values):
        """Return ``(self_message, cross_message, diagnostics)``.

        Attention axes are ``[batch, head, receiver, source]`` and values are
        ``[batch, head, source, head_dim]``.  The caller supplies the same
        post-dropout attention used by the original MSHA message products.
        """
        self._check(self_attention, self_values, "self")
        self._check(cross_attention, cross_values, "cross")
        if self_attention.shape[:3] != cross_attention.shape[:3]:
            raise ValueError("self/cross receiver axes must match")

        native_self = torch.matmul(self_attention, self_values)
        native_cross = torch.matmul(cross_attention, cross_values)
        if self.mode == "off":
            return native_self, native_cross, {"mode": self.mode}

        self_mass = self_attention.sum(dim=-1, keepdim=True)
        cross_mass = cross_attention.sum(dim=-1, keepdim=True)
        self_cond = self_attention / self_mass.clamp_min(self.eps)
        cross_cond = cross_attention / cross_mass.clamp_min(self.eps)

        if self.mode == "cross_row":
            # Cross budget is deliberately reset to one per head/receiver.
            # This is the only difference from the native joint-softmax path
            # in this arm; no learned gate or auxiliary loss is introduced.
            self_message = native_self
            cross_message = torch.matmul(cross_cond, cross_values)
            budget = cross_mass
        else:
            # A per-head budget would cancel the conditional normalization and
            # reproduce joint softmax exactly.  Share the observed cross mass
            # over heads to test a genuine block-budget intervention.
            rho = cross_mass.mean(dim=1, keepdim=True)
            self_message = (1.0 - rho) * torch.matmul(self_cond, self_values)
            cross_message = rho * torch.matmul(cross_cond, cross_values)
            budget = rho

        return self_message, cross_message, {
            "mode": self.mode,
            "self_mass": self_mass,
            "cross_mass": cross_mass,
            "effective_cross_budget": budget,
        }
