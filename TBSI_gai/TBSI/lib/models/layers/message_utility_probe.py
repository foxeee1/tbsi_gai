"""Runtime-only probes for causal message utility in TBSI template MSHA.

This module is intentionally independent from the production TBSI layer.  It
registers a forward hook on ``attn_drop`` and removes selected cross-modal
attention entries after softmax.  The hook is used for fixed-state diagnosis;
it is not a trainable model component.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch


@dataclass(frozen=True)
class ProbeMode:
    """A single intervention in the cross-attention matrix.

    ``direction`` names the receiver/source block: ``rt`` means RGB receiver
    from TIR source, and ``tr`` means TIR receiver from RGB source.
    ``source`` and ``receiver`` are token indices within their own modality;
    ``None`` means all tokens on that axis.
    """

    name: str
    direction: Optional[str] = None
    source: Optional[int] = None
    receiver: Optional[int] = None


class CrossAttentionUtilityProbe:
    """Mask cross-modal attention entries and record their mass.

    The probe does not renormalize attention after masking.  This makes the
    intervention a direct removal of the selected value messages:
    ``A_selected V_source``.  The production network and all Q/K logits stay
    unchanged.
    """

    def __init__(self, modules: Iterable[torch.nn.Module], n_rgb: int = 64):
        self.modules = list(modules)
        self.n_rgb = int(n_rgb)
        self.mode = ProbeMode("off")
        self.capture = False
        self.records = []
        self._handles = [module.register_forward_hook(self._hook)
                         for module in self.modules]

    @staticmethod
    def _as_int(value):
        return None if value is None else int(value)

    def set_mode(self, mode: ProbeMode):
        if mode.direction not in (None, "rt", "tr"):
            raise ValueError("direction must be None, rt, or tr")
        self.mode = mode
        self.records = []

    def _mask(self, attention: torch.Tensor) -> torch.Tensor:
        if attention.ndim != 4:
            raise RuntimeError("expected [B,H,N,N] attention, got " +
                               str(tuple(attention.shape)))
        total = attention.shape[-1]
        n_rgb = self.n_rgb
        if total != n_rgb * 2:
            raise RuntimeError("expected two 64-token modalities, got " +
                               str(total))
        mode = self.mode
        mask = torch.ones_like(attention)
        if mode.name == "off":
            return mask
        if mode.name == "cross_all":
            mask[:, :, :n_rgb, n_rgb:] = 0
            mask[:, :, n_rgb:, :n_rgb] = 0
            return mask
        if mode.direction == "rt":
            receiver_slice = slice(0, n_rgb)
            source_offset = n_rgb
        elif mode.direction == "tr":
            receiver_slice = slice(n_rgb, total)
            source_offset = 0
        else:
            raise ValueError("direction is required for a directional mode")
        receiver = mode.receiver
        source = mode.source
        if receiver is None:
            receiver_slice = receiver_slice
        else:
            receiver = self._as_int(receiver)
            if receiver < 0 or receiver >= n_rgb:
                raise ValueError("receiver token out of range: %s" % receiver)
            receiver_slice = slice(receiver + (0 if mode.direction == "rt" else n_rgb),
                                   receiver + 1 + (0 if mode.direction == "rt" else n_rgb))
        if source is None:
            source_slice = slice(source_offset, source_offset + n_rgb)
        else:
            source = self._as_int(source)
            if source < 0 or source >= n_rgb:
                raise ValueError("source token out of range: %s" % source)
            source_slice = slice(source_offset + source, source_offset + source + 1)
        mask[:, :, receiver_slice, source_slice] = 0
        return mask

    def _hook(self, module, inputs, output):
        if not torch.is_tensor(output):
            raise RuntimeError("attn_drop hook received non-tensor output")
        mask = torch.ones_like(output) if self.mode.name == "off" else self._mask(output)
        changed = output * (1.0 - mask)
        # Keep a compact per-layer audit record.  Values are detached so this
        # diagnostic cannot retain an autograd graph.
        if self.capture:
            rt = output[:, :, :self.n_rgb, self.n_rgb:]
            tr = output[:, :, self.n_rgb:, :self.n_rgb]
            selected = changed.abs().sum().detach()
            record = {
                "mode": self.mode.name,
                "direction": self.mode.direction,
                "source": self.mode.source,
                "receiver": self.mode.receiver,
                "shape": list(output.shape),
                "rt_mass": float(rt.sum().detach().cpu()),
                "tr_mass": float(tr.sum().detach().cpu()),
                "removed_attention_mass": float(selected.cpu()),
            }
            # Store only top entries per direction.  This is enough to select
            # source/pair interventions and keeps the JSON audit compact.
            mean_attention = output.detach().float().mean(dim=(0, 1))
            rows = []
            for direction, q0, q1, k0, k1 in (("rt", 0, self.n_rgb,
                                                self.n_rgb, self.n_rgb * 2),
                                               ("tr", self.n_rgb, self.n_rgb * 2,
                                                0, self.n_rgb)):
                block = mean_attention[q0:q1, k0:k1]
                topk = min(32, block.numel())
                values, indices = torch.topk(block.reshape(-1), topk)
                for value, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
                    receiver = int(index // self.n_rgb)
                    source = int(index % self.n_rgb)
                    rows.append({"direction": direction, "receiver": receiver,
                                 "source": source, "value": float(value)})
            record["attention_rows"] = rows
            self.records.append(record)
        return output * mask

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []


def module_contract(modules: Iterable[torch.nn.Module]) -> Dict[str, object]:
    """Return a serializable description used by the probe manifest."""
    modules = list(modules)
    return {
        "count": len(modules),
        "types": [type(module).__name__ for module in modules],
        "has_attn_drop": [hasattr(module, "attn_drop") for module in modules],
    }
