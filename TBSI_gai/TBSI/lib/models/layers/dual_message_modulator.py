"""Receiver-side self/cross actions conditioned on spatial evidence."""
import torch
from torch import nn


class DualMessageModulator(nn.Module):
    def __init__(self, dim=768, evidence_dim=32, hidden_dim=32, message_dim=16,
                 max_delta=0.1):
        super().__init__()
        self.dim = int(dim)
        self.evidence_dim = int(evidence_dim)
        self.message_dim = int(message_dim)
        self.max_delta = float(max_delta)
        self.message_norm = nn.LayerNorm(self.dim)
        self.message_projection = nn.Linear(self.dim, self.message_dim)
        self.self_head = nn.Sequential(
            nn.Linear(2 * self.evidence_dim + 2 + self.message_dim + 1, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 1))
        cross_dim = 4 * self.evidence_dim + 3 * self.message_dim + 5
        self.cross_head = nn.Sequential(
            nn.Linear(cross_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.reset_identity()

    def reset_identity(self):
        for head in (self.self_head, self.cross_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    @staticmethod
    def _transport(attention, source_state, source_coverage):
        # Attention rows are receiver tokens and columns are source tokens.
        relation = attention.float().mean(dim=1)
        mass = relation.sum(dim=-1, keepdim=True)
        conditional = relation / mass.clamp_min(1e-6)
        transported_state = torch.bmm(conditional, source_state.float())
        transported_coverage = torch.bmm(conditional, source_coverage.float().unsqueeze(-1))
        return transported_state, transported_coverage, mass

    def _message_features(self, message):
        b, heads, tokens, channels = message.shape
        if heads * channels != self.dim:
            raise ValueError('message head/channel product must match configured model dim')
        merged = message.transpose(1, 2).reshape(b, tokens, self.dim).float()
        projected = self.message_projection(self.message_norm(merged))
        magnitude = torch.log1p(merged.norm(dim=-1, keepdim=True) / (self.dim ** 0.5))
        return projected, magnitude

    def forward(self, receiver_state, source_state, receiver_coverage,
                source_coverage, self_message, cross_message,
                self_attention, cross_attention, enable_self=False):
        if receiver_state.ndim != 3 or source_state.ndim != 3:
            raise ValueError('evidence states must have shape [B,N,32]')
        if receiver_state.shape[0] != source_state.shape[0] or \
                receiver_state.shape[-1] != self.evidence_dim or \
                source_state.shape[-1] != self.evidence_dim:
            raise ValueError('receiver/source evidence shape mismatch')
        if self_attention.shape[2] != receiver_state.shape[1] or \
                self_attention.shape[3] != receiver_state.shape[1]:
            raise ValueError('self attention must be [B,H,Nreceiver,Nselfsource]')
        if cross_attention.shape[2] != receiver_state.shape[1] or \
                cross_attention.shape[3] != source_state.shape[1]:
            raise ValueError('cross attention must be [B,H,Nreceiver,Ncrosssource]')

        self_source, self_weight, _ = self._transport(
            self_attention, receiver_state, receiver_coverage)
        cross_source, cross_weight, cross_mass = self._transport(
            cross_attention, source_state, source_coverage)
        own_message, own_magnitude = self._message_features(self_message)
        other_message, other_magnitude = self._message_features(cross_message)

        self_input = torch.cat((receiver_state.float(), self_source,
                                receiver_coverage.float().unsqueeze(-1), self_weight,
                                own_message, own_magnitude), dim=-1)
        self_scale = 1.0 + self.max_delta * torch.tanh(self.self_head(self_input))
        cross_input = torch.cat((receiver_state.float(), cross_source,
                                 receiver_state.float() - cross_source,
                                 receiver_state.float() * cross_source,
                                 own_message, other_message, own_message - other_message,
                                 receiver_coverage.float().unsqueeze(-1), cross_weight,
                                 cross_mass, own_magnitude, other_magnitude), dim=-1)
        cross_scale = 1.0 + self.max_delta * torch.tanh(self.cross_head(cross_input))
        if not enable_self:
            self_scale = torch.ones_like(self_scale)
        return self_scale.to(self_message.dtype), cross_scale.to(cross_message.dtype), {
            'self_source_state': self_source,
            'cross_source_state': cross_source,
            'cross_attention_mass': cross_mass,
            'self_message_magnitude': own_magnitude,
            'cross_message_magnitude': other_magnitude,
        }


def reset_dual_message_modulator(model):
    count = 0
    for module in model.modules():
        if isinstance(module, DualMessageModulator):
            module.reset_identity()
            count += 1
    return count
