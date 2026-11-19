"""Target-referenced pairwise allocation for Early-gated cross-modal messages."""
import torch
from torch import nn
from torch.nn import functional as F

from .structured_spatial_evidence import box_to_patch_coverage


class FineGrainedUnifiedAllocator(nn.Module):
    """Reallocate an existing cross-attention prior using receiver/source evidence.

    The module preserves the aggregate coefficient mass of the original
    attention-times-Early-gate message, then applies a bounded rejection term.
    Its zero scales make the initial output exactly the input message.
    """

    def __init__(self, dim, rank=8, max_relation=0.5, max_rejection=0.5,
                 retrieval_temperature=0.1):
        super().__init__()
        if rank < 1 or max_relation <= 0 or not 0 <= max_rejection < 1:
            raise ValueError('invalid fine-grained allocator dimensions or bounds')
        if retrieval_temperature <= 0:
            raise ValueError('retrieval temperature must be positive')
        self.rank = int(rank)
        self.max_relation = float(max_relation)
        self.max_rejection = float(max_rejection)
        self.retrieval_temperature = float(retrieval_temperature)
        self.content_norm = nn.LayerNorm(dim)
        self.receiver_content = nn.Linear(dim, rank, bias=False)
        self.source_content = nn.Linear(dim, rank, bias=False)
        self.retrieved_content = nn.Linear(dim, rank, bias=False)
        self.receiver_reference = nn.Linear(5, rank, bias=False)
        self.source_reference = nn.Linear(5, rank, bias=False)
        # A relation-only ablation must not carry a dormant rejection head.
        # This keeps the F arm's only learned action the receiver/source
        # redistribution; the legacy FU path still uses the rejection head.
        self.reject = nn.Linear(rank, 1) if self.max_rejection > 0 else None
        self.relation_scale = nn.Parameter(torch.zeros(()))
        self.rejection_bias = (nn.Parameter(torch.tensor(-6.0))
                              if self.max_rejection > 0 else None)

    def reset_identity(self):
        with torch.no_grad():
            self.relation_scale.zero_()
            if self.max_rejection > 0:
                self.rejection_bias.fill_(-6.0)

    @staticmethod
    def _target_background_affinity(tokens, boxes):
        coverage = box_to_patch_coverage(boxes, grid_size=8).to(tokens.device)
        if coverage.shape[1] != tokens.shape[1]:
            raise ValueError('template token count must match the configured 8x8 box grid')
        target_w = coverage / coverage.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        background = 1.0 - coverage
        background_w = background / background.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        normalized = F.normalize(tokens.float(), dim=-1, eps=1e-6)
        target_proto = F.normalize(torch.bmm(target_w[:, None], normalized), dim=-1, eps=1e-6)
        background_proto = F.normalize(torch.bmm(background_w[:, None], normalized), dim=-1, eps=1e-6)
        target_affinity = torch.bmm(normalized, target_proto.transpose(1, 2)).squeeze(-1)
        background_affinity = torch.bmm(normalized, background_proto.transpose(1, 2)).squeeze(-1)
        has_background = (background.sum(dim=-1, keepdim=True) > 1e-6).to(tokens.dtype)
        background_affinity = background_affinity * has_background
        return torch.stack((target_affinity, background_affinity), dim=-1), coverage

    def forward(self, attention, values, source_gate, receiver_tokens, source_tokens,
                receiver_boxes, source_boxes, receiver_evidence, source_evidence,
                source_similarity, source_search):
        if attention.ndim != 4 or values.ndim != 4:
            raise ValueError('attention and values must have shapes [B,H,Nr,Ns] and [B,H,Ns,D]')
        batch, heads, n_receiver, n_source = attention.shape
        if values.shape[:3] != (batch, heads, n_source):
            raise ValueError('value source axes must match cross attention')
        if source_gate.shape != (batch, n_source):
            raise ValueError('source gate must have shape [B,Nsource]')
        if receiver_tokens.shape[:2] != (batch, n_receiver) or source_tokens.shape[:2] != (batch, n_source):
            raise ValueError('receiver/source token axes do not match cross attention')
        if receiver_evidence.shape != (batch, n_receiver, 3) or source_evidence.shape != (batch, n_source, 3):
            raise ValueError('early evidence tensors must have shape [B,Ntoken,3]')
        if source_similarity.shape[:2] != (batch, n_source):
            raise ValueError('source similarity must have shape [B,Nsource,Nsearch]')
        if source_search.shape[:2] != (batch, source_similarity.shape[-1]):
            raise ValueError('source search token axes do not match the similarity map')

        recv_affinity, _ = self._target_background_affinity(receiver_tokens, receiver_boxes)
        src_affinity, _ = self._target_background_affinity(source_tokens, source_boxes)
        retrieval = torch.softmax(source_similarity.float() / self.retrieval_temperature, dim=-1)
        retrieved = torch.bmm(retrieval, source_search.float())

        recv_ref = torch.cat((recv_affinity, receiver_evidence.float()), dim=-1)
        src_ref = torch.cat((src_affinity, source_evidence.float()), dim=-1)
        recv = self.content_norm(receiver_tokens.float())
        src = self.content_norm(source_tokens.float())
        query = torch.tanh(self.receiver_content(recv) + self.receiver_reference(recv_ref))
        key = torch.tanh(self.source_content(src) + self.retrieved_content(retrieved) +
                         self.source_reference(src_ref))
        pair = torch.matmul(query, key.transpose(1, 2)) / (self.rank ** 0.5)
        delta = self.max_relation * torch.tanh(self.relation_scale) * torch.tanh(pair)

        base = attention.float() * source_gate[:, None, None, :].float()
        weighted = base * torch.exp(delta[:, None, :, :])
        base_mass = base.sum(dim=-1, keepdim=True)
        weighted_mass = weighted.sum(dim=-1, keepdim=True)
        redistributed = weighted * base_mass / weighted_mass.clamp_min(1e-12)

        if self.max_rejection > 0:
            rejection = self.max_rejection * torch.sigmoid(
                self.reject(query).squeeze(-1) + self.rejection_bias)
            redistributed = redistributed * (1.0 - rejection[:, None, :, None])
        return torch.matmul(redistributed.to(values.dtype), values)


def reset_fine_grained_unified_allocator(model):
    count = 0
    for module in model.modules():
        if isinstance(module, FineGrainedUnifiedAllocator):
            module.reset_identity()
            count += 1
    return count
