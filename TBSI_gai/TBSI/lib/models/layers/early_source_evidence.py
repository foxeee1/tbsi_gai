"""Distribution-aware source evidence for the Early-conditioned MSHA gate.

The module has one responsibility: refine the evidence supplied to the
existing Early gate.  It never changes Q/K, attention weights, or Value
content.  Its inputs are statistics of the already computed template-search
similarity matrices, so it does not introduce a second matching path.
"""
import torch
from torch import nn
from torch.nn import functional as F


class EarlySourceEvidence(nn.Module):
    """Learn a bounded residual over the parent three-channel evidence.

    The feature vector contains max similarity, top-1/top-2 margin, spread,
    mean and top-k concentration for each modality, plus template agreement.
    The final projection is zero initialized and is reset after backbone
    construction, so the disabled/initial candidate is an identity transform.
    """

    def __init__(self, hidden_dim=24, max_delta=0.25, topk=4):
        super().__init__()
        if hidden_dim < 8 or max_delta <= 0 or topk < 2:
            raise ValueError('invalid Early source evidence parameters')
        self.hidden_dim = int(hidden_dim)
        self.max_delta = float(max_delta)
        self.topk = int(topk)
        self.mlp = nn.Sequential(
            nn.Linear(11, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 3),
        )
        self.reset_identity()

    def reset_identity(self):
        with torch.no_grad():
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _distribution_stats(similarity, topk):
        # similarity: [B, Ntemplate, Nsearch]
        if similarity.ndim != 3 or similarity.shape[-1] < 2:
            raise ValueError('similarity must be [B,Ntemplate,Nsearch>=2]')
        values, _ = similarity.float().sort(dim=-1, descending=True)
        k = min(int(topk), values.shape[-1])
        top1 = values[..., 0]
        top2 = values[..., 1]
        topk_mean = values[..., :k].mean(dim=-1)
        return torch.stack((
            top1,
            top1 - top2,
            values.std(dim=-1, unbiased=False),
            values.mean(dim=-1),
            topk_mean,
        ), dim=-1)

    def forward(self, evidence, rgb_similarity, tir_similarity):
        if evidence.ndim != 3 or evidence.shape[-1] != 3:
            raise ValueError('evidence must have shape [B,Ntemplate,3]')
        if rgb_similarity.shape[:2] != evidence.shape[:2] or tir_similarity.shape[:2] != evidence.shape[:2]:
            raise ValueError('similarity and evidence token axes differ')
        rgb_stats = self._distribution_stats(rgb_similarity, self.topk)
        tir_stats = self._distribution_stats(tir_similarity, self.topk)
        features = torch.cat((rgb_stats, tir_stats, evidence[..., 2:3].float()), dim=-1)
        normalized = F.layer_norm(features, (features.shape[-1],))
        delta = self.max_delta * torch.tanh(self.mlp(normalized))
        return evidence + delta


def reset_early_source_evidence(model):
    """Reset all enabled evidence transforms after generic ViT initialization."""
    count = 0
    for module in model.modules():
        if isinstance(module, EarlySourceEvidence):
            module.reset_identity()
            count += 1
    return count
