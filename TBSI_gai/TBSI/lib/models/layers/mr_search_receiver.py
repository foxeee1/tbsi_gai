"""Search-receiver topology for the fixed-gain MR parent.

Design prototype, not connected to a tracker configuration. The caller supplies
the existing last fusion module; this module registers no parameters. Template
source context/evidence must follow the archived MR contract. Disabling the
feature must select the original host branch, not set an output gate to zero.
"""

import torch
from torch import nn
from torch.nn import functional as F


class MRSearchReceiver(nn.Module):
    """Replace the last template write with template-conditioned search writes.

    ``fusion`` owns the original norm/qkv/proj/dropouts and both MR allocators.
    Passing it at call time avoids a second registered copy of its state.
    Source axis 0 is RGB; source axis 1 is TIR. Search tokens are receivers,
    never keys or values in this operator. There is no search-to-search map.

    query_chunk_size limits temporary tensors; under autograd the retained
    backward activations still grow with the number of queries. Leave it zero
    for the simplest training reference. Chunking is not a speed guarantee.
    """

    def __init__(self, query_chunk_size=0):
        super().__init__()
        if query_chunk_size < 0:
            raise ValueError('query_chunk_size must be nonnegative')
        self.query_chunk_size = int(query_chunk_size)

    def forward(self, template_rgb, template_tir, search_rgb, search_tir, *,
                fusion, source_context, source_evidence):
        """Return (template_rgb, template_tir, updated_search_rgb, updated_search_tir).

        Templates: [B,Nt,D]; searches: [B,Ns,D]. Context: [B,Nt,64,2],
        evidence: [B,Nt,3]. Canonical evidence has RGB support first and TIR
        support second, as in the historical fixed-g=1 MR. Context/evidence
        are computed by the unchanged existing early-context path, outside
        this operator; they are not new predictors.
        """
        if template_rgb.ndim != 3 or search_rgb.ndim != 3:
            raise ValueError('expected [B,N,D] template and search tensors')
        b, nt, dim = template_rgb.shape
        ns = search_rgb.shape[1]
        if template_tir.shape != template_rgb.shape:
            raise ValueError('RGB/TIR template shapes must match')
        if search_tir.shape != search_rgb.shape or search_rgb.shape != (b, ns, dim):
            raise ValueError('RGB/TIR search shapes must match template batch/width')
        context_dim = fusion.mass_reallocator_rgb.context_dim
        if source_context.shape != (b, nt, context_dim, 2):
            raise ValueError('context must retain template source axes')
        if source_evidence.shape != (b, nt, 3):
            raise ValueError('evidence must retain template source axes')
        heads = fusion.num_heads
        if dim % heads or fusion.qkv.weight.shape != (3 * dim, dim):
            raise ValueError('incompatible existing qkv projection')
        # The chosen parent has zero attention/projection dropout. Reject other
        # settings until a separate random-mask contract has been specified.
        if fusion.attn_drop.p != 0 or fusion.proj_drop.p != 0:
            raise ValueError('this prototype requires the fixed-parent zero dropouts')
        head_dim = dim // heads
        bias = fusion.qkv.bias

        # Slice the SAME qkv parameter. Do not create new Linear modules, compute
        # template queries, or project search keys/values that will be discarded.
        templates = fusion.norm(torch.cat((template_rgb, template_tir), dim=1))
        kv = F.linear(templates, fusion.qkv.weight[dim:],
                      None if bias is None else bias[dim:])
        kv = kv.reshape(b, 2 * nt, 2, heads, head_dim).permute(2, 0, 3, 1, 4)
        keys, values = kv.unbind(0)
        searches = fusion.norm(torch.cat((search_rgb, search_tir), dim=1))
        queries = F.linear(searches, fusion.qkv.weight[:dim],
                           None if bias is None else bias[:dim])
        queries = queries.reshape(b, 2 * ns, heads, head_dim).transpose(1, 2)

        def write(query, residual, same_source, cross_source, allocator, context,
                  evidence):
            chunk = self.query_chunk_size or ns
            outputs = []
            same_values = values[:, :, same_source]
            cross_values = values[:, :, cross_source]
            for start in range(0, ns, chunk):
                stop = min(start + chunk, ns)
                logits = (query[:, :, start:stop] @ keys.transpose(-2, -1)) * fusion.scale
                attention = fusion.attn_drop(logits.softmax(dim=-1))
                same_attention = attention[:, :, :, same_source]
                cross_attention = attention[:, :, :, cross_source]
                same_message = same_attention @ same_values
                receiver_reference = same_message.transpose(1, 2).reshape(b, stop - start, dim)
                relation = allocator(cross_attention, receiver_reference, cross_values,
                                     context, evidence)
                cross_message = relation @ cross_values
                message = (same_message + cross_message).transpose(1, 2).reshape(b, stop - start, dim)
                outputs.append(residual[:, start:stop] +
                               fusion.proj_drop(fusion.proj(message)))
            return torch.cat(outputs, dim=1)

        rgb_source, tir_source = slice(0, nt), slice(nt, 2 * nt)
        updated_rgb = write(
            queries[:, :, :ns], search_rgb, rgb_source, tir_source,
            fusion.mass_reallocator_rgb, source_context[..., 1],
            source_evidence[..., [1, 0, 2]])
        updated_tir = write(
            queries[:, :, ns:], search_tir, tir_source, rgb_source,
            fusion.mass_reallocator_tir, source_context[..., 0], source_evidence)
        return template_rgb, template_tir, updated_rgb, updated_tir
