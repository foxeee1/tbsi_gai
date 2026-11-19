"""Parameter-free per-direction centering of source-token log gains."""
import torch


def center_source_log_gain(log_gain):
    """Remove each modality direction's token-mean log gain.

    Args:
        log_gain: Tensor shaped [batch, source_token, direction].
    """
    if log_gain.ndim != 3 or log_gain.shape[-1] != 2:
        raise ValueError('Expected log gains shaped [batch, source_token, 2]')
    return log_gain - log_gain.mean(dim=1, keepdim=True)
