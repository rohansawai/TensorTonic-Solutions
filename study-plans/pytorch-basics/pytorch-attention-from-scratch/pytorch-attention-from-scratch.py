import torch
import math
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 attention tensor of shape (batch, seq_q, d_v).
    """
    scores = Q@K.transpose(-1, -2)/math.sqrt(K.shape[-1])
    weights = torch.softmax(scores, dim=-1)
    return weights@V
