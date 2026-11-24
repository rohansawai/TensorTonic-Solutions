import torch

def softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 probability tensor with the same shape as logits.
    """
    shifted = logits - logits.max(dim=1, keepdim=True).values
    
    return torch.exp(shifted)/torch.exp(shifted).sum(dim=1, keepdim=True)
