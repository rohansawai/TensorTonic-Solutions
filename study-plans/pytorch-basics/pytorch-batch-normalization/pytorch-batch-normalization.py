import torch

def batch_norm(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as X.
    """
    muj  = X.mean(dim=0)
    sigmaSq = ((X-muj)**2).mean(dim=0)
    Y = gamma * (X-muj)/torch.sqrt(sigmaSq+eps) + beta
    return Y
