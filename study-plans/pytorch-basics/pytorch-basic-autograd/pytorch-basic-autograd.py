import torch

def compute_gradient(x: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 gradient tensor with the same shape as values.
    """
    x.requires_grad_()
    y = torch.sum(x**3 + 2*x)
    y.backward()
    return x.grad