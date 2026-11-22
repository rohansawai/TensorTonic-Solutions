import torch

def compute_loss(pred: torch.Tensor, target: torch.Tensor, method: str, delta: float = 1.0) -> float:
    """
    Returns the mean loss as a Python float.
    """
    if method == "mse":
        a = torch.abs(pred-target)
        return (a**2).mean().item()
    if method == "cross_entropy":
        shifted = pred - pred.max(dim=1, keepdim=True).values
        x = torch.exp(shifted).sum(dim=1).log()
        index = torch.arange(pred.shape[0])
        y = shifted[index, target.to(dtype=torch.long)]
        print(index, x, y)
        return (x-y).mean().item()
    else:
        a = torch.abs(pred-target)
        return torch.where( a <= delta, (a**2)*(1/2), delta*(a-delta/2)).mean().item()
