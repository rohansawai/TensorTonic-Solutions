import torch

def reshape_tensor(x: torch.Tensor, op: str) -> torch.Tensor:
    """
    Returns the reshaped float32 tensor, including a scalar tensor after a complete squeeze.
    """
    if op == "flatten":
        return x.flatten()
    if op == "view":
        return x.view()
    if op == "squeeze":
        return x.squeeze()
    if op == "unsqueeze":
        return x.unsqueeze()
    if op == "transpose":
        return x.permute(1, 0)
