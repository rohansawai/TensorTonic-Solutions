import torch

def gradient_accumulation(w_init: torch.Tensor, micro_batches: list, lr: float, accum_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (final_weights, last_mean_gradient) as float32 tensors.
    """
    w = w_init.clone().requires_grad_(True)
    mean_grad = 0.0
    for i, (X, t) in enumerate(micro_batches):
        pred = w @ X
        loss = (pred - t) ** 2
        loss.backward()

        if (i+1) % accum_steps == 0:
            mean_grad = w.grad/accum_steps
            with torch.no_grad():
                w -= lr * mean_grad        
            w.grad.zero_()

    return w, mean_grad