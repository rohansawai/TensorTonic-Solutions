import torch

class CustomSGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0):
        defaults = {"lr": lr, "momentum": momentum}
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        """
        Returns the closure loss when provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "mom_buf" in self.state[p]:
                    v = self.state[p]["mom_buf"]
                else:
                    v = torch.zeros_like(p, dtype = torch.float32)
                v = group["momentum"] * v + p.grad
                p -= group["lr"] * v
                self.state[p]["mom_buf"] = v
        return loss
                
