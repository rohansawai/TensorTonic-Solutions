import torch
import torch.nn as nn

def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """
    Returns the mean batch loss as a Python float.
    """
    model.train()
    loss_total = []
    for X, y in dataloader:
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss_total.append(loss.item())
        loss.backward()
        optimizer.step()åç
    return sum(loss_total)/len(loss_total)