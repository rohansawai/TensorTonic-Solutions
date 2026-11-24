import torch
import torch.nn as nn

def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """
    Returns the mean batch loss as a Python float.
    """
    model.train()
    loss_total = 0.0
    steps = 0
    for X, y in dataloader:
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss_total+=loss.item()
        steps+=1
        loss.backward()
        optimizer.step()
    return loss_total/steps