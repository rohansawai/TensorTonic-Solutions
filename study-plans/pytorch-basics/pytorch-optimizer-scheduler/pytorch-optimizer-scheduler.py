import torch
import torch.nn as nn

def train_with_scheduler(model: nn.Module, 
                         dataloader: torch.utils.data.DataLoader, 
                         criterion: nn.Module, 
                         optimizer: torch.optim.Optimizer, 
                         scheduler: torch.optim.lr_scheduler.StepLR, 
                         num_epochs: int) -> dict:
    """
    Returns losses and lrs as lists of Python floats in a dictionary.
    """
    model.train()
    losses = []
    lrs = []
    for epoch in range(num_epochs):
        batch_loss = 0
        for X, y in dataloader:
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            batch_loss += loss.item()
            loss.backward()
            optimizer.step()
        losses.append(batch_loss/len(dataloader))
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    return {"losses": losses, "lrs": lrs}
            
            
            
        
    
