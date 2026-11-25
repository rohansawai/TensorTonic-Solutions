import torch
import torch.nn as nn

def train_with_early_stopping(
    model: nn.Module, 
    train_loader: torch.utils.data.DataLoader, 
    val_loader: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    max_epochs: int, 
    patience: int
) -> dict:
    """
    Returns train_losses and val_losses as float lists, and stopped_epoch as an int.
    """
    train_loss = []
    val_loss = []
    best_loss = float('inf')
    bad_epoch = 0
    e = 0
    for epoch in range(max_epochs):
        e = epoch
        #training pass
        t_loss = 0.0
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(y, pred)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        train_loss.append(t_loss/len(train_loader))
        
        
        #validation loss
        with torch.no_grad():
            model.eval()
            v_loss = 0.0
            for X, y in val_loader:
                pred = model(X)
                loss = criterion(y, pred)
                v_loss += loss.item()
        val_loss.append(v_loss/len(val_loader))


        
        if val_loss[len(val_loss)-1] < best_loss:
            best_loss = loss
            bad_epoch = 0
        else:
            bad_epoch += 1
            if bad_epoch == patience:
                break

    return {"train_losses": train_loss,
            "val_losses": val_loss,
            "stopped_epoch": e+1}