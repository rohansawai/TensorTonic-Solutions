import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, data: list, label_col: int):
        t = torch.tensor(data, dtype=torch.float32)
        cols = [j for j in range(t.shape[1]) if j != label_col]
        self.X = t[:, cols]
        self.y = t[:, label_col].unsqueeze(1)
        self.rows = len(self.y)
    def __len__(self) -> int:
        """
        Returns the number of rows.
        """
        return self.rows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (features, label) as float32 tensors of shapes (D,) and (1,).
        """
        return self.X[idx], self.y[idx]
