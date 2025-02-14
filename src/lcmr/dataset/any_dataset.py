import torch
from torch.utils.data import Dataset
from typing import Any

from lcmr.grammar.scene_data import SceneData


class AnyDataset(Dataset):
    def __init__(self, data: Any):
        self.data = data
        
    def __getitem__(self, idx: int) -> SceneData:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        if type(batch[0]) is tuple:
            batch = (tuple(x) for x in zip(*batch))
            return tuple(torch.cat(x, dim=0) for x in batch)
        else:
            return torch.cat(batch, dim=0)
