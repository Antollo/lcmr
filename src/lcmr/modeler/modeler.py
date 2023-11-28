import abc
import torch
import torch.nn as nn


class Modeler(nn.Module, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__()

    @abc.abstractmethod
    def forward(self, x: torch.Tensor):
        pass
