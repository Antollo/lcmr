import abc
import torch
from torchtyping import TensorType

from lcmr.utils.guards import optional_dims, vec_dim


class Transformation(abc.ABC):
    @abc.abstractmethod
    def apply(self, vec: TensorType[optional_dims:..., vec_dim, torch.float32]) -> TensorType[optional_dims:..., vec_dim, torch.float32]:
        pass
