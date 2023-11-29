import abc
import torch
from torchtyping import TensorType

batch_dims = "batch_dims"
vec_dim = "vec_dim"


class Transformation(abc.ABC):
    @abc.abstractmethod
    def apply(self, vec: TensorType[batch_dims:..., vec_dim, torch.float32]) -> TensorType[batch_dims:..., vec_dim, torch.float32]:
        pass
