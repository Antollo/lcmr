import torch
from torchtyping import TensorType
from ...utils.guards import checked_tensorclass
from .transformation import Transformation, optional_dims, vec_dim


@checked_tensorclass
class Identity(Transformation):
    def apply(self, vec: TensorType[optional_dims:..., vec_dim, torch.float32]) -> TensorType[optional_dims:..., vec_dim, torch.float32]:
        return vec
