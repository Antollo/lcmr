import torch
from torchtyping import TensorType
from ..guards import checked_tensorclass
from .transformation import Transformation, batch_dims, vec_dim


@checked_tensorclass
class Identity(Transformation):
    def apply(self, vec: TensorType[batch_dims:..., vec_dim, torch.float32]) -> TensorType[batch_dims:..., vec_dim, torch.float32]:
        return vec
