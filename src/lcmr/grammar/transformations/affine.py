import torch
from torchtyping import TensorType
from ..guards import checked_tensorclass, batch_dim, layer_dim, object_dim
from .transformation import Transformation, batch_dims, vec_dim


@checked_tensorclass
class Affine(Transformation):
    matrix: TensorType[batch_dim, layer_dim, object_dim, vec_dim, vec_dim, torch.float32]

    def apply(self, vec: TensorType[batch_dims:..., vec_dim, torch.float32]) -> TensorType[batch_dims:..., vec_dim, torch.float32]:
        return NotImplementedError
