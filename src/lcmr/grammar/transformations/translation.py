import torch
from torchtyping import TensorType
from ...utils.guards import checked_tensorclass, batch_dim, layer_dim, object_dim
from .transformation import Transformation, optional_dims, vec_dim


@checked_tensorclass
class Translation(Transformation):
    vec: TensorType[batch_dim, layer_dim, object_dim, vec_dim, torch.float32]

    def apply(self, vec: TensorType[optional_dims:..., vec_dim, torch.float32]) -> TensorType[optional_dims:..., vec_dim, torch.float32]:
        return NotImplementedError
