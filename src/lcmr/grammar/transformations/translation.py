import torch
from torchtyping import TensorType

from lcmr.grammar.transformations import Transformation
from lcmr.utils.guards import checked_tensorclass, batch_dim, layer_dim, object_dim, optional_dims, vec_dim


@checked_tensorclass
class Translation(Transformation):
    vec: TensorType[batch_dim, layer_dim, object_dim, vec_dim, torch.float32]

    def apply(self, vec: TensorType[optional_dims:..., vec_dim, torch.float32]) -> TensorType[optional_dims:..., vec_dim, torch.float32]:
        return NotImplementedError
