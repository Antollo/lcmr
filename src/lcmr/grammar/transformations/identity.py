import torch
from torchtyping import TensorType

from lcmr.grammar.transformations import Transformation
from lcmr.utils.guards import checked_tensorclass, optional_dims, vec_dim


@checked_tensorclass
class Identity(Transformation):
    def apply(self, vec: TensorType[optional_dims:..., vec_dim, torch.float32]) -> TensorType[optional_dims:..., vec_dim, torch.float32]:
        return vec
