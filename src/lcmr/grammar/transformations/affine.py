import torch
from torchtyping import TensorType

from lcmr.utils.guards import checked_tensorclass, typechecked, batch_dim, layer_dim, object_dim
from lcmr.grammar.transformations.transformation import Transformation, optional_dims, vec_dim
from lcmr.grammar.transformations.utils import matrix3x3_from_tensors


@checked_tensorclass
class Affine(Transformation):
    matrix: TensorType[batch_dim, layer_dim, object_dim, vec_dim, vec_dim, torch.float32]

    @staticmethod
    @typechecked
    def from_tensors(
        translation: TensorType[batch_dim, layer_dim, object_dim, 2, torch.float32],
        scale: TensorType[batch_dim, layer_dim, object_dim, 2, torch.float32],
        angle: TensorType[batch_dim, layer_dim, object_dim, 1, torch.float32],
    ) -> "Affine":
        # TODO: support 4x4 matrices?
        batch_len, layer_len, object_len, _ = translation.shape
        return Affine(batch_size=[batch_len, layer_len, object_len], matrix=matrix3x3_from_tensors(translation, scale, angle))

    def apply(self, vec: TensorType[optional_dims:..., vec_dim, torch.float32]) -> TensorType[optional_dims:..., vec_dim, torch.float32]:
        return NotImplementedError()
