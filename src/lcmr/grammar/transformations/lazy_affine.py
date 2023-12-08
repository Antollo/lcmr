import torch
from torchtyping import TensorType

from ...utils.guards import checked_tensorclass, typechecked, batch_dim, layer_dim, object_dim
from .transformation import Transformation, optional_dims, vec_dim
from .utils import matrix3x3_from_tensors


# TODO: support 4x4 matrices?
@checked_tensorclass
class LazyAffine(Transformation):
    translation: TensorType[batch_dim, layer_dim, object_dim, 2, torch.float32]
    scale: TensorType[batch_dim, layer_dim, object_dim, 2, torch.float32]
    angle: TensorType[batch_dim, layer_dim, object_dim, 1, torch.float32]

    @staticmethod
    @typechecked
    def from_tensors(
        translation: TensorType[batch_dim, layer_dim, object_dim, 2, torch.float32],
        scale: TensorType[batch_dim, layer_dim, object_dim, 2, torch.float32],
        angle: TensorType[batch_dim, layer_dim, object_dim, 1, torch.float32],
    ) -> "LazyAffine":
        batch_len, layer_len, object_len, _ = translation.shape
        return LazyAffine(batch_size=[batch_len, layer_len, object_len], translation=translation, scale=scale, angle=angle)

    @property
    @typechecked
    def matrix(self) -> TensorType[optional_dims:..., vec_dim, vec_dim, torch.float32]:
        return matrix3x3_from_tensors(translation=self.translation, scale=self.scale, angle=self.angle)

    def apply(self, vec: TensorType[optional_dims:..., vec_dim, torch.float32]) -> TensorType[optional_dims:..., vec_dim, torch.float32]:
        return NotImplementedError()
