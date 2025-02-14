from typing import Optional
import torch
from torchtyping import TensorType

from lcmr.grammar.transformations import Transformation
from lcmr.utils.guards import checked_tensorclass, typechecked, batch_dim, layer_dim, object_dim, optional_dims, vec_dim
from lcmr.grammar.transformations.utils import matrix3x3_from_tensors


# TODO: support 4x4 matrices?
@checked_tensorclass
class LazyAffine(Transformation):
    translation: TensorType[batch_dim, layer_dim, object_dim, 2, torch.float32]
    scale: TensorType[batch_dim, layer_dim, object_dim, 2, torch.float32]
    angle: Optional[TensorType[batch_dim, layer_dim, object_dim, 1, torch.float32]]
    rotation_vec: Optional[TensorType[batch_dim, layer_dim, object_dim, 2, torch.float32]]

    @staticmethod
    @typechecked
    def from_tensors(
        translation: TensorType[batch_dim, layer_dim, object_dim, 2, torch.float32],
        scale: TensorType[batch_dim, layer_dim, object_dim, 2, torch.float32],
        angle: Optional[TensorType[batch_dim, layer_dim, object_dim, 1, torch.float32]] = None,
        rotation_vec: Optional[TensorType[batch_dim, layer_dim, object_dim, 2, torch.float32]] = None,
    ) -> "LazyAffine":
        batch_len, layer_len, object_len, _ = translation.shape
        assert angle != None or rotation_vec != None
        return LazyAffine(batch_size=[batch_len, layer_len, object_len], translation=translation, scale=scale, angle=angle, rotation_vec=rotation_vec)

    @property
    @typechecked
    def matrix(self) -> TensorType[optional_dims:..., vec_dim, vec_dim, torch.float32]:
        if self.rotation_vec != None:
            angle = torch.atan2(self.rotation_vec[..., 0, None], self.rotation_vec[..., 1, None])
        else:
            angle = self.angle
        return matrix3x3_from_tensors(translation=self.translation, scale=self.scale, angle=angle)

    def apply(self, vec: TensorType[optional_dims:..., vec_dim, torch.float32]) -> TensorType[optional_dims:..., vec_dim, torch.float32]:
        return NotImplementedError()
