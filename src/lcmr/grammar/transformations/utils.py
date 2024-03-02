import torch
from torchtyping import TensorType
from kornia.geometry.transform import get_affine_matrix2d
from math import pi, prod

from lcmr.utils.guards import typechecked, optional_dims


@typechecked
def matrix3x3_from_tensors(
    translation: TensorType[optional_dims:..., 2, torch.float32],
    scale: TensorType[optional_dims:..., 2, torch.float32],
    angle: TensorType[optional_dims:..., 1, torch.float32],
) -> TensorType[optional_dims:..., 3, 3, torch.float32]:
    shape = translation.shape[:-1]
    device = translation.device

    total_objects = prod(shape)
    center = torch.zeros(total_objects, 2, device=device)
    matrix = get_affine_matrix2d(translation.reshape(-1, 2), center.reshape(-1, 2), scale.reshape(-1, 2), angle.reshape(-1) / pi * 180).reshape(*shape, 3, 3)
    return matrix
