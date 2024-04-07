import torch
from torchtyping import TensorType

from lcmr.utils.guards import typechecked, optional_dims


@typechecked
def angle_to_rotation_matrix(angle: TensorType[optional_dims:..., 1, torch.float32]) -> TensorType[optional_dims:..., 2, 2, torch.float32]:
    # based on kornia.geometry.conversions.angle_to_rotation_matrix
    # removed deg to rad conversion

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    return torch.stack([cos, sin, -sin, cos], dim=-1).view(*angle.shape[:-1], 2, 2)
