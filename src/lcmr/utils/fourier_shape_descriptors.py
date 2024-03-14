import torch
import numpy as np
from torchtyping import TensorType
from functools import cache
from collections.abc import Iterable
from mapbox_earcut import triangulate_float32

from lcmr.utils.guards import typechecked, object_dim, vec_dim, optional_dims


@cache
@typechecked
def order_phases(n_orders: int, n_points: int, device: torch.device):
    t = torch.linspace(0, 1.0, n_points, device=device)[None, ...]
    orders = torch.arange(1, n_orders, device=device)[..., None]
    order_phases = 2 * np.pi * orders * t
    order_phases = order_phases[None, ...]

    return torch.cos(order_phases), torch.sin(order_phases)


@typechecked
def reconstruct_contour(descriptors: TensorType[optional_dims:..., -1, 4, torch.float32], n_points=64) -> TensorType[optional_dims:..., -1, 2, torch.float32]:
    # based on pyefd.reconstruct_contour

    device = descriptors.device
    descriptors = descriptors[..., None]

    order_phases_cos, order_phases_sin = order_phases(descriptors.shape[-3] + 1, n_points, device)

    xt_all = descriptors[..., 0, :] * order_phases_cos + descriptors[..., 1, :] * order_phases_sin
    yt_all = descriptors[..., 2, :] * order_phases_cos + descriptors[..., 3, :] * order_phases_sin

    xt_all = xt_all.sum(axis=-2)
    yt_all = yt_all.sum(axis=-2)

    reconstruction = torch.stack((xt_all, yt_all), axis=-1)
    return reconstruction


@typechecked
def simplify_contour(contour: TensorType[object_dim, vec_dim, 2, torch.float32], threshold: float = 0.001) -> TensorType[object_dim, vec_dim, 1, torch.bool]:
    contour_padded = torch.nn.functional.pad(contour, (0, 0, 1, 1), "circular")

    a, b, c = contour_padded[:, :-2, :], contour_padded[:, 1:-1, :], contour_padded[:, 2:, :]
    ba = a - b
    bc = c - b
    cosine_angle = torch.bmm(ba.view(-1, 1, 2), bc.view(-1, 2, 1)).view(contour.shape[0], contour.shape[1]) / (
        torch.linalg.norm(ba, dim=-1) * torch.linalg.norm(bc, dim=-1)
    )
    mask = cosine_angle[..., None] > -1 + threshold

    return mask


@typechecked
def triangularize_contour(
    contours: Iterable[TensorType[-1, 2, torch.float32]] | TensorType[-1, -1, 2, torch.float32], contour_only: bool = False
) -> np.ndarray:
    faces_list = []
    total = 0
    for contour in contours:
        if contour_only:
            # just contour
            faces = np.roll(np.repeat(np.arange(len(contour), dtype=np.int32), 2), -1)
        else:
            # real triangularisation
            faces = triangulate_float32(contour.detach().cpu().numpy(), np.array([len(contour)])).astype(np.int32)

        faces_list.append(faces + total)
        total += contour.shape[0]

    return np.concatenate(faces_list).reshape(-1, 2 if contour_only else 3)