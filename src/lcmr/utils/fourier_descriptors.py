import torch
from shapely import Polygon, centroid
from torchinterp1d import interp1d
from torchtyping import TensorType

from lcmr.utils.guards import optional_dims, typechecked
from lcmr.utils.math import angle_to_rotation_matrix


@typechecked
def cart_to_polar(
    contour: TensorType[optional_dims:..., -1, 2, torch.float32], n_points: int = 128
) -> tuple[TensorType[optional_dims:..., -1, torch.float32], TensorType[optional_dims:..., -1, torch.float32]]:

    device = contour.device

    center = [centroid(Polygon(c)) for c in contour.detach().cpu().numpy()]
    center = torch.tensor([(c.x, c.y) for c in center], device=device)[:, None, :]

    contour = contour - center

    dxy = torch.diff(contour, dim=-2)
    dt = torch.norm(dxy, dim=-1, p=2)

    t = torch.nn.functional.pad(torch.cumsum(dt, dim=-1), (1, 0))
    t = t / t[..., -1, None]

    new_t = torch.linspace(0, 1, n_points, device=device)[None]

    contour = torch.cat((interp1d(t, contour[..., 0], new_t)[..., None], interp1d(t, contour[..., 1], new_t)[..., None]), dim=-1)

    rho = torch.norm(contour, p=2, dim=-1)
    phi = torch.arctan2(contour[..., 1], contour[..., 0])

    idx = torch.argsort(phi, dim=-1)
    rho = torch.gather(rho, -1, idx)
    phi = torch.gather(phi, -1, idx)

    # make equal phi steps
    new_phi = torch.linspace(-torch.pi, torch.pi, n_points, device=device)[None]
    rho = interp1d(phi, rho, new_phi)
    phi = torch.broadcast_to(new_phi, phi.shape)
    return rho, phi


def rot_matrix(descriptors: TensorType[optional_dims:..., -1, 2, torch.float32]) -> TensorType[optional_dims:..., 2, 2, torch.float32]:
    device = descriptors.device
    shape = descriptors.shape
    descriptors = descriptors.reshape(-1, shape[-2], shape[-1])

    k_vals = torch.arange(1, descriptors.shape[-2] + 1, dtype=torch.float32, device=device)[None, :]

    a, b = descriptors[..., 0], descriptors[..., 1]

    norm = descriptors.norm(dim=-1, p=2) ** 0.5
    norm = norm / norm.mean(dim=-1, keepdim=True)

    phi = torch.arctan2(b, a) / k_vals
    phi = (phi).mean(dim=-1)

    phi_array = (k_vals * phi[:, None])[..., None]
    phi_rotation_matrix = angle_to_rotation_matrix(phi_array)

    return phi_rotation_matrix


@typechecked
def fourier_descriptors(
    contour: TensorType[optional_dims:..., -1, 2, torch.float32], n_points: int = 256, order: int = 32, normalize: bool = True
) -> TensorType[optional_dims:..., -1, 2, torch.float32]:

    rho, _ = cart_to_polar(contour, n_points=n_points)
    rho = rho[..., None]

    device = rho.device
    rho = rho[..., None, :, :]
    n_points = rho.shape[-2]

    rho = rho / rho.mean(dim=(1, 2, 3), keepdim=True)

    d_rho = torch.diff(rho, dim=-2)
    t = torch.linspace(0, 1, n_points)[None, None]

    phi = 2 * torch.pi * t
    orders = torch.arange(1, order + 1, device=device)[None, :, None]

    consts = 1 / (2 * orders * orders * torch.pi * torch.pi)
    phi = phi * orders

    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    d_cos_phi = cos_phi[..., 1:] - cos_phi[..., :-1]
    d_sin_phi = sin_phi[..., 1:] - sin_phi[..., :-1]

    a = consts * torch.sum((d_rho[..., 0]) * d_cos_phi * n_points, axis=-1, keepdim=True)
    b = consts * torch.sum((d_rho[..., 0]) * d_sin_phi * n_points, axis=-1, keepdim=True)

    fd = torch.cat([a, b], dim=-1)

    if normalize:
        fd = normalize_fourier_descriptors(fd)

    return fd


# @cache
@typechecked
def order_phases(order: int, n_points: int, device: torch.device, a):
    t = torch.linspace(0, 1.0, n_points, device=device)[None, None, ...] - a / (2 * torch.pi)
    # print(t.shape, a.shape if type(a) != int else 1, (t+a).shape)
    orders = torch.arange(1, order, device=device)[None, ..., None]
    order_phases = 2 * torch.pi * orders * t
    # order_phases = order_phases[None, ...]

    return torch.cos(order_phases), torch.sin(order_phases)


@typechecked
def normalize_fourier_descriptors(descriptors: TensorType[optional_dims:..., -1, 2, torch.float32]) -> TensorType[optional_dims:..., -1, 2, torch.float32]:
    return torch.matmul(descriptors[..., None, :], rot_matrix(descriptors))[..., 0, :]


@typechecked
def reconstruct_rho(descriptors: TensorType[optional_dims:..., -1, 2, torch.float32], n_points=128, a=0) -> TensorType[optional_dims:..., -1, torch.float32]:

    device = descriptors.device
    descriptors = descriptors[..., None]

    order_phases_cos, order_phases_sin = order_phases(descriptors.shape[-3] + 1, n_points, device, a)

    rho = descriptors[..., 0, :] * order_phases_cos + descriptors[..., 1, :] * order_phases_sin
    rho = rho.sum(axis=-2) + 1

    return rho
