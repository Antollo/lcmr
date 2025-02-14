from dataclasses import dataclass
from functools import cache
from typing import Optional, Union

import numpy as np
import torch
from pyefd import elliptic_fourier_descriptors as pyefd_elliptic_fourier_descriptors
from sklearn.mixture import BayesianGaussianMixture
from torch_earcut import triangulate
from torchtyping import TensorType

from lcmr.utils.guards import batch_dim, object_dim, optional_dims, typechecked, vec_dim
from lcmr.utils.math import angle_to_rotation_matrix


@typechecked
def elliptic_fourier_descriptors(
    contour: TensorType[optional_dims:..., -1, 2, torch.float32], order: int = 10, normalize: bool = True
) -> TensorType[optional_dims:..., -1, 4, torch.float32]:
    # based on pyefd.elliptic_fourier_descriptors

    device = contour.device
    contour = contour[..., None, :, :]

    dxy = torch.diff(contour, dim=-2)
    dt = torch.sqrt((dxy**2).sum(dim=-1))
    t = torch.nn.functional.pad(torch.cumsum(dt, dim=-1), (1, 0))

    T = t[..., -1, None]
    phi = (2 * torch.pi * t) / T

    orders = torch.arange(1, order + 1, device=device)[None, :, None]

    consts = T / (2 * orders * orders * torch.pi * torch.pi)
    phi = phi * orders

    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    d_cos_phi = cos_phi[..., 1:] - cos_phi[..., :-1]
    d_sin_phi = sin_phi[..., 1:] - sin_phi[..., :-1]

    a = consts * torch.sum((dxy[..., 0] / dt) * d_cos_phi, axis=-1, keepdim=True)
    b = consts * torch.sum((dxy[..., 0] / dt) * d_sin_phi, axis=-1, keepdim=True)
    c = consts * torch.sum((dxy[..., 1] / dt) * d_cos_phi, axis=-1, keepdim=True)
    d = consts * torch.sum((dxy[..., 1] / dt) * d_sin_phi, axis=-1, keepdim=True)

    coeffs = torch.cat([a, b, c, d], dim=-1)

    if normalize:
        coeffs = normalize_efd(coeffs)

    return coeffs


@cache
@typechecked
def order_phases(order: int, n_points: int, device: torch.device):
    t = torch.linspace(0, 1.0, n_points, device=device)[None, ...]
    orders = torch.arange(1, order, device=device)[..., None]
    order_phases = 2 * torch.pi * orders * t
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
def normalize_efd(descriptors: TensorType[optional_dims:..., -1, 4, torch.float32]) -> TensorType[optional_dims:..., -1, 4, torch.float32]:
    # based on pyefd.normalize_efd
    device = descriptors.device
    shape = descriptors.shape
    descriptors = descriptors.reshape(-1, shape[-2], shape[-1])

    a, b, c, d = descriptors[..., 0, 0], descriptors[..., 0, 1], descriptors[..., 0, 2], descriptors[..., 0, 3]

    theta_1 = 0.5 * torch.arctan2(2 * (a * b + c * d), a**2 - b**2 + c**2 - d**2).nan_to_num(0, 0, 0)

    theta_array = (torch.arange(1, descriptors.shape[-2] + 1, dtype=torch.float32, device=device)[None, :] * theta_1[:, None])[..., None]
    theta_rotation_matrix = angle_to_rotation_matrix(-theta_array)

    descriptors = torch.matmul(descriptors.unflatten(dim=-1, sizes=(2, 2)), theta_rotation_matrix).flatten(-2, -1)

    psi_1 = torch.arctan2(descriptors[..., 0, 2], descriptors[..., 0, 0]).nan_to_num(0, 0, 0)[..., None]
    psi_rotation_matrix = angle_to_rotation_matrix(psi_1)[:, None, ...]

    descriptors = torch.matmul(psi_rotation_matrix, descriptors.unflatten(dim=-1, sizes=(2, 2))).flatten(-2, -1)

    size = descriptors[..., 0, 0]
    descriptors = descriptors / torch.abs(size)[:, None, None]

    descriptors = descriptors.reshape(*shape)

    return descriptors


@typechecked
def simplify_contour(contour: TensorType[object_dim, vec_dim, 2, torch.float32], threshold: float = 0.001) -> TensorType[object_dim, vec_dim, 1, torch.bool]:
    contour_padded = torch.nn.functional.pad(contour, (0, 0, 1, 1), "circular")

    a, b, c = contour_padded[:, :-2, :], contour_padded[:, 1:-1, :], contour_padded[:, 2:, :]
    ba = a - b
    bc = c - b
    cosine_angle = torch.bmm(ba.view(-1, 1, 2), bc.view(-1, 2, 1)).view(contour.shape[0], contour.shape[1]) / (torch.linalg.norm(ba, dim=-1) * torch.linalg.norm(bc, dim=-1))
    mask = cosine_angle[..., None] > -1 + threshold

    return mask


@typechecked
def triangulate_contour(
    contours: TensorType[batch_dim, object_dim, -1, 2, torch.float32], contour_only: bool = False, return_list: bool = False
) -> Union[TensorType[batch_dim, -1, -1, torch.int32], tuple[TensorType[batch_dim, -1, -1, torch.int32], list[TensorType[-1, -1, torch.int32]]]]:
    batch_len, object_len, vec_len, _ = contours.shape

    if contour_only:
        indices = torch.from_numpy(np.roll(np.repeat(np.arange(vec_len, dtype=np.int32), 2), -1))
        faces = [indices.clone() for _ in range(batch_len * object_len)]
    else:
        faces = triangulate(contours.flatten(0, 1).cpu().contiguous())
    faces = [faces[i : i + object_len] for i in range(0, len(faces), object_len)]
    for i in range(len(faces)):
        total = 0
        for indices in faces[i]:
            indices += total
            total += vec_len
        faces[i] = torch.cat(faces[i])
    faces_list = faces
    faces = torch.nn.utils.rnn.pad_sequence(faces, batch_first=True, padding_value=-1).view(batch_len, -1, 2 if contour_only else 3).to(contours.device)

    if return_list:
        faces_list = [f.view(-1, 3).to(contours.device) for f in faces_list]
        return faces, faces_list

    return faces


def gcd(v: torch.Tensor) -> torch.Tensor:
    while v.shape[1] > 1:
        n = v.shape[1]
        # If odd, leave the last element aside.
        if n % 2 == 1:
            last = v[:, -1].unsqueeze(1)
            v_even = v[:, :-1]
        else:
            last = None
            v_even = v
        # Pair up columns: new shape (B, n//2, 2)
        v_pairs = v_even.view(v.shape[0], -1, 2)
        # Compute elementwise gcd for each pair.
        v_reduced = torch.gcd(v_pairs[..., 0], v_pairs[..., 1])
        # If an element was left unpaired, concatenate it back.
        v = v_reduced if last is None else torch.cat([v_reduced, last], dim=1)
    return v.squeeze(1)


def compute_rotational_symmetry(efd: torch.Tensor, threshold: float = 0.075) -> torch.Tensor:
    # efd = efd[:, 1:]
    B, num_harmonics, _ = efd.shape
    # Compute L2 norm over the 4 coefficients (dim=2)
    norms = torch.norm(efd, p=2, dim=2)  # shape: (B, num_harmonics)
    # Create 1-indexed harmonic indices (same for every batch)
    indices = torch.arange(2, num_harmonics + 2, device=efd.device, dtype=torch.int64)
    indices = indices.unsqueeze(0).expand(B, num_harmonics)  # shape: (B, num_harmonics)
    # Create a mask of significant harmonics.
    mask = norms > threshold

    multiplier = ((torch.norm(efd[..., 0, 0:2], p=2, dim=-1) - torch.norm(efd[..., 0, 2:4], p=2, dim=-1)).abs() > threshold).to(torch.int64)
    # Zero out indices for insignificant harmonics.
    sig_indices = indices * mask.to(torch.int64)  # shape: (B, num_harmonics)
    sig_indices[..., 0] *= multiplier
    # Compute the GCD over each batch using our GPU-based tree reduction.
    gcd_vals = gcd(sig_indices)
    # If no significant harmonic was found (gcd==0), return 1.
    gcd_vals[gcd_vals == 0] = 1
    return gcd_vals


@typechecked
@dataclass
class EfdGeneratorOptions:
    order: int = 8
    n_points: int = 6
    irregularity: float = 0.8
    spikiness: float = 0.5
    choices: Optional[TensorType[object_dim, -1, 4, torch.float32]] = None
    use_gmm: bool = False


@typechecked
class EfdGenerator:
    def __init__(self, options: EfdGeneratorOptions):
        self.order = options.order
        self.n_points = options.n_points
        self.irregularity = options.irregularity
        self.spikiness = options.spikiness
        self.choices = options.choices
        self.use_gmm = options.use_gmm
        self.gmm = None

    def random_angles(self, n_objects: int) -> TensorType[object_dim, -1, 1, torch.float32]:
        low = (2 * np.pi / self.n_points) * (1.0 - self.irregularity)
        high = (2 * np.pi / self.n_points) * (1.0 + self.irregularity)
        angles = torch.rand((n_objects, self.n_points)) * (high - low) + low
        angles = angles / (angles.sum(axis=-1, keepdims=True) / (2 * np.pi))
        angles = angles.cumsum(axis=-1) + 2 * np.pi * torch.rand((n_objects, 1))
        return angles[..., None]

    def random_polygons(self, n_objects: int) -> TensorType[object_dim, -1, 2, torch.float32]:
        # polygon generation code inspired by https://stackoverflow.com/a/25276331/14344875
        angles = self.random_angles(n_objects)
        r = (torch.randn_like(angles) * self.spikiness + 1).clip(0.01, 2)
        x = torch.cos(angles)
        y = torch.sin(angles)
        contour = r * torch.cat((x, y), axis=-1)
        contour = torch.nn.functional.pad(contour, (0, 0, 1, 1), "circular")
        return contour

    def __sample(self, n_objects: int) -> TensorType[object_dim, -1, 4, torch.float32]:
        return torch.from_numpy(np.array([pyefd_elliptic_fourier_descriptors(x, order=self.order, normalize=True).astype(np.float32) for x in self.random_polygons(n_objects)]))

    def sample(self, n_objects: int = 1) -> TensorType[object_dim, -1, 4, torch.float32]:
        if self.choices != None:
            indices = torch.randint(low=0, high=len(self.choices), size=(n_objects,), dtype=torch.int32)
            return self.choices[indices]
        if self.use_gmm:
            if self.gmm == None:
                n_objects_X = 10_000
                X = self.__sample(n_objects_X)
                self.gmm = BayesianGaussianMixture(n_components=10, max_iter=500).fit(X.reshape(n_objects_X, -1))

            return torch.from_numpy(self.gmm.sample(n_samples=n_objects)[0].astype(np.float32).reshape(n_objects, self.order, 4))
        else:
            return self.__sample(n_objects)
