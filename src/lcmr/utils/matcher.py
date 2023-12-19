import torch
import numpy as np
from typing import Sequence
from torchtyping import TensorType
from scipy.optimize import linear_sum_assignment

from lcmr.utils.guards import typechecked, batch_dim, layer_dim, object_dim, channel_dim

linear_sum_assignment_vectorized = np.vectorize(linear_sum_assignment, signature="(m,m)->(m),(m)")


@typechecked
class Matcher:
    @torch.no_grad()
    def match(
        self, a: TensorType[batch_dim, layer_dim, object_dim, channel_dim], b: TensorType[batch_dim, layer_dim, object_dim, channel_dim]
    ) -> tuple[TensorType[batch_dim, layer_dim, object_dim, 1, torch.int64], TensorType[batch_dim, layer_dim, object_dim, 1, torch.int64]]:
        batch_len, layer_len, _, _ = a.shape
        device = a.device

        cost = torch.cdist(a.flatten(0, 1), b.flatten(0, 1)).cpu()
        indices = torch.from_numpy(np.concatenate([x[None, ...] for x in linear_sum_assignment_vectorized(cost)])).to(device)
        indices_a = indices[0, :, :, None].unflatten(0, (batch_len, layer_len))
        indices_b = indices[1, :, :, None].unflatten(0, (batch_len, layer_len))
        return indices_a, indices_b

    def gather(
        self, indices: TensorType[batch_dim, layer_dim, object_dim, 1, torch.int64], args: Sequence[TensorType[batch_dim, layer_dim, object_dim, -1]]
    ) -> Sequence[TensorType[batch_dim, layer_dim, object_dim, -1]]:
        return [torch.gather(x, 2, torch.broadcast_to(indices, x.shape)) for x in args]
