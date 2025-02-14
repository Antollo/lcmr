from typing import Sequence, Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchtyping import TensorType

from lcmr.grammar.object import Object
from lcmr.utils.guards import batch_dim, layer_dim, object_dim, typechecked


def linear_sum_assignment_vectorized(cost):
    b, h, w = cost.shape
    d = min(h, w)

    all_ind_col = np.arange(w, dtype=np.int64)
    all_ind_row = np.arange(h, dtype=np.int64)

    row_ind_list = np.zeros(shape=(b, h), dtype=np.int64)
    col_ind_list = np.zeros(shape=(b, w), dtype=np.int64)
    weight_list = np.zeros(shape=(b, w), dtype=np.float32)
    
    for i in range(len(cost)):
        last_row = np.nonzero(np.isnan(cost[i]))
        if len(last_row) != 0 and len(last_row[0]) != 0:
            last_row = last_row[0][0]
        else:
            last_row = h
        row_ind, col_ind = linear_sum_assignment(cost[i, :last_row, :])

        row_ind_list[i, : min(last_row, d)] = row_ind
        row_ind_list[i, min(last_row, d) :] = np.setdiff1d(all_ind_row, row_ind, assume_unique=True)

        col_ind_list[i, :last_row] = col_ind
        col_ind_list[i, last_row:] = np.setdiff1d(all_ind_col, col_ind)
        weight_list[i, :last_row] = 1.0
    return row_ind_list, col_ind_list, weight_list


object_dim_a = "object_dim_a"
object_dim_b = "object_dim_b"


@typechecked
@torch.no_grad()
def match(*data: tuple[float, TensorType[batch_dim, layer_dim, object_dim_a, -1], TensorType[batch_dim, layer_dim, object_dim_b, -1]]) -> tuple[
    TensorType[batch_dim, layer_dim, object_dim_a, 1, torch.int64],
    TensorType[batch_dim, layer_dim, object_dim_b, 1, torch.int64],
    TensorType[batch_dim, layer_dim, object_dim_b, 1, torch.float32],
]:
    batch_len, layer_len, object_len_a, _ = data[0][1].shape
    _, _, object_len_b, _ = data[0][2].shape

    assert object_len_b >= object_len_a

    device = data[0][1].device
    cost = torch.sum(torch.cat([w * torch.cdist(a.flatten(0, 1), b.flatten(0, 1))[None, ...] for w, a, b in data], dim=0), dim=0).cpu()

    indices_a, indices_b, weights = linear_sum_assignment_vectorized(cost.numpy())
    indices_a = torch.from_numpy(indices_a[..., None]).unflatten(0, (batch_len, layer_len)).to(device)
    indices_b = torch.from_numpy(indices_b[..., None]).unflatten(0, (batch_len, layer_len)).to(device)
    weights = torch.from_numpy(weights[..., None]).unflatten(0, (batch_len, layer_len)).to(device)

    return indices_a, indices_b, weights


@typechecked
def gather(
    indices: TensorType[batch_dim, layer_dim, object_dim, 1, torch.int64], *args: Union[TensorType[batch_dim, layer_dim, object_dim, -1], Object]
) -> Union[Sequence[Union[TensorType[batch_dim, layer_dim, object_dim, -1], Object]], TensorType[batch_dim, layer_dim, object_dim, -1], Object]:
    result = [torch.gather(x, 2, torch.broadcast_to(indices if x.ndim == indices.ndim else indices.squeeze(-1), x.shape)) for x in args]
    if len(result) == 1:
        return result[0]
    return result
