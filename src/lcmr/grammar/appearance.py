import torch
from torchtyping import TensorType
from lcmr.utils.guards import checked_tensorclass, batch_dim, layer_dim, object_dim


@checked_tensorclass
class Appearance:
    confidence: TensorType[batch_dim, layer_dim, object_dim, 1, torch.float32]
    color: TensorType[batch_dim, layer_dim, object_dim, 3, torch.float32]
