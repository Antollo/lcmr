import torch
from torchtyping import TensorType
from lcmr.grammar.object import Object
from lcmr.utils.guards import checked_tensorclass, batch_dim, layer_dim


@checked_tensorclass
class Layer:
    object: Object
    scale: TensorType[batch_dim, layer_dim, 1, torch.float32]
    composition: TensorType[batch_dim, layer_dim, 1, torch.uint8]
