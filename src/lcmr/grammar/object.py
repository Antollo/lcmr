from typing import Union
import torch
from torchtyping import TensorType
from ..utils.guards import checked_tensorclass, batch_dim, layer_dim, object_dim
from .appearance import Appearance
from .transformations import Identity, Translation, Affine, LazyAffine


@checked_tensorclass
class Object:
    objectShape: TensorType[batch_dim, layer_dim, object_dim, 1, torch.uint8]
    transformation: Union[Identity, Translation, Affine, LazyAffine]
    appearance: Appearance
