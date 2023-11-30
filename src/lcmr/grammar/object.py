from typing import Union
import torch
from torchtyping import TensorType
from .guards import checked_tensorclass, batch_dim, layer_dim, object_dim
from .appearance import Appearance
from .transformations import Identity, Translation, Affine


@checked_tensorclass
class Object:
    objectShape: TensorType[batch_dim, layer_dim, object_dim, 1, torch.uint8]
    transformation: Union[Identity, Translation, Affine]
    appearance: Appearance
