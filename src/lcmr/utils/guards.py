import torch
from tensordict.prototype import tensorclass
from torchtyping import patch_typeguard, TensorType
from typeguard import typechecked
from dataclasses import dataclass
from typing import NewType, TypeVar

patch_typeguard()

Type = TypeVar('Type')

def checked_constructor(cls: Type) -> Type:
    if hasattr(cls, "__init__"):
        cls.__init__ = typechecked(cls.__init__)
    return cls


def checked_tensorclass(cls: Type) -> Type:
    return checked_constructor(tensorclass(cls))


def checked_dataclass(cls: Type) -> Type:
    return checked_constructor(dataclass(cls))


batch_dim = "batch_dim"
layer_dim = "layer_dim"
object_dim = "object_dim"

height_dim = "height_dim"
width_dim = "width_dim"

reduced_height_dim = "reduced_height_dim"
reduced_width_dim = "reduced_width_dim"

optional_dims = "optional_dims"

vec_dim = "vec_dim"
channel_dim = "channel_dim"
channel_dim_2 = "channel_dim_2"

grid_width = "grid_width"
grid_height = "grid_height"

ImageBHWC4 = NewType("ImageBHWC4", TensorType[batch_dim, height_dim, width_dim, 4, torch.float32])
ImageBHWC3 = NewType("ImageBHWC3", TensorType[batch_dim, height_dim, width_dim, 3, torch.float32])
ImageHWC4 = NewType("ImageHWC4", TensorType[height_dim, width_dim, 4, torch.float32])