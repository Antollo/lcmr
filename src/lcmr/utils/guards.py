from dataclasses import dataclass
from typing import Any, Generic, NewType, TypeVar, Union

import torch
from tensordict.prototype import tensorclass
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing_extensions import Self

# typechecked = lambda x: x

if __debug__:
    patch_typeguard()

Type = TypeVar("Type")


class SelfIndexable(Generic[Type]):
    def __getitem__(self, *args: Any) -> Union[Type, Self]: ...
    def to(self, *args: Any) -> Union[Type, Self]: ...
    def clone(self) -> Union[Type, Self]: ...


def checked_constructor(cls: Type) -> Type:
    if hasattr(cls, "__init__"):
        cls.__init__ = typechecked(cls.__init__)
    return cls


def checked_tensorclass(cls: Type) -> Union[type[Type], type[SelfIndexable[Type]]]:
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

ImageBHWC = NewType("ImageBHWC", TensorType[batch_dim, height_dim, width_dim, -1, torch.float32])
ImageBHWC4 = NewType("ImageBHWC4", TensorType[batch_dim, height_dim, width_dim, 4, torch.float32])
ImageBHWC3 = NewType("ImageBHWC3", TensorType[batch_dim, height_dim, width_dim, 3, torch.float32])
ImageHWC = NewType("ImageHWC", TensorType[height_dim, width_dim, channel_dim, torch.float32])
ImageHWC4 = NewType("ImageHWC4", TensorType[height_dim, width_dim, 4, torch.float32])
