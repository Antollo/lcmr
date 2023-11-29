from tensordict.prototype import tensorclass
from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()


def checked_constructor(cls: type):
    if hasattr(cls, "__init__"):
        cls.__init__ = typechecked(cls.__init__)
    return cls


def checked_tensorclass(cls: type):
    return checked_constructor(tensorclass(cls))


batch_dim = "batch_dim"
layer_dim = "layer_dim"
object_dim = "object_dim"
