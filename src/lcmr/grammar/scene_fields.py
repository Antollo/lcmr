from dataclasses import dataclass
from typing import Optional, Union

import torch
from torchtyping import TensorType

from lcmr.utils.guards import batch_dim, layer_dim, object_dim, typechecked


@typechecked
@dataclass
class SceneFields:
    translation: Optional[TensorType[batch_dim, layer_dim, object_dim, 2, torch.float32]] = None
    scale: Optional[TensorType[batch_dim, layer_dim, object_dim, 2, torch.float32]] = None
    angle: Optional[TensorType[batch_dim, layer_dim, object_dim, 1, torch.float32]] = None
    rotation_vec: Optional[TensorType[batch_dim, layer_dim, object_dim, 2, torch.float32]] = None
    color: Optional[TensorType[batch_dim, layer_dim, object_dim, 3, torch.float32]] = None
    confidence: Optional[TensorType[batch_dim, layer_dim, object_dim, 1, torch.float32]] = None
    efd: Optional[TensorType[batch_dim, layer_dim, object_dim, -1, 4, torch.float32]] = None
    background_color: Optional[TensorType[batch_dim, 3, torch.float32]] = None

    def __getitem__(self, fields: str) -> Union[list[Optional[torch.Tensor]], Optional[torch.Tensor]]:
        SceneFields.fields: dict[str, str]
        arr = [self.__dict__[SceneFields.fields[id]] for id in fields if id in SceneFields.fields.keys()]
        return arr[0] if len(arr) == 1 else arr


SceneFields.fields = dict(t="translation", s="scale", a="angle", r="rotation_vec", c="color", f="confidence", e="efd", b="background_color")
