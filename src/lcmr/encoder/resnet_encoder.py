from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torchtyping import TensorType
from torchvision.models import (
    ResNet,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    WeightsEnum,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)

from lcmr.encoder import PretrainedEncoder
from lcmr.utils.guards import ImageBHWC3, batch_dim, reduced_height_dim, reduced_width_dim, typechecked


@typechecked
class ResNet50Encoder(PretrainedEncoder):
    def __init__(
        self,
        resnet: Callable[..., ResNet],
        weights: WeightsEnum,
        input_size: Optional[tuple[int, int]] = None,
        replace_stride_with_dilation: list[bool, bool, bool] = [False, False, False],
        frozen: bool = True,
    ):
        model = resnet(weights=weights, replace_stride_with_dilation=replace_stride_with_dilation)
        model = nn.Sequential(*list(model.children())[:-2])
        super().__init__(model, input_size=input_size, frozen=frozen)

    def forward(self, x: ImageBHWC3) -> TensorType[batch_dim, -1, reduced_height_dim, reduced_width_dim, torch.float32]:
        return super().forward(x)


@typechecked
class ResNet18Encoder(ResNet50Encoder):
    def __init__(self, **kwargs: Any):
        super().__init__(resnet=resnet18, weights=ResNet18_Weights.DEFAULT, **kwargs)


@typechecked
class ResNet34Encoder(ResNet50Encoder):
    def __init__(self, **kwargs: Any):
        super().__init__(resnet=resnet34, weights=ResNet34_Weights.DEFAULT, **kwargs)


@typechecked
class ResNet50Encoder(ResNet50Encoder):
    def __init__(self, **kwargs: Any):
        super().__init__(resnet=resnet50, weights=ResNet50_Weights.DEFAULT, **kwargs)


@typechecked
class ResNet101Encoder(ResNet50Encoder):
    def __init__(self, **kwargs: Any):
        super().__init__(resnet=resnet101, weights=ResNet101_Weights.DEFAULT, **kwargs)


@typechecked
class ResNet152Encoder(ResNet50Encoder):
    def __init__(self, **kwargs: Any):
        super().__init__(resnet=resnet152, weights=ResNet152_Weights.DEFAULT, **kwargs)
