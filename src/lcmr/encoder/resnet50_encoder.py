import torch
import torch.nn as nn
from torchtyping import TensorType
from torchvision.models import resnet50, ResNet50_Weights
from typing import Optional

from lcmr.encoder import PretrainedEncoder
from lcmr.utils.guards import typechecked, ImageBHWC3, batch_dim, reduced_height_dim, reduced_width_dim


@typechecked
class ResNet50Encoder(PretrainedEncoder):
    def __init__(self, input_size: Optional[tuple[int, int]] = None, replace_stride_with_dilation: list[bool, bool, bool] = [False, False, False]):

        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights, replace_stride_with_dilation=replace_stride_with_dilation)
        model = nn.Sequential(*list(model.children())[:-2])

        super().__init__(model, input_size=input_size)

    def forward(self, x: ImageBHWC3) -> TensorType[batch_dim, 2048, reduced_height_dim, reduced_width_dim, torch.float32]:
        return super().forward(x)
