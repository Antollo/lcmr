import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import Sequence, Type, Callable
from itertools import chain

from lcmr.encoder import Encoder
from lcmr.utils.guards import typechecked, ImageBHWC3, batch_dim, reduced_height_dim, reduced_width_dim, channel_dim


@typechecked
class ConvNetEncoder(Encoder):
    def __init__(
        self,
        in_features: int = 3,
        n_channels: Sequence[int] = None,
        avg_pool: Sequence[bool] = None,
        kernel_size: int = 3,
        Activation: Type[nn.Module] | Callable = nn.GELU,
    ):
        super().__init__()

        assert len(n_channels) == len(avg_pool)

        self.model = nn.Sequential(
            nn.Conv2d(in_features, n_channels[0], kernel_size, padding="same"),
            Activation(),
            nn.AvgPool2d(2) if avg_pool[0] else nn.Identity(),
            nn.BatchNorm2d(n_channels[0]),
            *chain(
                *(
                    (
                        nn.Conv2d(prev, next, kernel_size, padding="same"),
                        Activation(),
                        nn.AvgPool2d(2) if pool else nn.Identity(),
                        nn.BatchNorm2d(next),
                    )
                    for (prev, next, pool) in zip(n_channels, n_channels[1:], avg_pool[1:])
                )
            )
        )

    def forward(self, x: ImageBHWC3) -> TensorType[batch_dim, channel_dim, reduced_height_dim, reduced_width_dim, torch.float32]:
        # BHWC to BCHW
        x = x.permute(0, 3, 1, 2)
        return self.model(x)
