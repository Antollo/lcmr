import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import Union

from lcmr.encoder import Encoder
from lcmr.modeler import Modeler
from lcmr.renderer import Renderer
from lcmr.grammar import Scene
from lcmr.utils.guards import typechecked, batch_dim, height_dim, width_dim, channel_dim, channel_dim_2


@typechecked
class ReconstructionModel(nn.Module):
    def __init__(self, encoder: Encoder, modeler: Modeler, renderer: Renderer, return_images: bool = True):
        super().__init__()

        self.encoder = encoder
        self.modeler = modeler
        self.renderer = renderer
        self.return_images = return_images

    def forward(self, x: TensorType[batch_dim, height_dim, width_dim, channel_dim, torch.float32]) -> Union[
        Scene,
        TensorType[batch_dim, height_dim, width_dim, channel_dim_2, torch.float32],
        tuple[Scene, TensorType[batch_dim, height_dim, width_dim, channel_dim_2, torch.float32]],
    ]:
        emb = self.encoder(x)
        scene = self.modeler(emb)
        if self.return_images:
            img = self.renderer.render(scene)
            return scene, img
        else:
            return scene
