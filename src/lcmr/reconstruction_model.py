import torch
import torch.nn as nn
from torchtyping import TensorType

from lcmr.encoder import Encoder
from lcmr.grammar.scene_data import SceneData
from lcmr.modeler import Modeler
from lcmr.renderer import Renderer
from lcmr.utils.guards import batch_dim, channel_dim, height_dim, typechecked, width_dim


@typechecked
class ReconstructionModel(nn.Module):
    def __init__(self, encoder: Encoder, modeler: Modeler, renderer: Renderer, frozen_encoder: bool = True):
        super().__init__()

        self.encoder = encoder
        self.modeler = modeler
        self.renderer = renderer
        self.frozen_encoder = frozen_encoder

    def forward(self, x: TensorType[batch_dim, height_dim, width_dim, channel_dim, torch.float32], render: bool = True) -> SceneData:

        with torch.set_grad_enabled(not self.frozen_encoder):
            emb = self.encoder(x)

        scene = self.modeler(emb)
        if render:
            scene_data = self.renderer.render(scene)
        else:
            scene_data = SceneData(scene=scene, batch_size=scene.shape)
        return scene_data
