import torch
import torch.nn as nn
from lcmr.encoder import Encoder
from lcmr.modeler import Modeler
from lcmr.renderer import Renderer


class ReconstructionModel(nn.Module):
    def __init__(self, encoder: Encoder, modeler: Modeler, renderer: Renderer):
        self.encoder = encoder
        self.modeler = modeler
        self.renderer = renderer

    def forward(x: torch.Tensor):
        pass
