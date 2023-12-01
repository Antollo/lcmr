import abc
import torch
from torchtyping import TensorType

from ...grammar import Scene
from ...grammar.guards import batch_dim
from ..renderer import Renderer

raster_dim = "raster_dim"


class Renderer2D(Renderer):
    def __init__(self, raster_size: tuple[int, int]):
        self.raster_size = raster_size

    @abc.abstractmethod
    def render(scene: Scene) -> TensorType[batch_dim, raster_dim, raster_dim, 4, torch.float32]:
        pass
