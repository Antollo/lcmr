import abc
import torch
from lcmr.grammar import Scene


class Renderer(abc.ABC):
    @abc.abstractmethod
    def render(scene: Scene) -> torch.Tensor:
        pass
