import abc

import torch
from torchtyping import TensorType

from lcmr.grammar import Scene
from lcmr.grammar.scene_data import SceneData
from lcmr.renderer.renderer import Renderer
from lcmr.utils.guards import height_dim, optional_dims, width_dim


class Renderer2D(Renderer):
    def __init__(self, raster_size: tuple[int, int], device: torch.device = torch.device("cpu")):
        self.raster_size = raster_size
        self.device = device

    @abc.abstractmethod
    def render(self, scene: Scene) -> SceneData:
        pass

    @staticmethod
    def alpha_compositing(
        src: TensorType[optional_dims:..., height_dim, width_dim, 4, torch.float32],
        dst: TensorType[optional_dims:..., height_dim, width_dim, 4, torch.float32],
    ) -> TensorType[optional_dims:..., height_dim, width_dim, 4, torch.float32]:
        # https://stackoverflow.com/a/60401248/14344875

        src_alpha = src[..., 3, None]
        dst_alpha = dst[..., 3, None]
        out_alpha = src_alpha + dst_alpha * (1 - src_alpha)

        src_rgb = src[..., :3]
        dst_rgb = dst[..., :3]
        out_rgb = (src_rgb * src_alpha + dst_rgb * dst_alpha * (1 - src_alpha)) / torch.clamp(out_alpha, min=0.0001)

        return torch.cat((out_rgb, out_alpha), dim=-1)
