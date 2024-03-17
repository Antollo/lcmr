import torch
from typing import Type, Optional
from torchtyping import TensorType
from dataclasses import dataclass

from lcmr.utils.guards import typechecked
from lcmr.renderer.renderer2d import Renderer2D
from lcmr.renderer.renderer2d import OpenGLRenderer2D
from lcmr.utils.colors import colors
from lcmr.utils.fourier_shape_descriptors import FourierDescriptorsGeneratorOptions


@typechecked
@dataclass
class DatasetOptions:
    name: str = ""
    split: str = "train"
    n_samples: int = 1
    n_objects: int = 1
    return_images: bool = True
    return_scenes: bool = True
    raster_size: tuple[int, int] = (128, 128)
    background_color: TensorType[4, torch.float32] = colors.black
    Renderer: Type[Renderer2D] = OpenGLRenderer2D
    renderer_device: torch.device = torch.device("cpu")
    n_jobs: int = 1
    fourier_shapes_options: Optional[FourierDescriptorsGeneratorOptions] = None
