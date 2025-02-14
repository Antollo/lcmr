from dataclasses import dataclass
from typing import Optional, Type

import torch
from torchtyping import TensorType

from lcmr.renderer.renderer2d import OpenGLRenderer2D, Renderer2D
from lcmr.utils.elliptic_fourier_descriptors import EfdGeneratorOptions
from lcmr.utils.guards import typechecked


@typechecked
@dataclass
class DatasetOptions:
    name: str = ""
    split: str = "train"
    seed: int = 123
    n_samples: int = 1
    n_objects: int = 1
    background_color: Optional[TensorType[4, torch.float32]] = None
    efd_options: Optional[EfdGeneratorOptions] = None
    use_single_scale: bool = True
    raster_size: tuple[int, int] = (128, 128)
    return_images: bool = True
    return_scenes: bool = True
    Renderer: Type[Renderer2D] = OpenGLRenderer2D
    renderer_device: torch.device = torch.device("cpu")
    renderer_batch_size: int = 128
    n_jobs: int = 1
    use_cache: bool = False
    cache_filename: str = ""
    verbose: bool = False