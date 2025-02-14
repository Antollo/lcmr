from typing import Optional

import torch

from lcmr.grammar.scene import Scene
from lcmr.utils.guards import ImageBHWC, checked_tensorclass


@checked_tensorclass
class SceneData:
    scene: Optional[Scene] = None
    image: Optional[ImageBHWC] = None
    mask: Optional[ImageBHWC] = None
    extra: Optional[torch.tensor] = None
    
    @property
    def image_rgb(self) -> ImageBHWC:
        return self.image[..., :3]
    
    @property
    def image_top(self, n: int = 8) -> ImageBHWC:
        return self.image[:n, ..., :3]
    
    @property
    def image_rgb_top(self, n: int = 8) -> ImageBHWC:
        return self.image_rgb[:n, ..., :3]
