from typing import List
from tensordict.prototype import tensorclass
from .layer import Layer


@tensorclass
class Scene:
    layers: List[Layer]
