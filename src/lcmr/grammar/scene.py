from .guards import checked_tensorclass
from .layer import Layer


@checked_tensorclass
class Scene:
    layer: Layer
