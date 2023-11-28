from tensordict.prototype import tensorclass
from .appearance import Appearance
from .transformations import Transformation
from .shapes import Shape


@tensorclass
class Object:
    shape: Shape
    transformation: Transformation
    appearance: Appearance
