from .transformation import Transformation
from lcmr.grammar.shapes.shape import Shape


class Identity(Transformation):
    def apply(shape: Shape) -> Shape:
        return shape