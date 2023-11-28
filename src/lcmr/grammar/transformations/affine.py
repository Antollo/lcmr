from typing import Tuple
from .transformation import Transformation
from lcmr.grammar.shapes.shape import Shape


class Affine(Transformation):
    def __init__(self, x: Tuple[float], y: Tuple[float]):
        self.x = x
        self.y = y

    def apply(shape: Shape) -> Shape:
        return NotImplementedError
