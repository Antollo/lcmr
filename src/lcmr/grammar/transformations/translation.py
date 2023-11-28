from typing import Tuple
from .transformation import Transformation
from lcmr.grammar.shapes.shape import Shape


class Translation(Transformation):
    def __init__(self, x: Tuple[float]):
        self.x = x

    def apply(shape: Shape) -> Shape:
        return NotImplementedError
