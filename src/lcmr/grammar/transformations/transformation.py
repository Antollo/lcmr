import abc
from lcmr.grammar.shapes.shape import Shape


class Transformation(abc.ABC):
    @abc.abstractmethod
    def apply(shape: Shape) -> Shape:
        pass