from typing import List
from tensordict.prototype import tensorclass
from .object import Object


@tensorclass
class ObjectGroup:
    objects: List[Object]
