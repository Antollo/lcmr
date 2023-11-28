from typing import List
from tensordict.prototype import tensorclass
from lcmr.grammar.blend_modes import BlendModes
from lcmr.grammar.object_group import ObjectGroup


@tensorclass
class Layer:
    groups: List[ObjectGroup]
    scale: float
    composition: BlendModes

