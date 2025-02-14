import random
import torch.nn as nn

from lcmr.utils.guards import typechecked

@typechecked
def random_detach(*modules: nn.Module, p: float = 0.2):
    for module in modules:
        module.requires_grad_(random.choices([False, True], weights=[p, 1 - p])[0])